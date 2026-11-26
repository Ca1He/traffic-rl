"""Unit tests for the environment package.  Run:  python -m pytest -q tests"""
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import load_config, make_env, run_episode  # noqa: E402
from envs.metrics import episode_metrics, jain_index, read_tripinfo  # noqa: E402
from envs.wrappers import FairnessCostWrapper  # noqa: E402


# ----------------------------------------------------------------------------- metrics
def test_jain_index_bounds():
    assert jain_index([5, 5, 5, 5]) == pytest.approx(1.0)
    assert jain_index([1, 0, 0, 0]) == pytest.approx(0.25)  # worst case = 1/n
    assert jain_index([0, 0]) == 1.0


def test_read_tripinfo_handles_unfinished_and_undeparted():
    xml = """<tripinfos>
      <tripinfo id="f0_N_through.0" depart="0.00" departLane="n_t_0" departDelay="2.00" arrival="40.00"
                duration="40.00" waitingTime="10.00" timeLoss="12.00"/>
      <tripinfo id="f0_E_left.0" depart="5.00" departLane="e_t_1" departDelay="0.00" arrival="-1.00"
                duration="100.00" waitingTime="90.00" timeLoss="95.00"/>
      <tripinfo id="f0_E_left.1" depart="-1" departLane="" departDelay="300.00" arrival="-1.00"
                duration="0.00" waitingTime="0.00" timeLoss="0.00"/>
    </tripinfos>"""
    path = os.path.join(tempfile.mkdtemp(), "t.xml")
    with open(path, "w") as fh:
        fh.write(xml)
    df = read_tripinfo(path)
    assert list(df["approach"]) == ["N", "E", "E"]
    assert list(df["delay"]) == [12.0, 90.0, 300.0]          # wait + departDelay
    m = episode_metrics(df, horizon=3600)
    assert m["n_vehicles"] == 3 and m["unfinished"] == 2 and m["undeparted"] == 1
    assert m["max_delay"] == 300.0
    assert m["delay_E"] == pytest.approx(195.0) and m["delay_N"] == pytest.approx(12.0)
    assert m["throughput_vph"] == pytest.approx(1.0)


# ----------------------------------------------------------------------------- cost
def test_cost_definitions():
    w = FairnessCostWrapper.__new__(FairnessCostWrapper)  # cost() only needs threshold/kind
    w.threshold, w.kind = 90.0, "indicator"
    assert w.cost(89.0) == 0.0 and w.cost(91.0) == 1.0
    w.kind = "hinge"
    assert w.cost(45.0) == 0.0 and w.cost(180.0) == pytest.approx(1.0)


# ----------------------------------------------------------------------------- simulation
@pytest.fixture(scope="module")
def cfg():
    return load_config(overrides={"env": {"scenario": "asym", "num_seconds": 600}})


def test_max_green_is_enforced(cfg):
    env = make_env(cfg, seed=1)
    env.reset(seed=1)
    ts = env.unwrapped.traffic_signals[env.unwrapped.ts_ids[0]]
    longest, done = 0, False
    while not done:
        _, _, t, tr, info = env.step(ts.green_phase)  # always ask to keep the current phase
        longest = max(longest, ts.time_since_last_phase_change - ts.yellow_time)
        done = t or tr
    env.close()
    assert info["forced_switches"] > 0
    assert longest <= cfg["env"]["max_green"] + cfg["env"]["delta_time"]


def test_observation_shape_and_range(cfg):
    env = make_env(cfg, seed=0)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    env.close()


def test_full_dropout_zeroes_detector_readings(cfg):
    c = load_config(overrides={"env": {"scenario": "asym", "num_seconds": 300}, "sensing": {"dropout_p": 1.0}})
    env = make_env(c, seed=0)
    env.reset(seed=0)
    for _ in range(20):
        obs, *_ = env.step(0)
    n_phase = env.unwrapped.traffic_signals["t"].num_green_phases
    assert np.all(obs[n_phase + 1:] == 0.0)          # density + queue all zero
    assert obs[:n_phase].sum() == 1.0               # phase one-hot is not a sensor -> intact
    env.close()


def test_same_seed_is_reproducible(cfg):
    pol = lambda o, e: 0  # noqa: E731
    m1 = run_episode(make_env(cfg, seed=3), pol, seed=3)
    m2 = run_episode(make_env(cfg, seed=3), pol, seed=3)
    assert m1["mean_delay"] == m2["mean_delay"] and m1["n_vehicles"] == m2["n_vehicles"]
