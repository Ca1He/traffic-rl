"""Tests for the baseline controllers.  Run:  python -m pytest -q tests"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.baselines import MaxPressure, webster_greens  # noqa: E402
from envs import load_config, make_env  # noqa: E402


def test_webster_symmetric_demand_gives_symmetric_plan():
    g, cycle, Y = webster_greens({"N": 500, "S": 500, "E": 500, "W": 500})
    assert g[0] == pytest.approx(g[2]) and g[1] == pytest.approx(g[3])
    assert 0 < Y < 1 and cycle > sum(g)


def test_webster_gives_more_green_to_heavier_road():
    g, _, _ = webster_greens({"N": 800, "S": 800, "E": 200, "W": 200})
    assert g[0] > g[2] and g[1] >= g[3]


def test_webster_oversaturated_caps_cycle():
    _, cycle, Y = webster_greens({"N": 3000, "S": 3000, "E": 3000, "W": 3000})
    assert Y >= 0.95 and cycle == 150.0


def test_max_pressure_serves_the_loaded_approach():
    cfg = load_config(overrides={"env": {"scenario": "asym", "num_seconds": 300}})
    env = make_env(cfg, seed=0)
    env.reset(seed=0)
    for _ in range(12):           # let queues build on phase 2 (E-W) so N-S accumulates
        env.step(2)
    mp = MaxPressure()
    p = mp.pressures(env)
    assert p.shape == (4,)
    assert int(np.argmax(p)) in (0, 1)  # N-S (heavy main road) has the largest pressure
    env.close()
