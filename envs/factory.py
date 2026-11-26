"""Build the full environment from a config dict, and run evaluation episodes."""
from __future__ import annotations

import copy
import os
import tempfile
from typing import Any, Callable, Dict, Optional

import numpy as np
import yaml

os.environ.setdefault("LIBSUMO_AS_TRACI", "1")  # libsumo is ~10x faster than TraCI sockets
import sumo_rl  # noqa: E402

from .metrics import episode_metrics, read_tripinfo  # noqa: E402
from .rewards import resolve_reward  # noqa: E402
from .scenarios import NET_FILE, random_route_pool, route_file  # noqa: E402
from .sensing import make_observation_class  # noqa: E402
from .structural import MaxRedCostWrapper  # noqa: E402
from .wrappers import DemandRandomizer, FairnessCostWrapper, MaxGreenWrapper  # noqa: E402

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: Optional[str] = None, overrides: Optional[dict] = None) -> Dict[str, Any]:
    """Load configs/base.yaml, then an optional experiment yaml, then dict overrides."""
    with open(os.path.join(CONFIG_DIR, "base.yaml"), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if path:
        with open(path, encoding="utf-8") as fh:
            _deep_update(cfg, yaml.safe_load(fh) or {})
    if overrides:
        _deep_update(cfg, copy.deepcopy(overrides))
    return cfg


def make_env(cfg: Dict[str, Any], seed: int = 0, tripinfo_path: Optional[str] = None):
    """Single-intersection Gymnasium env: sumo-rl + noisy sensors + max-green + cost (+ demand randomization)."""
    e, s, c = cfg["env"], cfg["sensing"], cfg["cost"]
    if tripinfo_path is None:
        tripinfo_path = os.path.join(tempfile.mkdtemp(prefix="tsc_"), "tripinfo.xml")
    obs_cls = make_observation_class(s["noise_std"], s["dropout_p"], s["dropout_mode"], seed=seed + 10_000)

    env = sumo_rl.SumoEnvironment(
        net_file=NET_FILE,
        route_file=route_file(e["scenario"]) if e["scenario"] != "random" else route_file("balanced"),
        num_seconds=e["num_seconds"],
        delta_time=e["delta_time"],
        yellow_time=e["yellow_time"],
        min_green=e["min_green"],
        max_green=e["max_green"],
        reward_fn=resolve_reward(e),
        observation_class=obs_cls,
        single_agent=True,
        use_gui=e.get("gui", False),
        sumo_seed=seed,
        sumo_warnings=False,
        add_per_agent_info=False,
        additional_sumo_cmd=f"--tripinfo-output {tripinfo_path} --tripinfo-output.write-unfinished true --tripinfo-output.write-undeparted true",
    )
    env.tripinfo_path = tripinfo_path
    env = MaxGreenWrapper(env, e["max_green"])
    env = FairnessCostWrapper(env, threshold=c["threshold"], kind=c["kind"])
    if c.get("structural_max_red"):  # structural constraint replaces the training cost (MAPPO-LCE style)
        env = MaxRedCostWrapper(env, max_red=c["structural_max_red"])
    if e["scenario"] == "random" or cfg.get("demand_randomization", False):
        env = DemandRandomizer(env, random_route_pool(**cfg.get("random_pool", {})), seed=seed)
    return env


def run_episode(env, policy: Callable[[np.ndarray, Any], int], seed: int) -> Dict[str, float]:
    """Run one full episode with ``policy(obs, env) -> action`` and return its metrics.

    Closes the simulation at the end so SUMO flushes the tripinfo file.
    """
    obs, _ = env.reset(seed=seed)
    done, ret, costs, steps = False, 0.0, [], 0
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs, env))
        ret += r
        costs.append(info.get("outcome_cost", info["cost"]))  # always report the max-wait cost
        steps += 1
        done = term or trunc
    env.close()
    base = env.unwrapped
    m = episode_metrics(read_tripinfo(base.tripinfo_path), horizon=base.sim_max_time - base.begin_time)
    m.update({"return": ret, "mean_cost": float(np.mean(costs)), "steps": steps,  # mean_cost = violation rate
              "forced_switches": info.get("forced_switches", 0)})
    return m
