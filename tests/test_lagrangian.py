"""Tests for the Lagrangian machinery and the structural cost."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.lagrangian import LagrangeState, PenalizedRewardWrapper  # noqa: E402
from envs import load_config, make_env, run_episode  # noqa: E402


def test_dual_update_direction_and_projection():
    s = LagrangeState(lam=0.0, target=0.05, lr=1.0, lam_max=2.0)
    assert s.update(0.55, 1) == pytest.approx(0.5)    # violation -> lambda increases
    assert s.update(0.0, 2) == pytest.approx(0.45)    # satisfied -> lambda decreases
    s.update(10.0, 3)
    assert s.lam == 2.0                               # clipped at lam_max
    s.lam = 0.01
    s.update(0.0, 4)
    assert s.lam == 0.0                               # projected onto lambda >= 0


def test_fixed_lambda_never_changes():
    s = LagrangeState(fixed=0.3)
    s.update(1.0, 1)
    assert s.lam == 0.3


def test_penalized_reward():
    cfg = load_config(overrides={"env": {"scenario": "asym", "num_seconds": 200}, "cost": {"threshold": 0}})
    state = LagrangeState(fixed=2.0)
    env = PenalizedRewardWrapper(make_env(cfg, seed=0), state)
    env.reset(seed=0)
    for _ in range(10):
        _, r, _, _, info = env.step(0)
    assert r == pytest.approx(info["raw_reward"] - 2.0 * info["cost"])
    env.close()


def test_structural_cost_reports_outcome_cost_for_evaluation():
    cfg = load_config(overrides={"env": {"scenario": "asym", "num_seconds": 600},
                                 "cost": {"structural_max_red": 30}})
    m = run_episode(make_env(cfg, seed=0), lambda o, e: 0, seed=0)  # phase 0 forever -> others red
    assert 0.0 <= m["mean_cost"] <= 1.0
