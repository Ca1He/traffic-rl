"""Lagrangian PPO (PPO-Lagrangian / RCPO-style) for the max-wait fairness constraint.

Constrained problem (CMDP, Altman 1999):
    max_pi  E[ sum_t r_t ]   s.t.   J_c(pi) = E[ (1/H) sum_t c_t ] <= d
Lagrangian relaxation:  L(pi, lam) = E[ sum_t (r_t - lam * c_t) ] + lam * d * H.

* Primal step: ordinary PPO on the penalized reward r_t - lam * c_t
  (one critic on the penalized reward, as in RCPO, Tessler et al. 2019).
* Dual step: after every rollout, projected gradient ascent on lam,
      lam <- clip(lam + eta * (J_c_hat - d), 0, lam_max),
  where J_c_hat is the mean per-step cost of the rollout.  The dual step runs once per rollout
  (much slower than the many PPO minibatch updates), i.e. the two-timescale scheme of RCPO.

Setting ``fixed_lambda`` turns this into plain reward shaping with a constant penalty, the
"fixed-lambda" ablation (shows why the multiplier must adapt).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


@dataclass
class LagrangeState:
    """Shared between the env wrapper (reads lam) and the callback (updates lam)."""

    lam: float = 0.0
    target: float = 0.05        # d: allowed mean per-step cost
    lr: float = 0.05            # eta: dual step size
    lam_max: float = 20.0
    fixed: Optional[float] = None
    history: List[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.fixed is not None:
            self.lam = float(self.fixed)

    def update(self, mean_cost: float, timesteps: int) -> float:
        if self.fixed is None:
            self.lam = float(np.clip(self.lam + self.lr * (mean_cost - self.target), 0.0, self.lam_max))
        self.history.append({"timesteps": timesteps, "lambda": self.lam, "rollout_cost": mean_cost})
        return self.lam


class PenalizedRewardWrapper(gym.Wrapper):
    """Returns r - lam * c (c from info['cost']); keeps the raw reward in info['raw_reward']."""

    def __init__(self, env, state: LagrangeState):
        super().__init__(env)
        self.state = state

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        info["raw_reward"] = r
        info["lambda"] = self.state.lam
        return obs, r - self.state.lam * info["cost"], term, trunc, info


class LagrangeCallback(BaseCallback):
    """Dual update of lam at the end of every PPO rollout."""

    def __init__(self, state: LagrangeState):
        super().__init__()
        self.state = state
        self._costs: List[float] = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            self._costs.append(info.get("cost", 0.0))
        return True

    def _on_rollout_end(self) -> None:
        if self._costs:
            self.state.update(float(np.mean(self._costs)), self.num_timesteps)
            self._costs = []
