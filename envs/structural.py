"""Structural constraint cost (the MAPPO-LCE-style alternative we compare against).

Instead of constraining the *outcome* (how long vehicles wait), a structural constraint limits
the *signal program*: no green phase may stay red for longer than R_max seconds
("max-red" / phase-skip limit, cf. the GreenSkip / PhaseSkip constraints of Satheesh & Powell 2025).

    c_t = 1[ max_k (time since phase k was last green) > R_max ]

It replaces info['cost'] so the same Lagrangian machinery can be used unchanged.
"""
from __future__ import annotations

import gymnasium as gym


class MaxRedCostWrapper(gym.Wrapper):
    def __init__(self, env, max_red: float = 90.0):
        super().__init__(env)
        self.max_red = float(max_red)
        self._last_green = None

    def _ts(self):
        base = self.env.unwrapped
        return base.traffic_signals[base.ts_ids[0]]

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        ts = self._ts()
        now = self.env.unwrapped.sim_step
        self._last_green = [now] * ts.num_green_phases
        return out

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        ts = self._ts()
        now = self.env.unwrapped.sim_step
        if not ts.is_yellow:
            self._last_green[ts.green_phase] = now
        red = max(now - t for t in self._last_green)
        info["outcome_cost"] = info.get("cost", 0.0)  # keep the max-wait cost for evaluation
        info["max_red_now"] = red
        info["cost"] = float(red > self.max_red)
        return obs, r, term, trunc, info
