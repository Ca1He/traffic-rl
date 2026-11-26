"""Gymnasium wrappers for the single-agent sumo-rl environment.

* MaxGreenWrapper      -- enforces the max-green limit (sumo-rl ignores ``max_green``).
* FairnessCostWrapper  -- adds the per-step constraint cost and waiting-time stats to ``info``.
* DemandRandomizer     -- samples a new route file at every reset (domain randomization).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import gymnasium as gym
import numpy as np


def _signal(env):
    """The single TrafficSignal object of the (unwrapped) sumo-rl environment."""
    base = env.unwrapped
    return base.traffic_signals[base.ts_ids[0]]


class MaxGreenWrapper(gym.Wrapper):
    """If the current green has lasted >= max_green, force a switch to the next phase."""

    def __init__(self, env, max_green: int):
        super().__init__(env)
        self.max_green = max_green
        self.forced_switches = 0

    def reset(self, **kwargs):
        self.forced_switches = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        ts = _signal(self.env)
        green_elapsed = ts.time_since_last_phase_change - ts.yellow_time
        if int(action) == ts.green_phase and green_elapsed >= self.max_green:
            action = (ts.green_phase + 1) % ts.num_green_phases
            self.forced_switches += 1
        obs, r, term, trunc, info = self.env.step(action)
        info["forced_switches"] = self.forced_switches
        return obs, r, term, trunc, info


class FairnessCostWrapper(gym.Wrapper):
    """Adds the constraint cost c_t to ``info['cost']``.

    Per-vehicle waiting time w_v = accumulated waiting time (s) of a vehicle on an incoming lane.
    With W_t = max_v w_v over vehicles currently on incoming lanes and threshold T:

        indicator:  c_t = 1[W_t > T]           (fraction of decision steps with a violation)
        hinge:      c_t = max(0, W_t - T) / T  (how badly the threshold is exceeded)

    The constrained problem is  max E[sum r_t]  s.t.  E[(1/H) sum c_t] <= d.
    """

    def __init__(self, env, threshold: float = 90.0, kind: str = "indicator"):
        super().__init__(env)
        if kind not in ("indicator", "hinge"):
            raise ValueError("kind must be 'indicator' or 'hinge'")
        self.threshold = float(threshold)
        self.kind = kind

    def _max_wait(self) -> float:
        ts = _signal(self.env)
        sumo = ts.sumo
        waits = [
            sumo.vehicle.getAccumulatedWaitingTime(v)
            for lane in ts.lanes
            for v in sumo.lane.getLastStepVehicleIDs(lane)
        ]
        return float(max(waits)) if waits else 0.0

    def cost(self, max_wait: float) -> float:
        if self.kind == "indicator":
            return float(max_wait > self.threshold)
        return max(0.0, max_wait - self.threshold) / self.threshold

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        w = self._max_wait()
        info["max_wait_now"] = w
        info["cost"] = self.cost(w)
        return obs, r, term, trunc, info


class DemandRandomizer(gym.Wrapper):
    """Pick a random route file from a pool before every reset (domain randomization)."""

    def __init__(self, env, route_files: Sequence[str], seed: Optional[int] = None):
        super().__init__(env)
        self.route_files: List[str] = list(route_files)
        self.rng = np.random.default_rng(seed)
        self.current_route: Optional[str] = None

    def reset(self, **kwargs):
        self.current_route = self.route_files[int(self.rng.integers(len(self.route_files)))]
        self.env.unwrapped._route = self.current_route  # sumo-rl reads this when restarting SUMO
        return self.env.reset(**kwargs)
