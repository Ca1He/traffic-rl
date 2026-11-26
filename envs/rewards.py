"""Custom reward functions (in addition to sumo-rl's diff-waiting-time / queue / pressure).

fair-shaped: reward-shaping approach to fairness, in the spirit of Raeis & Leon-Garcia (2021).
    r_t = -(1/10) * sum_{v halted on incoming lanes} (1 + alpha * w_v / tau)
where w_v is the accumulated waiting time of vehicle v.  Summed over time, a vehicle that waits
W seconds contributes ~ W + alpha * W^2 / (2 tau), so alpha > 0 penalizes long individual waits
super-linearly.  alpha = 0 recovers the plain (negative) queue-length reward.
This is the "fairness via reward shaping" baseline we compare our explicit constraint against.
"""
from __future__ import annotations

from typing import Callable


def make_fair_shaped_reward(alpha: float = 1.0, tau: float = 60.0) -> Callable:
    def fair_shaped(ts) -> float:
        sumo = ts.sumo
        total = 0.0
        for lane in ts.lanes:
            for v in sumo.lane.getLastStepVehicleIDs(lane):
                if sumo.vehicle.getSpeed(v) < 0.1:  # halted (SUMO's definition)
                    total += 1.0 + alpha * sumo.vehicle.getAccumulatedWaitingTime(v) / tau
        return -total / 10.0

    fair_shaped.__name__ = f"fair-shaped(alpha={alpha})"
    return fair_shaped


def resolve_reward(env_cfg: dict):
    """Map the config's reward name to what sumo-rl accepts (a built-in name or a callable)."""
    name = env_cfg["reward"]
    if name == "fair-shaped":
        return make_fair_shaped_reward(env_cfg.get("fair_alpha", 1.0), env_cfg.get("fair_tau", 60.0))
    if name in ("diff-waiting-time", "queue", "pressure", "average-speed"):
        return name
    raise ValueError(f"Unknown reward '{name}'")
