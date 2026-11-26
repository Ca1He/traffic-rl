"""Run single-intersection baseline controllers independently at every grid intersection."""
from __future__ import annotations

from .baselines import FixedTime, MaxPressure, Webster


class _OneSignal:
    """Minimal env stand-in exposing a single TrafficSignal the way the baselines expect."""

    def __init__(self, ts):
        self.unwrapped = self
        self.traffic_signals = {ts.id: ts}
        self.ts_ids = [ts.id]


class PerIntersection:
    """policy(obs_array, venv) -> list of actions, one independent controller per intersection."""

    def __init__(self, make_controller):
        self.make_controller = make_controller
        self.ctrls = {}

    def __call__(self, obs, venv):
        acts = []
        for i, t in enumerate(venv.ts_ids):
            ctrl = self.ctrls.setdefault(t, self.make_controller())
            acts.append(ctrl(obs[i], _OneSignal(venv.sumo.traffic_signals[t])))
        return acts


def grid_baselines(vph):
    """vph: per-approach demand used as each intersection's design demand for Webster."""
    return {
        "Fixed-Time": lambda: PerIntersection(FixedTime),
        "Webster": lambda: PerIntersection(lambda: Webster(vph)),
        "Max-Pressure": lambda: PerIntersection(MaxPressure),
    }
