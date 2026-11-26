"""Non-learning baseline controllers for the single intersection.

All controllers have the signature ``policy(obs, env) -> action`` used by ``envs.run_episode``.
Phase structure of the network (green phases of sumo-rl):
    0: N-S through + right   1: N-S left   2: E-W through + right   3: E-W left
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from envs.scenarios import SCENARIOS, TURN_RATIOS

# Saturation flows measured in SUMO with scripts/measure_saturation.py (veh/h/lane).
SAT_THROUGH = 1630.0
SAT_LEFT = 1120.0  # lower bound: the left lane could not be fully oversaturated


def _ts(env):
    base = env.unwrapped
    return base.traffic_signals[base.ts_ids[0]]


class FixedTime:
    """Demand-agnostic fixed cycle: holds each green phase for a preset time."""

    def __init__(self, greens: Sequence[float] = (30, 10, 30, 10)):
        self.greens = list(greens)

    def __call__(self, obs, env) -> int:
        ts = _ts(env)
        elapsed = ts.time_since_last_phase_change - ts.yellow_time
        if elapsed >= self.greens[ts.green_phase]:
            return (ts.green_phase + 1) % ts.num_green_phases
        return ts.green_phase

    def describe(self) -> str:
        return f"greens={self.greens}"


def webster_greens(vph: Dict[str, float], yellow: float = 3.0, startup_loss: float = 2.0,
                   min_green: float = 10.0, max_green: float = 60.0, max_cycle: float = 150.0
                   ) -> Tuple[List[float], float, float]:
    """Webster (1958) optimal cycle C0 = (1.5 L + 5) / (1 - Y) with green splits proportional to y_i.

    Critical lane flow per phase (lane 0 carries through + right, lane 1 carries left):
        phase 0: max(N, S) * (through + right) / SAT_THROUGH,   phase 1: max(N, S) * left / SAT_LEFT
        phase 2: max(E, W) * (through + right) / SAT_THROUGH,   phase 3: max(E, W) * left / SAT_LEFT
    Returns (greens, cycle, Y).
    """
    tr = TURN_RATIOS["through"] + TURN_RATIOS["right"]
    lt = TURN_RATIOS["left"]
    ns, ew = max(vph["N"], vph["S"]), max(vph["E"], vph["W"])
    y = np.array([ns * tr / SAT_THROUGH, ns * lt / SAT_LEFT, ew * tr / SAT_THROUGH, ew * lt / SAT_LEFT])
    Y = float(y.sum())
    L = 4 * (yellow + startup_loss)  # total lost time per cycle
    cycle = max_cycle if Y >= 0.95 else min(max_cycle, (1.5 * L + 5.0) / (1.0 - Y))
    greens = np.clip((cycle - L) * y / Y, min_green, max_green)
    return [float(g) for g in greens], float(cycle), Y


def scenario_design_demand(scenario: str) -> Dict[str, float]:
    """Time-weighted average demand of a scenario (a fixed plan is designed for one demand level)."""
    periods = SCENARIOS[scenario]
    total = sum(p.end - p.begin for p in periods)
    return {a: sum(p.vph[a] * (p.end - p.begin) for p in periods) / total for a in ("N", "S", "E", "W")}


class Webster(FixedTime):
    """Fixed-time plan tuned to the scenario's (average) demand with Webster's formula."""

    def __init__(self, vph: Dict[str, float], **kw):
        greens, self.cycle, self.Y = webster_greens(vph, **kw)
        super().__init__(greens)

    def describe(self) -> str:
        return f"greens={[round(g, 1) for g in self.greens]} cycle={self.cycle:.0f}s Y={self.Y:.2f}"


class MaxPressure:
    """Max-Pressure (Varaiya 2013): pick the phase with the largest sum over its movements of
    (upstream lane occupancy - downstream lane occupancy).

    It reads the *sensed* (possibly noisy / dropped) detector values stored by the observation
    function, so robustness experiments perturb it exactly like the RL agents.
    The env enforces min-green (switch requests are ignored until it is met) and max-green.
    """

    def __init__(self):
        self._movements = None  # phase -> list of (in_lane, out_lane)

    def _build(self, ts):
        links = ts.sumo.trafficlight.getControlledLinks(ts.id)
        self._movements = []
        for phase in ts.green_phases:
            mv = [(link[0][0], link[0][1]) for i, link in enumerate(links)
                  if link and phase.state[i] in "Gg"]
            self._movements.append(mv)

    def pressures(self, env) -> np.ndarray:
        ts = _ts(env)
        if self._movements is None:
            self._build(ts)
        s = ts.observation_fn.last
        q_in = dict(zip(s["in_lanes"], s["in_density"]))
        q_out = dict(zip(s["out_lanes"], s["out_density"]))
        return np.array([sum(q_in[i] - q_out.get(o, 0.0) for i, o in mv) for mv in self._movements])

    def __call__(self, obs, env) -> int:
        p = self.pressures(env)
        ts = _ts(env)
        best = int(np.argmax(p))
        return ts.green_phase if p[ts.green_phase] >= p[best] else best  # keep current on ties

    def describe(self) -> str:
        return "pressure = in-lane density - out-lane density (sensed)"
