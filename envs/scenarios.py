"""Demand scenarios (SUMO route files) for the single 4-way intersection.

Arrivals are Bernoulli per second (SUMO ``probability`` flows), i.e. approximately
Poisson, so different SUMO seeds give different traffic realisations.

Approaches: N, S (main road) and E, W (side road). Each approach splits its demand
into through / left / right movements with fixed turning ratios.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

NET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nets", "single")
NET_FILE = os.path.join(NET_DIR, "intersection.net.xml")

APPROACHES = ("N", "S", "E", "W")
IN_EDGE = {"N": "n_t", "S": "s_t", "E": "e_t", "W": "w_t"}
EDGE_TO_APPROACH = {v: k for k, v in IN_EDGE.items()}
OUT_EDGE = {"N": "t_n", "S": "t_s", "E": "t_e", "W": "t_w"}
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
# Right-hand traffic: arriving from N (heading south), a right turn exits to W, a left turn to E.
RIGHT_OF = {"N": "W", "S": "E", "E": "N", "W": "S"}
LEFT_OF = {"N": "E", "S": "W", "E": "S", "W": "N"}
TURN_RATIOS = {"through": 0.6, "left": 0.2, "right": 0.2}


@dataclass
class Period:
    """A time window with a constant demand (veh/h) per approach."""

    begin: int
    end: int
    vph: Dict[str, float]


# ---------------------------------------------------------------------------
# Named scenarios (veh/h per approach).  Episode length is 3600 s.
# ---------------------------------------------------------------------------
SCENARIOS: Dict[str, List[Period]] = {
    # Symmetric, moderate load.
    "balanced": [Period(0, 3600, {"N": 500, "S": 500, "E": 500, "W": 500})],
    # Main road : side road = 4 : 1  -> exposes starvation of the side road.
    "asym": [Period(0, 3600, {"N": 800, "S": 800, "E": 200, "W": 200})],
    # Distribution shift: asym, then a 20-min side-road surge, then asym again.
    "surge": [
        Period(0, 1200, {"N": 800, "S": 800, "E": 200, "W": 200}),
        Period(1200, 2400, {"N": 500, "S": 500, "E": 700, "W": 700}),
        Period(2400, 3600, {"N": 800, "S": 800, "E": 200, "W": 200}),
    ],
}


def _movements(approach: str) -> Dict[str, str]:
    """Route edges for each movement of an approach."""
    return {
        "through": f"{IN_EDGE[approach]} {OUT_EDGE[OPPOSITE[approach]]}",
        "left": f"{IN_EDGE[approach]} {OUT_EDGE[LEFT_OF[approach]]}",
        "right": f"{IN_EDGE[approach]} {OUT_EDGE[RIGHT_OF[approach]]}",
    }


def write_routes(periods: Sequence[Period], path: str) -> str:
    """Write a SUMO route file for the given demand periods."""
    lines = ["<routes>", '    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="13.89"/>']
    for a in APPROACHES:
        for mv, edges in _movements(a).items():
            lines.append(f'    <route id="r_{a}_{mv}" edges="{edges}"/>')
    flows = []
    for k, p in enumerate(periods):
        for a in APPROACHES:
            for mv, ratio in TURN_RATIOS.items():
                prob = p.vph[a] * ratio / 3600.0
                if prob <= 0:
                    continue
                flows.append(
                    (p.begin, f'    <flow id="f{k}_{a}_{mv}" type="car" route="r_{a}_{mv}" begin="{p.begin}" '
                     f'end="{p.end}" probability="{prob:.6f}" departSpeed="max" departLane="best"/>')
                )
    lines += [f for _, f in sorted(flows, key=lambda x: x[0])]  # SUMO requires flows sorted by begin
    lines.append("</routes>")
    content = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            if fh.read() == content:
                return path  # identical file already there (parallel jobs share route files)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(content)
    for attempt in range(20):
        try:
            os.replace(tmp, path)  # atomic: readers never see a half-written file
            return path
        except PermissionError:  # Windows: another process has the file open -> wait and retry
            time.sleep(0.2 * (attempt + 1))
            with open(path, encoding="utf-8") as fh:
                if fh.read() == content:  # someone else already wrote the same content
                    os.remove(tmp)
                    return path
    raise PermissionError(f"could not replace {path}")


def route_file(scenario: str) -> str:
    """Path of the route file for a named scenario (generated on demand)."""
    if scenario not in SCENARIOS:
        raise KeyError(f"Unknown scenario '{scenario}'. Options: {list(SCENARIOS)}")
    path = os.path.join(NET_DIR, f"{scenario}.rou.xml")
    if not os.path.exists(path):
        write_routes(SCENARIOS[scenario], path)
    return path


def random_route_pool(n: int = 50, low: float = 150, high: float = 900, seed: int = 12345,
                      num_seconds: int = 3600) -> List[str]:
    """Pool of route files with per-approach demand ~ U[low, high] veh/h (for domain randomization).

    Half of the files also contain a mid-episode demand change, so the randomized
    training distribution covers non-stationary traffic as well.
    """
    rng = np.random.default_rng(seed)
    out_dir = os.path.join(NET_DIR, "random")
    paths = []
    for i in range(n):
        path = os.path.join(out_dir, f"rand_{seed}_{i:03d}.rou.xml")
        draw = lambda: {a: float(rng.uniform(low, high)) for a in APPROACHES}  # noqa: E731
        if i % 2 == 0:
            periods = [Period(0, num_seconds, draw())]
        else:
            t = int(rng.integers(num_seconds // 4, 3 * num_seconds // 4))
            periods = [Period(0, t, draw()), Period(t, num_seconds, draw())]
        write_routes(periods, path)  # cheap; always rewrite so files match the seed exactly
        paths.append(path)
    return paths


if __name__ == "__main__":
    for name in SCENARIOS:
        print(name, "->", write_routes(SCENARIOS[name], os.path.join(NET_DIR, f"{name}.rou.xml")))
    print(len(random_route_pool()), "randomized route files")
