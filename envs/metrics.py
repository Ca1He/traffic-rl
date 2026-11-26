"""Episode metrics computed from SUMO's per-vehicle tripinfo output.

SUMO is run with ``--tripinfo-output <file> --tripinfo-output.write-unfinished``, so
vehicles still in the network at the end of the episode are included -- otherwise the
most-starved vehicles would silently drop out of the fairness statistics.

``--tripinfo-output.write-undeparted`` additionally records vehicles that never managed to
enter the network (spill-back into the source), with the time they have waited so far.

Per-vehicle delay used for fairness = waitingTime (stopped time in the network)
                                      + departDelay (time spent waiting to enter the network).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .scenarios import APPROACHES, EDGE_TO_APPROACH


def jain_index(x: Iterable[float]) -> float:
    """Jain's fairness index (sum x)^2 / (n * sum x^2); 1 = perfectly fair, 1/n = worst."""
    x = np.asarray(list(x), dtype=float)
    if x.size == 0 or np.all(x == 0):
        return 1.0
    return float(x.sum() ** 2 / (x.size * np.square(x).sum()))


def read_tripinfo(path: str) -> pd.DataFrame:
    """One row per vehicle: approach, depart, arrival, duration, wait, depart_delay, time_loss, finished."""
    rows = []
    for el in ET.parse(path).getroot().iter("tripinfo"):
        vid = el.get("id")
        # Flow ids look like "f<k>_<approach>_<movement>.<n>"; undeparted vehicles have no departLane,
        # so the approach is read from the id, falling back to the departure edge.
        parts = vid.split(".")[0].split("_")
        if len(parts) == 3 and parts[1] in APPROACHES:
            approach = parts[1]
        else:
            edge = el.get("departLane", "").rsplit("_", 1)[0]
            approach = EDGE_TO_APPROACH.get(edge, edge)
        arrival = float(el.get("arrival", -1))
        rows.append(
            {
                "id": vid,
                "approach": approach,
                "departed": float(el.get("depart", -1)) >= 0,
                "depart": float(el.get("depart", -1)),
                "arrival": arrival,
                "finished": arrival >= 0,
                "duration": float(el.get("duration", 0)),
                "wait": float(el.get("waitingTime", 0)),
                "depart_delay": float(el.get("departDelay", 0)),
                "time_loss": float(el.get("timeLoss", 0)),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["delay"] = df["wait"] + df["depart_delay"]
        df["travel_time"] = df["duration"] + df["depart_delay"]
    return df


def episode_metrics(df: pd.DataFrame, horizon: float) -> Dict[str, float]:
    """Aggregate metrics for one episode."""
    if df.empty:
        return {"n_vehicles": 0}
    by_app = df.groupby("approach")["delay"].mean().reindex(list(APPROACHES)).fillna(0.0)
    m = {
        "n_vehicles": int(len(df)),
        "throughput_vph": float(df["finished"].sum() * 3600.0 / horizon),
        "unfinished": int((~df["finished"]).sum()),
        "undeparted": int((~df["departed"]).sum()),
        "mean_travel_time": float(df["travel_time"].mean()),
        "mean_delay": float(df["delay"].mean()),
        "mean_time_loss": float(df["time_loss"].mean()),
        "p95_delay": float(df["delay"].quantile(0.95)),
        "max_delay": float(df["delay"].max()),
        "jain_approach": jain_index(by_app.values),
        "worst_approach_delay": float(by_app.max()),
    }
    for a in APPROACHES:
        m[f"delay_{a}"] = float(by_app[a])
    return m
