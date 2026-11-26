"""Empirical check of Proposition 1: as side-road demand lambda_2 shrinks (main road fixed),
an efficiency-oriented controller (Max-Pressure) lets the side-road maximum wait grow, while the
overall mean delay stays low.  Proposition 1 predicts the max side-road wait ~ lambda_2^(-1/2).

    python scripts/proposition_sweep.py          -> results/proposition_sweep.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from agents.baselines import MaxPressure, Webster  # noqa: E402
from envs import load_config, make_env, run_episode  # noqa: E402
from envs.metrics import read_tripinfo  # noqa: E402
from envs.scenarios import SCENARIOS, Period  # noqa: E402

MAIN = 800
SIDE = [25, 50, 100, 200, 400]
SEEDS = range(1000, 1005)


def main():
    rows = []
    for side in SIDE:
        name = f"sweep_side{side}"
        vph = {"N": MAIN, "S": MAIN, "E": side, "W": side}
        SCENARIOS[name] = [Period(0, 3600, vph)]
        cfg = load_config(overrides={"env": {"scenario": name}})
        for method, ctrl in [("Max-Pressure", MaxPressure()), ("Webster", Webster(vph))]:
            for seed in SEEDS:
                env = make_env(cfg, seed=seed)
                m = run_episode(env, ctrl, seed=seed)
                df = read_tripinfo(env.unwrapped.tripinfo_path)
                side_df = df[df["approach"].isin(["E", "W"])]
                rows.append({"side_vph": side, "method": method, "seed": seed,
                             "side_max_delay": side_df["delay"].max(),
                             "side_mean_delay": side_df["delay"].mean(),
                             "mean_delay": m["mean_delay"], "mean_cost": m["mean_cost"]})
        print(f"side={side} done", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/proposition_sweep.csv", index=False)
    s = df.groupby(["method", "side_vph"])[["side_max_delay", "side_mean_delay", "mean_delay", "mean_cost"]].mean()
    print(s.round(2).to_string())
    for method in ("Max-Pressure", "Webster"):
        g = s.loc[method]
        slope = np.polyfit(np.log(g.index.values), np.log(g["side_max_delay"].values), 1)[0]
        print(f"{method}: log-log slope of side max wait vs side demand = {slope:.2f} (Prop. 1 predicts -0.5)")


if __name__ == "__main__":
    main()
