"""Step 4: evaluate the non-learning baselines (Fixed-Time, Webster, Max-Pressure).

    python scripts/eval_baselines.py                 # all scenarios, 5 evaluation seeds
    python scripts/eval_baselines.py --seeds 10

Writes results/baselines.csv (one row per episode) and prints mean ± std per method/scenario.
Evaluation seeds start at 1000 so they never overlap with training seeds.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from agents.baselines import FixedTime, MaxPressure, Webster, scenario_design_demand  # noqa: E402
from envs import load_config, make_env, run_episode  # noqa: E402

EVAL_SEED0 = 1000
KEY_METRICS = ["mean_delay", "p95_delay", "max_delay", "worst_approach_delay", "jain_approach",
               "throughput_vph", "mean_cost"]


def make_baselines(scenario: str, env_cfg: dict):
    return {
        "Fixed-Time": FixedTime(),
        "Webster": Webster(scenario_design_demand(scenario), yellow=env_cfg["yellow_time"],
                           min_green=env_cfg["min_green"], max_green=env_cfg["max_green"]),
        "Max-Pressure": MaxPressure(),
    }


def summarize(df: pd.DataFrame, metrics=KEY_METRICS) -> pd.DataFrame:
    """'mean ± std' table, rows = (scenario, method)."""
    g = df.groupby(["scenario", "method"], sort=False)[metrics]
    mean, std = g.mean(), g.std()
    out = mean.copy().astype(object)
    for c in metrics:
        fmt = "{:.3f} ± {:.3f}" if c in ("jain_approach", "mean_cost") else "{:.1f} ± {:.1f}"
        out[c] = [fmt.format(m, s) for m, s in zip(mean[c], std[c])]
    return out


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # show ± correctly on Windows consoles
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=["balanced", "asym", "surge"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="results/baselines.csv")
    args = ap.parse_args()

    rows = []
    for scen in args.scenarios:
        cfg = load_config(overrides={"env": {"scenario": scen}})
        for name, ctrl in make_baselines(scen, cfg["env"]).items():
            print(f"[{scen:8s}] {name:12s} {ctrl.describe()}")
            for k in range(args.seeds):
                seed = EVAL_SEED0 + k
                t0 = time.time()
                m = run_episode(make_env(cfg, seed=seed), ctrl, seed=seed)
                m.update(scenario=scen, method=name, seed=seed, wall_s=time.time() - t0)
                rows.append(m)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print()
    print(summarize(df).to_string())
    print(f"\nSaved {len(df)} episodes to {args.out}")


if __name__ == "__main__":
    main()
