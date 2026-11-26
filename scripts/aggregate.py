"""Collect evaluation results of all runs (+ baselines) into one table.

    python scripts/aggregate.py                      # all tags
    python scripts/aggregate.py --tags main reward-queue --scenarios asym surge

Statistics: each training seed's evaluation episodes are averaged first, then mean ± std is
taken across training seeds (n = number of seeds), as recommended by Henderson et al. (2018).
Baselines have no training seed, so their n = number of evaluation episodes.
Writes results/all_runs.csv (seed-level) and prints the summary table.
"""
import argparse
import glob
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = ["mean_delay", "p95_delay", "max_delay", "worst_approach_delay", "jain_approach",
           "throughput_vph", "mean_cost"]


def load_runs(tags=None) -> pd.DataFrame:
    rows = []
    for path in glob.glob(os.path.join(ROOT, "runs", "*", "*", "eval.csv")):
        tag, name = path.split(os.sep)[-3], path.split(os.sep)[-2]
        if tags and tag not in tags:
            continue
        algo, train_scen, seed = name.rsplit("_", 2)[0], name.rsplit("_", 2)[1], int(name.rsplit("_s", 1)[1])
        df = pd.read_csv(path)
        g = df.groupby("scenario")[METRICS].mean().reset_index()
        g["method"] = f"{algo.upper()} [{tag}]"
        g["train_scenario"], g["train_seed"], g["tag"] = train_scen, seed, tag
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_baselines() -> pd.DataFrame:
    path = os.path.join(ROOT, "results", "baselines.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["train_scenario"], df["train_seed"], df["tag"] = "-", df["seed"], "baseline"
    return df[["scenario", "method", "train_scenario", "train_seed", "tag"] + METRICS]


def summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["scenario", "method", "train_scenario"], sort=True)[METRICS]
    mean, std, n = g.mean(), g.std(), g.size()
    out = pd.DataFrame(index=mean.index)
    for c in METRICS:
        fmt = "{:.3f}±{:.3f}" if c in ("jain_approach", "mean_cost") else "{:.1f}±{:.1f}"
        out[c] = [fmt.format(m, s) for m, s in zip(mean[c], std[c].fillna(0))]
    out["n"] = n
    return out


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*")
    ap.add_argument("--scenarios", nargs="*")
    ap.add_argument("--no-baselines", action="store_true")
    args = ap.parse_args()
    df = load_runs(args.tags)
    if not args.no_baselines:
        df = pd.concat([load_baselines(), df], ignore_index=True)
    if args.scenarios:
        df = df[df["scenario"].isin(args.scenarios)]
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    df.to_csv(os.path.join(ROOT, "results", "all_runs.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_rows", 200)
    print(summary(df).to_string())


if __name__ == "__main__":
    main()
