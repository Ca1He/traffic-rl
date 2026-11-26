"""Significance tests and LaTeX tables for the report.

    python scripts/aggregate.py && python scripts/stats.py

* Welch's t-test (unequal variances) on seed-level means, two-sided, for the key comparisons.
* LaTeX tables written to results/tables/*.tex (booktabs; best value per column in bold).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
TAB = os.path.join(RES, "tables")

COMPARISONS = [  # (scenario, method A, method B, metric)
    ("asym", "PPO [lag-T90]", "PPO [main]", "mean_cost"),
    ("asym", "PPO [lag-T90]", "PPO [main]", "max_delay"),
    ("asym", "PPO [lag-T90]", "PPO [main]", "mean_delay"),
    ("asym", "PPO [lag-T90]", "Max-Pressure", "max_delay"),
    ("asym", "PPO [lag-T90]", "PPO [fairshaped-a5]", "mean_cost"),
    ("asym", "PPO [lag-T90]", "PPO [struct-R90]", "mean_delay"),
    ("surge", "PPO [lag-T90]", "PPO [main]", "mean_cost"),
    ("surge", "PPO [dr800-lag]", "PPO [lag-T90]", "mean_cost"),
    ("surge", "PPO [dr800-lag]", "PPO [lag-T90]", "max_delay"),
    ("asym", "PPO [dr800-lag]", "PPO [lag-T90]", "mean_cost"),
]

MAIN_ROWS = [  # (method key in all_runs.csv, label in the table)
    ("Fixed-Time", "Fixed-Time"),
    ("Webster", "Webster"),
    ("Max-Pressure", "Max-Pressure"),
    ("DQN [main]", "DQN"),
    ("PPO [main]", "PPO"),
    ("PPO [fairshaped-a5]", "PPO + reward shaping ($\\alpha{=}5$)"),
    ("PPO [struct-R90]", "PPO + structural constr. ($R{=}90$)"),
    ("PPO [lag-T90]", "Lagrangian PPO (ours)"),
    ("PPO [dr800-lag]", "DR + Lagrangian PPO (ours)"),
]
COLS = [("mean_delay", "Mean delay (s)", "min", "{:.1f}"), ("max_delay", "Max delay (s)", "min", "{:.0f}"),
        ("jain_approach", "Jain", "max", "{:.2f}"), ("mean_cost", "Viol. rate", "min", "{:.3f}")]


def welch(df):
    rows = []
    for scen, a, b, metric in COMPARISONS:
        xa = df[(df.scenario == scen) & (df.method == a)][metric].dropna()
        xb = df[(df.scenario == scen) & (df.method == b)][metric].dropna()
        if len(xa) < 2 or len(xb) < 2:
            continue
        t, p = stats.ttest_ind(xa, xb, equal_var=False)
        rows.append({"scenario": scen, "A": a, "B": b, "metric": metric, "mean_A": xa.mean(),
                     "mean_B": xb.mean(), "n_A": len(xa), "n_B": len(xb), "t": t, "p": p})
    return pd.DataFrame(rows)


def main_table(df, scenarios=("asym", "surge")):
    """One block per scenario; rows = methods; entries = mean ± std over seeds."""
    lines = ["\\begin{tabular}{l" + "c" * len(COLS) * len(scenarios) + "}", "\\toprule"]
    head1 = " & " + " & ".join(f"\\multicolumn{{{len(COLS)}}}{{c}}{{{s}}}" for s in
                                ["Training demand (asym)", "Demand shift (surge)"][:len(scenarios)]) + " \\\\"
    rules = " ".join(f"\\cmidrule(lr){{{2 + i * len(COLS)}-{1 + (i + 1) * len(COLS)}}}" for i in range(len(scenarios)))
    head2 = "Method & " + " & ".join(c[1] for _ in scenarios for c in COLS) + " \\\\"
    lines += [head1, rules, head2, "\\midrule"]
    stat = {}
    for scen in scenarios:
        d = df[(df.scenario == scen) & (df.train_scenario.isin(["asym", "random", "-"]))]
        g = d.groupby("method")
        stat[scen] = (g.mean(numeric_only=True), g.std(numeric_only=True))
    best = {}
    for scen in scenarios:
        m = stat[scen][0].loc[[k for k, _ in MAIN_ROWS if k in stat[scen][0].index]]
        for col, _, how, _ in COLS:
            best[(scen, col)] = m[col].idxmin() if how == "min" else m[col].idxmax()
    for key, label in MAIN_ROWS:
        if key not in stat[scenarios[0]][0].index:
            continue
        cells = []
        for scen in scenarios:
            mean, std = stat[scen]
            for col, _, _, fmt in COLS:
                if key not in mean.index:
                    cells.append("--")
                    continue
                v = fmt.format(mean.loc[key, col])
                s = std.loc[key, col]
                cell = v + (f"\\tiny{{$\\pm${fmt.format(s)}}}" if not np.isnan(s) else "")
                cells.append(f"\\textbf{{{v}}}" + cell[len(v):] if best[(scen, col)] == key else cell)
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    df = pd.read_csv(os.path.join(RES, "all_runs.csv"))
    os.makedirs(TAB, exist_ok=True)
    w = welch(df)
    w.to_csv(os.path.join(RES, "welch_tests.csv"), index=False)
    pd.set_option("display.width", 200)
    print(w.round(4).to_string(index=False))
    with open(os.path.join(TAB, "main_table.tex"), "w", encoding="utf-8") as fh:
        fh.write(main_table(df))
    print("wrote results/tables/main_table.tex")


if __name__ == "__main__":
    main()
