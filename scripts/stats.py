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


def grid_table():
    """Table 3: 2x2 grid (local per-intersection violation rate)."""
    import glob
    rows = []
    base = pd.read_csv(os.path.join(RES, "grid_baselines.csv"))
    for m, g in base.groupby("method"):
        for scen, gg in g.groupby("scenario"):
            rows.append({"method": m, "scenario": scen, "run": "-", **gg[[c for c, *_ in COLS]].mean().to_dict()})
    for tag, label in (("grid1500-ippo", "IPPO"), ("gridlocal-lag", "Lagrangian IPPO (ours)")):
        for f in glob.glob(os.path.join(ROOT, "runs", tag, "ippo_asym_s*", "eval.csv")):
            d = pd.read_csv(f)
            for scen, gg in d.groupby("scenario"):
                rows.append({"method": label, "scenario": scen, "run": f, **gg[[c for c, *_ in COLS]].mean().to_dict()})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "grid_all.csv"), index=False)
    order = ["Fixed-Time", "Webster", "Max-Pressure", "IPPO", "Lagrangian IPPO (ours)"]
    nl = r" \\"
    lines = [r"\begin{tabular}{l" + "c" * 2 * len(COLS) + "}", r"\toprule",
             r" & \multicolumn{4}{c}{asym} & \multicolumn{4}{c}{surge}" + nl,
             r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
             "Method & " + " & ".join(c[1] for _ in range(2) for c in COLS) + nl, r"\midrule"]
    for m in order:
        cells = []
        for scen in ("asym", "surge"):
            d = df[(df.method == m) & (df.scenario == scen)]
            for col, _, _, fmt in COLS:
                v, sd = d[col].mean(), d[col].std()
                pm = r"\tiny{$\pm$" + fmt.format(sd) + "}" if len(d) > 1 else ""
                cells.append(fmt.format(v) + pm)
        lines.append(f"{m} & " + " & ".join(cells) + nl)
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB, "grid_table.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("wrote results/tables/grid_table.tex")


ABLATION = [  # (group, method key, label)
    ("Reward", "PPO [main]", r"waiting-time decrease (default)"),
    ("Reward", "PPO [reward-queue]", r"$-$queue length"),
    ("Reward", "PPO [reward-pressure]", r"$-$pressure"),
    ("Shaping", "PPO [reward-fair-shaped]", r"fair-shaped, $\alpha{=}1$"),
    ("Shaping", "PPO [fairshaped-a5]", r"fair-shaped, $\alpha{=}5$"),
    ("Shaping", "PPO [fairshaped-a20]", r"fair-shaped, $\alpha{=}20$"),
    ("Constraint", "PPO [lag-T90]", r"Lagrangian, adaptive $\lambda$ (ours)"),
    ("Constraint", "PPO [fixedlam-0.025]", r"fixed $\lambda{=}0.025$"),
    ("Constraint", "PPO [fixedlam-0.25]", r"fixed $\lambda{=}0.25$"),
    ("Constraint", "PPO [fixedlam-2.5]", r"fixed $\lambda{=}2.5$"),
    ("Constraint", "PPO [struct-R90]", r"structural max-red $R{=}90$"),
    ("Constraint", "PPO [struct-R60]", r"structural max-red $R{=}60$"),
]


def ablation_table(df):
    """Table 2: reward / shaping / constraint ablations on asym (all PPO, violation at T = 90 s)."""
    d = df[df.scenario == "asym"]
    cols = [("mean_delay", "{:.1f}"), ("max_delay", "{:.0f}"), ("mean_cost", "{:.3f}")]
    nl = r" \\"
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r" & Variant & Mean delay (s) & Max delay (s) & Viol. rate" + nl, r"\midrule"]
    last = None
    for group, key, label in ABLATION:
        g = d[d.method == key]
        if g.empty:
            continue
        if last is not None and group != last:
            lines.append(r"\midrule")
        cells = [fmt.format(g[c].mean()) + r"\tiny{$\pm$" + fmt.format(g[c].std()) + "}" for c, fmt in cols]
        lines.append(f"{group if group != last else ''} & {label} & " + " & ".join(cells) + nl)
        last = group
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB, "ablation_table.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("wrote results/tables/ablation_table.tex")


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
    grid_table()
    ablation_table(df)


if __name__ == "__main__":
    main()
