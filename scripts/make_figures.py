"""Make all report figures (vector PDF, ICML single-column width).

    python scripts/make_figures.py            -> figures/*.pdf (+ .png previews)

Colors follow the method (never its rank) and are identical in every figure; every series is
also direct-labeled and has its own marker, so nothing relies on color alone.
Palette validated with the dataviz validator (light mode, categorical, adjacent pairs).
"""
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("LIBSUMO_AS_TRACI", "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
COL_W = 3.25  # ICML column width (inches)

STYLE = {  # method -> (color, marker, label)
    "Max-Pressure": ("#1baf7a", "s", "Max-Pressure"),
    "Webster": ("#eda100", "D", "Webster"),
    "PPO": ("#2a78d6", "o", "PPO"),
    "Lag-PPO": ("#eb6834", "^", "Lagrangian PPO (ours)"),
    "DR-Lag-PPO": ("#e87ba4", "v", "DR + Lagrangian PPO (ours)"),
    "DR-PPO": ("#008300", "P", "DR + PPO"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "lines.markersize": 4.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, f"{name}.pdf"))
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=200)
    plt.close(fig)
    print("wrote", name)


def style(key):
    c, m, label = STYLE[key]
    return dict(color=c, marker=m, label=label)


# ------------------------------------------------------------------ Fig. 1 teaser
def fig_teaser(seeds=(1000, 1001, 1002)):
    """ECDF of per-vehicle delay on the side road (E/W), asym scenario."""
    from stable_baselines3 import PPO
    from agents.baselines import MaxPressure
    from envs import load_config, make_env, run_episode
    from envs.metrics import read_tripinfo

    cfg = load_config(overrides={"env": {"scenario": "asym"}})
    methods = {"Max-Pressure": lambda: MaxPressure()}
    for key, tag in (("PPO", "main"), ("Lag-PPO", "lag-T90")):
        path = os.path.join(ROOT, "runs", tag, "ppo_asym_s0", "model.zip")
        if os.path.exists(path):
            model = PPO.load(path, device="cpu")
            methods[key] = (lambda m: (lambda: (lambda o, e: int(m.predict(o, deterministic=True)[0]))))(model)
    delays = {}
    for key, mk in methods.items():
        d = []
        for seed in seeds:
            env = make_env(cfg, seed=seed)
            run_episode(env, mk(), seed=seed)
            df = read_tripinfo(env.unwrapped.tripinfo_path)
            d.append(df.loc[df["approach"].isin(["E", "W"]), "delay"].values)
        delays[key] = np.concatenate(d)

    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    for key, d in delays.items():
        x = np.sort(d)
        y = 1.0 - np.arange(1, len(x) + 1) / len(x)  # survival: share of vehicles waiting longer than x
        c, m, label = STYLE[key]
        ax.step(x, y, where="post", color=c, label=f"{label} (max {x[-1]:.0f} s)")
        ax.plot(x[-1], 1.2e-3, marker=m, color=c, markeredgecolor="white", markeredgewidth=1, clip_on=False)
    ax.axvline(90, color=MUTED, lw=1, ls="--")
    ax.text(93, 0.35, "T = 90 s", color=MUTED, fontsize=6.5)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1.05)
    ax.set_xlabel("Side-road vehicle delay (s)")
    ax.set_ylabel("Share of vehicles\nwaiting longer")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1)
    save(fig, "fig1_teaser")


# ------------------------------------------------------------------ Fig. 2 training
def _curves(tag, col, n_bins=60):
    rows = []
    for p in glob.glob(os.path.join(ROOT, "runs", tag, "ppo_asym_s*", "progress.csv")):
        df = pd.read_csv(p)
        bins = np.linspace(0, df["timesteps"].max(), n_bins + 1)
        df["bin"] = pd.cut(df["timesteps"], bins, labels=bins[1:])
        rows.append(df.groupby("bin", observed=True)[col].mean())
    m = pd.concat(rows, axis=1)
    return m.index.astype(float).values, m.mean(axis=1).values, m.std(axis=1).values


def _lambda(tag, n_bins=60):
    rows = []
    for p in glob.glob(os.path.join(ROOT, "runs", tag, "ppo_asym_s*", "lambda.csv")):
        df = pd.read_csv(p)
        rows.append(df.set_index("timesteps")["lambda"])
    m = pd.concat(rows, axis=1).sort_index()
    return m.index.values, m.mean(axis=1).values, m.std(axis=1).values


def fig_training():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W * 2, 1.9), gridspec_kw={"wspace": 0.35})
    for key, tag in (("PPO", "main"), ("Lag-PPO", "lag-T90")):
        x, m, s = _curves(tag, "mean_cost")
        c, mk, label = STYLE[key]
        a1.plot(x / 1e3, m, color=c, label=label)
        a1.fill_between(x / 1e3, m - s, m + s, color=c, alpha=0.18, lw=0)
    a1.axhline(0.05, color=MUTED, ls="--", lw=1)
    a1.text(160, 0.15, "target d = 0.05", color=MUTED, fontsize=6.5)
    a1.annotate("", (200, 0.055), (200, 0.14), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8))
    a1.set_ylim(0, 0.6)
    a1.set_xlabel("Training steps (k)")
    a1.set_ylabel("Violation rate\n(share of steps, wait > 90 s)")
    a1.legend(frameon=False)
    x, m, s = _lambda("lag-T90")
    c = STYLE["Lag-PPO"][0]
    a2.plot(x / 1e3, m, color=c, label=STYLE["Lag-PPO"][2])
    a2.fill_between(x / 1e3, m - s, m + s, color=c, alpha=0.18, lw=0)
    a2.set_xlabel("Training steps (k)")
    a2.set_ylabel("Lagrange multiplier λ")
    a2.set_ylim(bottom=0)
    save(fig, "fig2_training")


# ------------------------------------------------------------------ Fig. 3 trade-off
def fig_tradeoff():
    """Mean delay vs max delay on asym, all methods (mean ± std over seeds)."""
    df = pd.read_csv(os.path.join(RES, "all_runs.csv"))
    df = df[df["scenario"] == "asym"]
    groups = {
        "Max-Pressure": ("Max-Pressure", STYLE["Max-Pressure"]),
        "Webster": ("Webster", STYLE["Webster"]),
        "PPO [main]": ("PPO", STYLE["PPO"]),
        "PPO [lag-T120]": ("Lagrangian T=120", STYLE["Lag-PPO"]),
        "PPO [lag-T90]": ("Lagrangian T=90 (ours)", STYLE["Lag-PPO"]),
        "PPO [struct-R90]": ("Structural R=90", ("#6250d6", "X", "")),
        "PPO [fairshaped-a5]": ("Reward shaping α=5", ("#52514e", "*", "")),
        "PPO [fixedlam-0.025]": ("Fixed λ=0.025", ("#8a8984", "p", "")),
    }
    offsets = {"Max-Pressure": (5, -3), "Webster": (-12, 8), "PPO": (4, 5), "Lagrangian T=120": (-66, -12),
               "Lagrangian T=90 (ours)": (6, -3), "Structural R=90": (4, -11), "Reward shaping α=5": (-6, 14),
               "Fixed λ=0.025": (6, 2)}
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    for method, (label, (c, m, _)) in groups.items():
        d = df[(df["method"] == method) & (df["train_scenario"].isin(["asym", "-"]))]
        if d.empty:
            continue
        x, y = d["mean_delay"], d["max_delay"]
        ax.errorbar(x.mean(), y.mean(), xerr=x.std(), yerr=y.std(), color=c, marker=m, ms=5,
                    elinewidth=0.8, capsize=0, markeredgecolor="white", markeredgewidth=0.8)
        ax.annotate(label, (x.mean(), y.mean()), xytext=offsets.get(label, (4, 3)), textcoords="offset points",
                    fontsize=6.5, color=INK)
    ax.set_xlim(left=12.0)
    ax.set_xlabel("Mean delay (s)  → worse")
    ax.set_ylabel("Max delay (s)  → worse")
    save(fig, "fig3_tradeoff")


# ------------------------------------------------------------------ Fig. 4 robustness
def fig_robustness():
    path = os.path.join(RES, "robustness.csv")
    if not os.path.exists(path):
        print("skip fig4 (no results/robustness.csv)")
        return
    df = pd.read_csv(path)
    name = {"Max-Pressure": "Max-Pressure", "Webster": "Webster", "PPO [main]": "PPO",
            "PPO [lag-T90]": "Lag-PPO", "PPO [dr-lag]": "DR-Lag-PPO", "PPO [dr-ppo]": "DR-PPO"}
    df["key"] = df["method"].map(name)
    df = df[df["key"].notna()]
    ps = {"clean": 0.0, "drop0.1": 0.1, "drop0.2": 0.2, "drop0.3": 0.3}
    df = df[df["condition"].isin(ps)]
    df["p"] = df["condition"].map(ps)
    fig, axes = plt.subplots(1, 2, figsize=(COL_W * 2, 2.0), sharey=True)
    for ax, scen, title in zip(axes, ("asym", "surge"), ("Training demand (asym)", "Demand shift (surge)")):
        d = df[df["scenario"] == scen]
        per_run = d.groupby(["key", "run", "p"])["mean_cost"].mean().reset_index()
        for key in ("Max-Pressure", "Webster", "PPO", "Lag-PPO", "DR-PPO", "DR-Lag-PPO"):
            g = per_run[per_run["key"] == key].groupby("p")["mean_cost"]
            if g.ngroups == 0:
                continue
            c, m, label = STYLE[key]
            ax.errorbar(g.mean().index, g.mean().values, yerr=g.std().fillna(0).values, color=c, marker=m,
                        label=label, elinewidth=0.8, capsize=0, markeredgecolor="white", markeredgewidth=0.8)
        ax.axhline(0.05, color=MUTED, ls="--", lw=1)
        ax.set_title(title, color=INK)
        ax.set_xlabel("Detector dropout probability p")
        ax.set_xticks([0, 0.1, 0.2, 0.3])
    axes[0].set_ylabel("Violation rate")
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "fig4_robustness")


# ------------------------------------------------------------------ Fig. 5 proposition
def fig_proposition():
    path = os.path.join(RES, "proposition_sweep.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    for key in ("Max-Pressure", "Webster"):
        g = df[df["method"] == key].groupby("side_vph")["side_max_delay"]
        c, m, label = STYLE[key]
        ax.errorbar(g.mean().index, g.mean().values, yerr=g.std().values, color=c, marker=m, label=label,
                    elinewidth=0.8, capsize=0, markeredgecolor="white", markeredgewidth=0.8)
    mp = df[df["method"] == "Max-Pressure"].groupby("side_vph")["side_max_delay"].mean()
    x = np.array([25, 400])
    ref = mp.iloc[0] * (x / mp.index[0]) ** -0.5
    ax.plot(x, ref, color=MUTED, ls="--", lw=1)
    ax.text(140, ref[0] * (140 / 25) ** -0.5 * 0.62, "slope −1/2\n(Prop. 1)", color=MUTED, fontsize=6.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([25, 50, 100, 200, 400])
    ax.set_xticklabels(["25", "50", "100", "200", "400"])
    ax.minorticks_off()
    ax.set_xlabel("Side-road demand (veh/h per approach)")
    ax.set_ylabel("Max side-road delay (s)")
    ax.legend(frameon=False, loc="lower left")
    save(fig, "fig5_proposition")


if __name__ == "__main__":
    which = sys.argv[1:] or ["teaser", "training", "tradeoff", "robustness", "proposition"]
    for w in which:
        globals()[f"fig_{w}"]()
