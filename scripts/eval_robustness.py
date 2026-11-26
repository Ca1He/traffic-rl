"""Step 7: robustness of trained policies and baselines to distribution shift and sensor faults.

For every method we evaluate on the asym (train) and surge (demand shift) scenarios under
detector dropout p in {0, 0.1, 0.2, 0.3} (transient, redrawn every step) and a persistent-failure
condition (p = 0.2 for the whole episode), plus Gaussian noise sigma = 0.1.

    python scripts/eval_robustness.py --tags main lag-T90 dr-ppo dr-lag --workers 10

Writes results/robustness.csv (one row per evaluation episode).
"""
import argparse
import copy
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("LIBSUMO_AS_TRACI", "1")

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

EVAL_SEEDS = [1000, 1001, 1002]
SCENARIOS = ["asym", "surge"]
CONDITIONS = {  # name -> sensing overrides
    "clean": {},
    "drop0.1": {"dropout_p": 0.1},
    "drop0.2": {"dropout_p": 0.2},
    "drop0.3": {"dropout_p": 0.3},
    "fail0.2": {"dropout_p": 0.2, "dropout_mode": "episode"},
    "noise0.1": {"noise_std": 0.1},
}


def _eval_one(task):
    """Evaluate one method (a trained model or a baseline) on all scenarios x conditions."""
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO, DQN
    from agents.baselines import MaxPressure, Webster, scenario_design_demand
    from envs import load_config, make_env, run_episode

    label, kind, path = task
    if kind == "model":
        with open(os.path.join(path, "config.yaml"), encoding="utf-8") as fh:
            saved = yaml.safe_load(fh)
        base_cfg = saved["cfg"]
        Algo = PPO if saved["args"]["algo"] == "ppo" else DQN
        model = Algo.load(os.path.join(path, "model.zip"), device="cpu")
        policy = lambda obs, env: int(model.predict(obs, deterministic=True)[0])  # noqa: E731
    else:
        base_cfg = load_config()
    rows = []
    for scen in SCENARIOS:
        if kind == "baseline":
            policy = MaxPressure() if label == "Max-Pressure" else Webster(scenario_design_demand("asym"))
            # Webster is designed for the asym demand it was tuned on: surge is a shift for it too.
        for cond, sensing in CONDITIONS.items():
            cfg = copy.deepcopy(base_cfg)
            cfg["env"]["scenario"] = scen
            cfg["demand_randomization"] = False
            cfg["sensing"] = {**load_config()["sensing"], **sensing}
            cfg["cost"].pop("structural_max_red", None)
            for seed in EVAL_SEEDS:
                m = run_episode(make_env(cfg, seed=seed), policy, seed=seed)
                m.update(method=label, run=os.path.basename(path) if path else label,
                         scenario=scen, condition=cond, seed=seed)
                rows.append(m)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["main", "lag-T90"])
    ap.add_argument("--train-scenario", default="asym")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="results/robustness.csv")
    args = ap.parse_args()

    tasks = [("Max-Pressure", "baseline", None), ("Webster", "baseline", None)]
    for tag in args.tags:
        pattern = os.path.join(ROOT, "runs", tag, f"ppo_*_s*")
        for path in sorted(glob.glob(pattern)):
            train_scen = os.path.basename(path).split("_")[1]
            if train_scen in (args.train_scenario, "random") and os.path.exists(os.path.join(path, "model.zip")):
                tasks.append((f"PPO [{tag}]", "model", path))
    print(f"{len(tasks)} methods x {len(SCENARIOS)} scenarios x {len(CONDITIONS)} conditions "
          f"x {len(EVAL_SEEDS)} seeds", flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(_eval_one, tasks):
            rows += r
            print(f"  done: {r[0]['method']} / {r[0]['run']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, args.out), index=False)
    print(df.groupby(["scenario", "condition", "method"])[["mean_delay", "max_delay", "mean_cost"]]
          .mean().round(2).to_string())


if __name__ == "__main__":
    main()
