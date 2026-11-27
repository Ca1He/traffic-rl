"""Step 8: parameter-shared IPPO (optionally Lagrangian) on the 2x2 grid, plus grid baselines.

    python train_grid.py --seed 0                               # IPPO
    python train_grid.py --seed 0 --set lagrangian.enabled=true --tag grid-lag
    python train_grid.py --baselines                            # Fixed-Time / Webster / Max-Pressure

Steps are counted over all intersections (4 per simulation step).
Outputs runs/<tag>/ippo_<scenario>_s<seed>/{model.zip, progress.csv, lambda.csv, eval.csv, summary.json}
and results/grid_baselines.csv.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

os.environ.setdefault("LIBSUMO_AS_TRACI", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from agents.grid_baselines import grid_baselines  # noqa: E402
from agents.lagrangian import LagrangeCallback, LagrangeState  # noqa: E402
from envs import load_config  # noqa: E402
from envs.grid import GRID_SCENARIOS, GridVecEnv, grid_route_file, run_grid_episode  # noqa: E402
from train import HYPERPARAMS, EpisodeLogger, parse_set  # noqa: E402

EVAL_SEEDS = [1000, 1001, 1002]


def design_demand(scenario):
    periods = GRID_SCENARIOS[scenario]
    total = sum(p.end - p.begin for p in periods)
    return {a: sum(p.vph[a] * (p.end - p.begin) for p in periods) / total for a in "NSEW"}


def evaluate(policy_factory, cfg, scenarios):
    rows = []
    for scen in scenarios:
        ecfg = copy.deepcopy(cfg)
        ecfg["env"]["scenario"] = scen
        for seed in EVAL_SEEDS:
            m = run_grid_episode(ecfg, policy_factory(scen), seed=seed)
            m.update(scenario=scen, seed=seed)
            rows.append(m)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="asym")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=600_000)
    ap.add_argument("--set", nargs="*")
    ap.add_argument("--tag", default="grid-ippo")
    ap.add_argument("--eval-scenarios", nargs="*", default=["asym", "surge"])
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--eval-only", action="store_true", help="re-evaluate an existing model.zip")
    args = ap.parse_args()
    torch.set_num_threads(1)
    for s in GRID_SCENARIOS:  # pre-generate route files (parallel jobs only read them)
        grid_route_file(s)

    if args.baselines:
        cfg = load_config()
        frames = []
        for name in ("Fixed-Time", "Webster", "Max-Pressure"):
            df = evaluate(lambda scen: grid_baselines(design_demand("asym"))[name](), cfg, args.eval_scenarios)
            df["method"] = name
            frames.append(df)
        out = pd.concat(frames)
        os.makedirs("results", exist_ok=True)
        out.to_csv("results/grid_baselines.csv", index=False)
        print(out.groupby(["scenario", "method"])[["mean_delay", "max_delay", "jain_approach", "mean_cost"]]
              .mean().round(3).to_string())
        return

    cfg = load_config(overrides={"env": {"scenario": args.scenario}, **parse_set(args.set)})
    run_dir = os.path.join("runs", args.tag, f"ippo_{args.scenario}_s{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    hp = dict(HYPERPARAMS["ppo"], batch_size=720)  # rollout = 720 steps x 4 intersections
    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump({"args": vars(args), "cfg": cfg, "hyperparams": hp}, fh)

    if args.eval_only:
        model = PPO.load(os.path.join(run_dir, "model.zip"), device="cpu")
        policy = lambda obs, v: model.predict(obs, deterministic=True)[0]  # noqa: E731
        df = evaluate(lambda scen: policy, cfg, args.eval_scenarios)
        df.to_csv(os.path.join(run_dir, "eval.csv"), index=False)
        print(df.groupby("scenario")[["mean_delay", "max_delay", "jain_approach", "mean_cost"]].mean().round(3))
        return

    lag = cfg["lagrangian"]
    state = None
    if lag.get("enabled") or lag.get("fixed") is not None:
        state = LagrangeState(lam=lag["lam_init"], target=lag["target"], lr=lag["lr"],
                              lam_max=lag["lam_max"], fixed=lag.get("fixed"))
    venv = GridVecEnv(cfg, seed=args.seed, lagrange_state=state)
    callbacks = [EpisodeLogger(os.path.join(run_dir, "progress.csv"))]
    if state is not None:
        callbacks.append(LagrangeCallback(state))
    model = PPO("MlpPolicy", venv, seed=args.seed, device="cpu", verbose=0, **hp)
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=callbacks)
    train_s = time.time() - t0
    model.save(os.path.join(run_dir, "model"))
    if state is not None:
        pd.DataFrame(state.history).to_csv(os.path.join(run_dir, "lambda.csv"), index=False)
    venv.close()

    policy = lambda obs, v: model.predict(obs, deterministic=True)[0]  # noqa: E731
    df = evaluate(lambda scen: policy, cfg, args.eval_scenarios)
    df.to_csv(os.path.join(run_dir, "eval.csv"), index=False)
    summary = df.groupby("scenario")[["mean_delay", "max_delay", "jain_approach", "mean_cost"]].mean().round(3)
    with open(os.path.join(run_dir, "summary.json"), "w") as fh:
        json.dump({"train_seconds": train_s, "eval": summary.to_dict(orient="index")}, fh, indent=1)
    print(f"[{run_dir}] trained {args.steps} steps in {train_s/60:.1f} min")
    print(summary.to_string())


if __name__ == "__main__":
    main()
