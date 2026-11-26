"""Train an RL signal controller on the single intersection and evaluate it.

    python train.py --algo ppo --scenario asym --seed 0
    python train.py --algo dqn --scenario balanced --seed 3 --steps 200000
    python train.py --algo ppo --scenario asym --seed 0 --set env.reward=queue --tag reward-queue

Outputs (under runs/<tag>/<algo>_<scenario>_s<seed>/):
    model.zip          trained policy
    progress.csv       per-episode training curve (return, mean_cost, delay metrics)
    config.yaml        full resolved config
    eval.csv           evaluation episodes on the train scenario (+ --eval-scenarios)
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from stable_baselines3 import DQN, PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

from agents.lagrangian import LagrangeCallback, LagrangeState, PenalizedRewardWrapper  # noqa: E402
from envs import load_config, make_env, run_episode  # noqa: E402
from envs.metrics import episode_metrics, read_tripinfo  # noqa: E402

EVAL_SEED0 = 1000

HYPERPARAMS = {
    "ppo": dict(learning_rate=3e-4, n_steps=720, batch_size=180, n_epochs=10, gamma=0.99,
                gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=[64, 64])),
    "dqn": dict(learning_rate=1e-3, buffer_size=50_000, learning_starts=1_000, batch_size=64,
                gamma=0.99, train_freq=1, gradient_steps=1, target_update_interval=500,
                exploration_fraction=0.2, exploration_final_eps=0.02,
                policy_kwargs=dict(net_arch=[64, 64])),
}


class EpisodeLogger(BaseCallback):
    """Logs one row per finished training episode (return, mean cost, and tripinfo metrics)."""

    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.rows, self.ret, self.costs = [], 0.0, []

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        self.ret += float(info.get("raw_reward", self.locals["rewards"][0]))  # un-penalized reward
        self.costs.append(info.get("outcome_cost", info.get("cost", 0.0)))   # max-wait violation
        if self.locals["dones"][0]:
            row = {"timesteps": self.num_timesteps, "return": self.ret, "mean_cost": float(np.mean(self.costs)),
                   "lambda": info.get("lambda", np.nan)}
            self.rows.append(row)
            self.ret, self.costs = 0.0, []
            if len(self.rows) % 10 == 0:
                pd.DataFrame(self.rows).to_csv(self.path, index=False)
        return True

    def _on_training_end(self) -> None:
        pd.DataFrame(self.rows).to_csv(self.path, index=False)


def evaluate(model, cfg: dict, scenarios, n_episodes: int, out_csv: str):
    rows = []
    policy = lambda obs, env: int(model.predict(obs, deterministic=True)[0])  # noqa: E731
    for scen in scenarios:
        ecfg = copy.deepcopy(cfg)
        ecfg["env"]["scenario"] = scen
        ecfg["demand_randomization"] = False
        for k in range(n_episodes):
            seed = EVAL_SEED0 + k
            m = run_episode(make_env(ecfg, seed=seed), policy, seed=seed)
            m.update(scenario=scen, seed=seed)
            rows.append(m)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def parse_set(pairs):
    """--set a.b=value  ->  nested override dict (values parsed as YAML)."""
    out = {}
    for p in pairs or []:
        key, val = p.split("=", 1)
        d = out
        *parents, leaf = key.split(".")
        for k in parents:
            d = d.setdefault(k, {})
        d[leaf] = yaml.safe_load(val)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    ap.add_argument("--scenario", default="asym")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--config", default=None, help="experiment yaml (overrides configs/base.yaml)")
    ap.add_argument("--set", nargs="*", help="extra overrides, e.g. env.reward=queue")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--eval-scenarios", nargs="*", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.set_num_threads(1)
    cfg = load_config(args.config, overrides={"env": {"scenario": args.scenario}, **parse_set(args.set)})
    run_dir = os.path.join("runs", args.tag, f"{args.algo}_{args.scenario}_s{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump({"args": vars(args), "cfg": cfg, "hyperparams": HYPERPARAMS[args.algo]}, fh)

    env = make_env(cfg, seed=args.seed)
    callbacks = [EpisodeLogger(os.path.join(run_dir, "progress.csv"))]
    lag = cfg.get("lagrangian", {})
    state = None
    if lag.get("enabled") or lag.get("fixed") is not None:
        if args.algo != "ppo":
            raise ValueError("Lagrangian training is implemented for PPO only")
        state = LagrangeState(lam=lag["lam_init"], target=lag["target"], lr=lag["lr"],
                              lam_max=lag["lam_max"], fixed=lag.get("fixed"))
        env = PenalizedRewardWrapper(env, state)
        callbacks.append(LagrangeCallback(state))
    Algo = PPO if args.algo == "ppo" else DQN
    model = Algo("MlpPolicy", env, seed=args.seed, device=args.device, verbose=0, **HYPERPARAMS[args.algo])

    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=callbacks)
    if state is not None:
        pd.DataFrame(state.history).to_csv(os.path.join(run_dir, "lambda.csv"), index=False)
    train_s = time.time() - t0
    model.save(os.path.join(run_dir, "model"))
    env.close()

    scen = [args.scenario] + [s for s in (args.eval_scenarios or []) if s != args.scenario]
    if args.scenario == "random":
        scen = [s for s in scen if s != "random"] or ["balanced", "asym", "surge"]
    df = evaluate(model, cfg, scen, args.eval_episodes, os.path.join(run_dir, "eval.csv"))
    summary = df.groupby("scenario")[["mean_delay", "max_delay", "jain_approach", "mean_cost"]].mean().round(3)
    with open(os.path.join(run_dir, "summary.json"), "w") as fh:
        json.dump({"train_seconds": train_s, "eval": summary.to_dict(orient="index")}, fh, indent=1)
    print(f"[{run_dir}] trained {args.steps} steps in {train_s/60:.1f} min")
    print(summary.to_string())


if __name__ == "__main__":
    main()
