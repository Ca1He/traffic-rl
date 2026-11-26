"""Run a named batch of training jobs in parallel (one SUMO process per job).

    python scripts/run_batch.py step5 --workers 10
    python scripts/run_batch.py step5 --dry-run        # just list the commands

Each job's stdout/stderr goes to runs/<tag>/logs/<name>.log. Jobs whose eval.csv already
exists are skipped, so an interrupted batch can simply be re-run.
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS5 = range(5)
SEEDS3 = range(3)


def job(algo, scenario, seed, tag, steps=300_000, sets=None, eval_scenarios=None, config=None):
    cmd = [sys.executable, "train.py", "--algo", algo, "--scenario", scenario, "--seed", str(seed),
           "--steps", str(steps), "--tag", tag]
    if config:
        cmd += ["--config", config]
    if sets:
        cmd += ["--set", *sets]
    if eval_scenarios:
        cmd += ["--eval-scenarios", *eval_scenarios]
    name = f"{algo}_{scenario}_s{seed}"
    return name, tag, cmd


def batch_step5():
    jobs = []
    # Main single-intersection results: DQN / PPO trained on balanced and asym.
    # asym-trained models are also evaluated on surge (distribution shift, used in step 7).
    for algo in ("ppo", "dqn"):
        for scen in ("balanced", "asym"):
            for s in SEEDS5:
                jobs.append(job(algo, scen, s, "main",
                                eval_scenarios=["surge"] if scen == "asym" else None))
    # Reward ablation (PPO, asym): diff-waiting-time is the main run above.
    for reward in ("queue", "pressure", "fair-shaped"):
        for s in SEEDS3:
            jobs.append(job("ppo", "asym", s, f"reward-{reward}", sets=[f"env.reward={reward}"],
                            eval_scenarios=["surge"]))
    return jobs


def batch_step6():
    """Fairness constraint experiments (all PPO, trained on asym, evaluated on asym + surge)."""
    jobs, ev = [], ["surge"]
    lag = ["lagrangian.enabled=true"]
    # Main method: Lagrangian PPO, T = 90 s, d = 0.05.
    for s in SEEDS5:
        jobs.append(job("ppo", "asym", s, "lag-T90", sets=lag, eval_scenarios=ev))
    for s in SEEDS3:
        jobs.append(job("ppo", "balanced", s, "lag-T90", sets=lag))
    # Threshold sweep -> Pareto curve (mean delay vs max delay).
    for T in (60, 120):
        for s in SEEDS3:
            jobs.append(job("ppo", "asym", s, f"lag-T{T}", sets=lag + [f"cost.threshold={T}"], eval_scenarios=ev))
    # Fixed-lambda ablation (lambda*/10, lambda*, 10 lambda*; lambda* ~ 0.25 from the pilot).
    for lam in (0.025, 0.25, 2.5):
        for s in SEEDS3:
            jobs.append(job("ppo", "asym", s, f"fixedlam-{lam}", sets=[f"lagrangian.fixed={lam}"], eval_scenarios=ev))
    # Structural constraint (MAPPO-LCE style): no phase red for more than R seconds.
    for R in (60, 90):
        for s in SEEDS3:
            jobs.append(job("ppo", "asym", s, f"struct-R{R}", sets=lag + [f"cost.structural_max_red={R}"],
                            eval_scenarios=ev))
    # Reward-shaping fairness (Raeis-style) with stronger penalties (alpha = 1 is in step 5).
    for a in (5, 20):
        for s in SEEDS3:
            jobs.append(job("ppo", "asym", s, f"fairshaped-a{a}",
                            sets=["env.reward=fair-shaped", f"env.fair_alpha={a}"], eval_scenarios=ev))
    return jobs


def batch_step7():
    """Domain randomization: train on random demand (pool of 50) + random detector dropout U[0, 0.3]."""
    jobs = []
    dr = ["sensing.dropout_p_max=0.3"]
    for s in SEEDS5:
        jobs.append(job("ppo", "random", s, "dr-ppo", sets=dr))
        jobs.append(job("ppo", "random", s, "dr-lag", sets=dr + ["lagrangian.enabled=true"]))
    # Ablation: demand randomization only (no sensor randomization).
    for s in SEEDS3:
        jobs.append(job("ppo", "random", s, "dr-lag-demandonly", sets=["lagrangian.enabled=true"]))
    return jobs


def batch_step8():
    """2x2 grid: parameter-shared IPPO, unconstrained vs Lagrangian (trained on asym)."""
    jobs = []
    for s in SEEDS3:
        for tag, sets in (("grid-ippo", []), ("grid-lag", ["lagrangian.enabled=true"])):
            cmd = [sys.executable, "train_grid.py", "--seed", str(s), "--tag", tag]
            if sets:
                cmd += ["--set", *sets]
            jobs.append((f"ippo_asym_s{s}", tag, cmd))
    return jobs


BATCHES = {"step5": batch_step5, "step6": batch_step6, "step7": batch_step7, "step8": batch_step8}


def run(name, tag, cmd):
    run_dir = os.path.join(ROOT, "runs", tag, name)
    if os.path.exists(os.path.join(run_dir, "eval.csv")):
        return name, tag, "skipped", 0.0
    log_dir = os.path.join(ROOT, "runs", tag, "logs")
    os.makedirs(log_dir, exist_ok=True)
    t0 = time.time()
    with open(os.path.join(log_dir, f"{name}.log"), "w", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                             env={**os.environ, "LIBSUMO_AS_TRACI": "1", "OMP_NUM_THREADS": "1"})
    return name, tag, "ok" if rc == 0 else f"FAILED rc={rc}", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", choices=list(BATCHES))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    jobs = BATCHES[args.batch]()
    print(f"{len(jobs)} jobs, {args.workers} workers", flush=True)
    if args.dry_run:
        for _, _, cmd in jobs:
            print(" ".join(cmd[1:]))
        return
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        for name, tag, status, dt in ex.map(lambda j: run(*j), jobs):
            print(f"[{(time.time()-t0)/60:6.1f} min] {tag}/{name}: {status} ({dt/60:.1f} min)", flush=True)
    print(f"Batch done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
