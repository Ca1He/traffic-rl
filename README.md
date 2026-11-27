# Fair and Robust Traffic Signal Control via Constrained RL

ELEN E6885 (Reinforcement Learning) final project.

## Setup
```bash
pip install -r requirements.txt
```
SUMO is installed via the `eclipse-sumo` pip package (no separate installer needed).
Set `LIBSUMO_AS_TRACI=1` to use the much faster libsumo backend.

## Smoke test
```bash
python scripts/smoke_test.py
```
Runs one 3600 s episode of a random policy on a single intersection.

## Project layout
```
envs/      scenarios (route files), noisy sensing, wrappers (max-green, fairness cost,
           demand randomization), rewards, metrics, make_env / run_episode
agents/    baselines: Fixed-Time, Webster, Max-Pressure
configs/   base.yaml (all defaults)
scripts/   smoke_test, check_env, measure_saturation, eval_baselines
tests/     unit tests (python -m pytest -q tests)
docs/      literature notes
report/    references.bib
```

## Reproduce
All numbers in the report come from the commands below (a laptop CPU, ~6 h in total).
Training batches skip runs whose `eval.csv` already exists, so they can be resumed.

| Result | Command | Output |
|---|---|---|
| Tests | `python -m pytest -q tests` | 16 tests |
| Saturation flows (Webster) | `python scripts/measure_saturation.py` | 1630 / 1120 veh/h/lane |
| Baselines (Tab. 1) | `python scripts/eval_baselines.py` | `results/baselines.csv` |
| DQN / PPO, reward ablation | `python scripts/run_batch.py step5` | `runs/main`, `runs/reward-*` |
| Constraint experiments | `python scripts/run_batch.py step6` | `runs/lag-*`, `fixedlam-*`, `struct-*`, `fairshaped-*` |
| Domain randomization (feasible) | `python scripts/run_batch.py step7feas` | `runs/drfeas-*` |
| DR without feasibility filter (negative result) | `python scripts/run_batch.py step7long` | `runs/dr800-*` |
| Robustness (Fig. 4) | `python scripts/eval_robustness.py --tags main lag-T90 drfeas-ppo drfeas-lag drfeas-lag-demandonly` | `results/robustness.csv` |
| 2x2 grid (Tab. 3) | `python train_grid.py --baselines`; `python scripts/run_batch.py step8long` (IPPO); `python scripts/run_batch.py step8local` (Lagrangian) | `results/grid_baselines.csv`, `runs/grid1500-ippo`, `runs/gridlocal-lag` |
| Proposition 1 sweep (Fig. 5) | `python scripts/proposition_sweep.py` | `results/proposition_sweep.csv` |
| All tables + t-tests | `python scripts/aggregate.py && python scripts/stats.py` | `results/tables/*.tex`, `results/welch_tests.csv` |
| All figures | `python scripts/make_figures.py` | `figures/*.pdf` |
| Overleaf upload bundle | `python scripts/bundle_report.py` | `report_bundle.zip` |

Evaluation seeds start at 1000 and never overlap with training seeds (0-4).
Decisions are taken every `delta_time` = 5 s, so fixed-time greens are effectively rounded up
to the decision grid (e.g. 30 s and 31.5 s greens behave identically).
Note on the grid: `runs/grid1500-lag` used the trip-level waiting time as cost (did not converge,
kept as a negative result); `runs/gridlocal-lag` uses the per-intersection local wait.
