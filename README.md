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
| Result | Command | Output |
|---|---|---|
| Saturation flows used by Webster | `python scripts/measure_saturation.py` | through 1630, left 1120 veh/h/lane |
| Baseline table (3 scenarios x 5 seeds) | `python scripts/eval_baselines.py` | `results/baselines.csv` |

Evaluation seeds start at 1000 and never overlap with training seeds.
Note: decisions are taken every `delta_time` = 5 s, so fixed-time greens are effectively
rounded up to the decision grid (e.g. 30 s and 31.5 s greens behave identically).
