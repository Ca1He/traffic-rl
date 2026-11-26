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
