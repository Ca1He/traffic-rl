# Literature notes

Every citation was checked against Crossref, arXiv, publisher pages or library catalogues.
BibTeX entries: `report/references.bib`. Items marked ⚠ still need a manual check before submission.

## A. RL for traffic signal control, benchmarks, simulators

| Key | Paper | What they did / method | Relation to our project |
|---|---|---|---|
| WeiZYL18 | IntelliLight (KDD 2018) | Early DQN single-intersection control on real data; reward = weighted sum of queue, delay, waiting, switches, throughput, travel time | Canonical DQN-for-TSC. Optimizes average efficiency, **no fairness / max-wait constraint** |
| WeiCZWGXL19 ⚠ | PressLight (KDD 2019) | DQN with reward = −pressure, links RL to Max-Pressure theory; arterial networks | Justifies pressure reward and MP baseline. Pressure is throughput-oriented and can starve low-demand lanes. ⚠ re-check author list on PDF |
| WeiXZZZCZZXL19 | CoLight (CIKM 2019) | Graph attention over neighbours, parameter-shared DQN, city scale | Related work for the 2×2 grid; we use shared-parameter IPPO, focus on constraints and robustness |
| ZhengXZFWZLXL19 | FRAP (CIKM 2019) — title "Learning Phase Competition for TSC" | Symmetry-invariant phase-competition network; generalizes across intersection structures | Architecture route to generalization; we study generalization via evaluation and constraints |
| WeiZGL19survey | Survey on TSC Methods (arXiv 2019) | Taxonomy of classical + RL methods, state/action/reward choices, metrics | General background; reference for Fixed-Time / Webster |
| AultS21 | RESCO benchmark (NeurIPS 2021 D&B) | SUMO benchmark with realistic scenarios; finds prior RL methods **not robust to varying sensing assumptions** | Direct motivation for our sensor noise / dropout study |
| sumorl | sumo-rl (GitHub 2019) | Gym/PettingZoo TSC interface; default obs = phase + min-green flag + lane density + queue; reward = Δ waiting time | Our environment |
| LopezBBEFHLRWW18 | SUMO (ITSC 2018) | Reference paper for the simulator | Cite once when describing the simulator |
| ZhangWSLDW22 | Advanced-MP (ICML 2022) | State representation matters more than algorithm complexity; Advanced-MP counts running + queued vehicles | Optional stronger rule-based baseline; shows MP-style controllers are hard to beat |
| MeiLSW23 | TSC under Missing Data (IJCAI 2023) | Imputes missing states (and rewards) when some intersections lack sensors | **Closest work to our dropout axis.** We test test-time noise/dropout without imputation, combined with a fairness constraint |
| ShiDLC23 ⚠ | Generalizability & robustness of large-scale TSC (arXiv 2023) | Deep-RL TSC is brittle under missing data and unseen networks; ensemble + distributional RL on GCN | Supports our joint demand-shift + dropout evaluation. ⚠ journal version unverified, cite as arXiv |

Seen in search but **not verified, do not cite yet**: arXiv 2506.13836 (robustness under incidents, 2025); arXiv 2603.15283 (robustness evaluation, 2026); Rodrigues & Azevedo 2019 (dropout training for sensor failure).

## B. Constrained RL, fairness, robustness + novelty check

### B1. Core constrained / robust RL

| Key | Paper | Key idea | How we use it |
|---|---|---|---|
| Altman99 | Constrained MDPs (1999 book) | CMDP theory, Lagrangian duality; optimal policies may need randomization | Formal definition of our constrained problem |
| TesslerMM19 | RCPO (ICLR 2019) | Multi-timescale Lagrangian actor-critic; λ updated slower than the policy; convergence proof | Theoretical grounding of our Lagrangian PPO |
| AchiamHTA17 | CPO (ICML 2017) | Trust-region with per-update constraint satisfaction | Canonical alternative; we choose PPO-Lagrangian for simplicity |
| RayAA19 | Safety Gym (OpenAI 2019 tech report) | PPO-Lagrangian is a strong, simple baseline | Source of PPO-Lagrangian |
| StookeAA20 | PID Lagrangian (ICML 2020) | Gradient-ascent λ = integral control → oscillation; add P and D terms | Explains λ oscillation; optional ablation |
| TobinFRSZA17 | Domain Randomization (IROS 2017) | Randomize simulator parameters to transfer | Our demand + sensor randomization |
| JainCH84 | Jain's fairness index (DEC TR-301, 1984) | J = (Σx)²/(nΣx²) ∈ [1/n, 1] | Secondary fairness metric. Caveat: high Jain ≠ good (all can be equally bad) → also report max/p95 |
| ZhangCXLLBH20 | SA-MDP (NeurIPS 2020) | Robustness to adversarial observation perturbations | Framing; our perturbations are **random, not adversarial** — say so |

### B2. Closest prior work: fairness in RL-based TSC

| Key | What they did | Overlap with us |
|---|---|---|
| RaeisL21 (ITSC 2021) | SUMO single intersection, DDQN; reward penalizes squared waits (scalarized); metrics p95/max wait + Jain; baselines SOTL, Max-Pressure | **High on motivation & metrics, low on method.** Most important contrast: reward shaping vs our explicit constraint. No noise / shift tests |
| HaydariAZC25 ⚠ (ACM JATS 2025) "FAirLight" | Constrained multi-objective RL: fairness + emissions as constraints ("peak and average constraint models"), max-entropy off-policy | **High — closest "constrained RL for fair TSC".** ⚠ full text not read: check whether the peak constraint is max waiting time. No robustness evaluation |
| YeDWZWC23 (IEEE TCAD 2023) | Fairness index built into reward/design; hierarchical action space | Medium; not a CMDP constraint, no robustness |
| YanZBB20 (IROS 2020) | Efficiency + equity reward weighting | Medium; reward weighting |
| SiddiqueWZ20 (ICML 2020) | Generalized Gini welfare over per-lane waits; SUMO intersection shows average-optimal DQN distributes waits unequally | Medium; **empirical precedent for our Proposition** |
| SatheeshP25 (ACM JATS 2025) "MAPPO-LCE" | MAPPO + Lagrange multipliers on structural constraints (max green, phase skips); CityFlow | **High on method**, but constrains structural proxies, not outcome (max wait); lists max-wait/variance constraints as future work; no robustness |

### B3. Robustness of RL-based TSC

| Key | What they did | Overlap |
|---|---|---|
| RodriguesA19 (ITSC 2019) | DQN in AIMSUN under demand surges, incidents, sensor failures; agents must see these in training | High on robustness axis; no fairness |
| TanSS20 (J. Big Data Anal. Transp. 2020) | Train with perturbed state inputs for noise tolerance | High on noise; no fairness / constraint |
| MeiLSW23 (IJCAI 2023) | Impute missing sensor data | Medium; they impute, we stress-test |
| ShiDLC23 (arXiv 2023) | GNN + distributional RL ensemble robust to missing data / new networks | Medium |
| arXiv 2603.15283 ⚠ (2026) | Training on diverse O-D patterns keeps RL robust to demand shift | Evidence for demand DR; not yet in .bib |

### Novelty verdict (from the search)

**Partially covered — a defensible niche remains.**

Already done, so we must **not** claim these as new:
- RL signal control can be unfair; max/p95 wait and Jain are known metrics (RaeisL21, YeDWZWC23, YanZBB20, SiddiqueWZ20).
- Constrained / Lagrangian RL for TSC in general (HaydariAZC25, SatheeshP25).
- Robustness of RL-TSC to sensor failure / noise / demand shift (RodriguesA19, TanSS20, MeiLSW23, ShiDLC23).

Not found in any paper:
1. An **outcome constraint on max lane waiting time** solved with PPO-Lagrangian, compared head-to-head with **reward-shaping fairness** (RaeisL21-style) and **structural constraints** (max-green, SatheeshP25-style) under one protocol.
2. **Does the fairness guarantee survive distribution shift?** No paper evaluates fairness together with noise / dropout / demand-shift robustness; recent papers list this as an open gap.
3. A formal (even simple) statement that average-wait-optimal policies can starve low-demand approaches — prior work argues it qualitatively or shows it empirically. Present it as an **illustrative proposition** grounded in priority-queueing results (Kleinrock76), not a new theorem.

**Repositioned pitch:** "Do fairness constraints survive distribution shift?" rather than "first fair/constrained RL for TSC".

## C. Classical control, core RL, MARL, evaluation

| Key | Paper | What / key idea | How we use it |
|---|---|---|---|
| Varaiya13 | Max pressure control (TR-C 2013) | Pick the phase with largest upstream − downstream queue; proven to stabilize any demand within capacity | Max-Pressure baseline. Guarantee is about **stability, not per-approach fairness** → motivates our constraint |
| Webster58 | Traffic Signal Settings (RRL Tech. Paper 39, 1958) | Optimal cycle C0 = (1.5L+5)/(1−Y); green splits ∝ flow ratios | Webster fixed-time baseline (not Tech. Paper 56!) |
| MnihKSRVBGRFOPB15 | DQN (Nature 2015) | Q-learning + replay + target network | DQN agent (SB3) |
| SchulmanWDRK17 | PPO (arXiv 2017) | Clipped surrogate objective | PPO, IPPO, base of Lagrangian PPO |
| SchulmanMLJA16 | GAE (ICLR 2016) | λ-weighted TD residuals | Advantages for reward critic and cost critic |
| YuVVGWBW22 | MAPPO (NeurIPS 2022 D&B) | Parameter-shared PPO is a strong MARL baseline | Justifies shared PPO on 2×2 grid. Title has **no comma** |
| deWittGMMTSW20 | IPPO (arXiv 2020) | Independent PPO matches centralized methods on SMAC | Precedent for IPPO. First author "Schroeder de Witt"; arXiv only |
| RaffinHGKED21 | Stable-Baselines3 (JMLR 2021) | Tested reference implementations | DQN / PPO |
| TerryBGJHSSDHPW21 | PettingZoo (NeurIPS 2021) | Multi-agent Gym API | sumo-rl multi-agent interface |
| SuttonB18 | Sutton & Barto, 2nd ed. (2018) | MDP, TD, policy gradient | Background |
| Kleinrock76 ⚠ | Queueing Systems Vol. II (1976), Ch. 3 | Cobham: low-priority wait ∝ 1/[(1−σ_{k−1})(1−σ_k)] → ∞ as high-priority load → 1 | **Basis for Proposition 1 (starvation).** ⚠ check page of Cobham's formula |
| TassiulasE92 | Max-weight / back-pressure (IEEE TAC 1992) | Throughput-optimal scheduling | With Varaiya13: stability-optimal ≠ delay-fair |
| HendersonIBPPM18 | Deep RL that Matters (AAAI 2018) | Seeds/hyperparameters change results a lot | Justifies multi-seed protocol |
| AgarwalSCCB21 | Statistical Precipice (NeurIPS 2021) | IQM + stratified bootstrap CIs | Report IQM with 95% bootstrap CI |
