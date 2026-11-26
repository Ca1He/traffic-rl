# Proposition 1: average-delay-optimal control starves low-demand approaches

Illustrative result for the report (Section 3) — full proof goes to the supplementary material.
It is an EOQ-style (economic order quantity) argument in a deterministic fluid model; it is
**not** claimed as a new theorem: it formalizes the folklore that delay-minimizing priority
service starves low-priority classes (cf. Cobham's formula in Kleinrock 1976, Ch. 3).

## Model

* Two approaches sharing one intersection. Approach 1 (main road) has arrival rate λ₁,
  approach 2 (side road) has arrival rate λ₂ ≪ λ₁ (vehicles/s, deterministic fluid arrivals).
* Saturation flow μ while green; main-road utilization ρ = λ₁/μ < 1.
* Switching the green from one approach to the other costs a lost time L (yellow + start-up),
  during which nobody is served. Serving the side road and coming back costs 2L.
* Policy class Π_T: serve approach 1 by default; every T seconds make an *excursion* that
  serves approach 2 until its queue is empty and switches back.
  (For small λ₂ the excursion length is τ ≈ 2L; the side-road service time λ₂T/μ is negligible.)

## Delay accounting (per cycle of length T)

* **Main road.** During an excursion of length τ the main-road queue grows to λ₁τ and then
  drains at rate μ − λ₁. The area under this queue triangle is the extra delay
  K = λ₁τ² / (2(1 − ρ)) = 2λ₁L² / (1 − ρ).
* **Side road.** Vehicles arrive uniformly over the cycle and all wait until the next
  excursion: total delay λ₂T² / 2. A vehicle that arrives right after an excursion waits ≈ T,
  so the **maximum side-road wait is W₂ = T**.

Total delay per unit time:  J(T) = K / T + λ₂T / 2.

## Proposition 1

(a) **Starvation.** The average-delay-optimal cycle is
  T* = argmin_T J(T) = √(2K/λ₂) = 2L · √( λ₁ / ((1 − ρ) λ₂) ),
so the maximum side-road wait W₂* = T* grows without bound as λ₂/λ₁ → 0 or ρ → 1,
scaling as λ₂^(−1/2). Meanwhile the average delay per vehicle,
  J(T*) / (λ₁ + λ₂) = √(2Kλ₂) / (λ₁ + λ₂),
goes to 0 as λ₂ → 0: the average looks excellent while a minority waits arbitrarily long.

(b) **Bounded price of fairness.** Imposing the constraint W₂ ≤ T_max (i.e. T ≤ T_max) when
T* > T_max yields the constrained optimum T = T_max, and the increase in average delay per
vehicle is at most
  [J(T_max) − J(T*)] / (λ₁ + λ₂)  ≤  K / (T_max (λ₁ + λ₂))  ≤  2L² / ((1 − ρ) T_max),
which is **independent of λ₂**: no matter how unbalanced the demand, enforcing a max-wait bound
costs at most a constant amount of average delay.

## Proof

(a) J is convex on T > 0 with J'(T) = −K/T² + λ₂/2; setting J'(T) = 0 gives T* = √(2K/λ₂).
Substituting K = 2λ₁L²/(1 − ρ) gives T* = 2L √(λ₁/((1 − ρ)λ₂)), which → ∞ as λ₂/λ₁ → 0 or
ρ → 1. At the optimum both terms of J are equal, J(T*) = 2·K/T* = √(2Kλ₂); dividing by the
total flow λ₁ + λ₂ and noting √λ₂ → 0 gives the second claim.

(b) Since J is convex with minimizer T* > T_max, the constrained minimizer over T ≤ T_max is the
boundary T_max, and J(T_max) − J(T*) = K/T_max + λ₂T_max/2 − √(2Kλ₂).
Because T_max < T*, λ₂T_max/2 < λ₂T*/2 = √(2Kλ₂)/2 ≤ √(2Kλ₂), so the last two terms sum to
a negative number and J(T_max) − J(T*) ≤ K/T_max. Dividing by λ₁ + λ₂ ≥ λ₁ and substituting K:
  K / (T_max(λ₁ + λ₂)) ≤ 2λ₁L² / ((1 − ρ) T_max λ₁) = 2L² / ((1 − ρ) T_max).  ∎

## Numbers for our "asym" scenario (sanity check)

L ≈ 5 s (3 s yellow + 2 s start-up), μ ≈ 1630 veh/h (measured), λ₁ = 0.8 · 800 veh/h on the
critical through lane ⇒ ρ ≈ 0.39; λ₂ = 0.8 · 200 veh/h.
  T* ≈ 2·5·√(640 / (0.61 · 160)) ≈ 26 s,   price-of-fairness bound for T_max = 90 s: ≤ 0.9 s/veh.
The fluid model ignores min-green, stochastic arrivals and the left-turn phases, so absolute
numbers are far below the simulated max waits (hundreds of seconds); the proposition is about
**scaling**, which we verify empirically by sweeping λ₂/λ₁ (Step 6.1 experiment).

## Caveats to state in the report

* Deterministic fluid model, two approaches, one policy class (periodic excursions). Not a
  claim about all policies or stochastic arrivals.
* "Average-delay-optimal" here = optimal within Π_T; RL agents trained on average-delay rewards
  approximate this objective, which is why they inherit the starvation tendency.
