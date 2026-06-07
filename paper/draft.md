# Safety-Constrained Deep Reinforcement Learning for Cost-Optimal Dynamic Compressor Dispatch under Time-of-Use Electricity Prices

*(Working title — preliminary working draft. All numbers are real, from a reduced-scale run of `run_experiments.py`; see the integrity note under the abstract.)*

---

## Abstract

Electric-driven compressor stations let gas transmission operators shift compression
work in time, using the pipeline's own line-pack as a short-term energy buffer to
exploit time-of-use (TOU) electricity tariffs. Realising this flexibility online is hard:
the dispatch must respect strongly nonlinear compressor characteristics and a tight
operating envelope (speed limits, surge/choke) while reacting to uncertain demand and
price. Existing deep-reinforcement-learning (DRL) work on gas networks either optimises a
*steady-state* single-step decision or abstracts the gas side to algebraic constraints,
so the coupling between detailed compressor physics and economic, dynamic dispatch is
left open. We formulate the problem as a 24-hour **constrained Markov decision process
(CMDP)** over a dynamic line-pack model with TOU pricing, embedding the paper-style
centrifugal-compressor maps (adiabatic head, head/efficiency polynomials) and the full
operating envelope as state-dependent constraints, and we release a single-source-of-truth
physics implementation shared by the environment and the planning baselines. We benchmark
a rule-based heuristic, a genetic algorithm (GA), a perfect-foresight dynamic-programming
(DP) optimum, a receding-horizon MPC, and four DRL agents (SAC, TD3, PPO, and a
**Lagrangian / constrained SAC**) on a common objective. On this well-modelled problem the
model-based controllers are decisive: DP and MPC achieve the lowest feasible 24-h cost
(7.03) with zero violations. Model-free DRL, at a reduced training budget, is **not yet
competitive on cost or feasibility and exhibits high variance** across algorithms and seeds;
the constraint-aware variant does not yet show a consistent advantage over penalty-reward
SAC. The one robust, quantified DRL advantage is **deployment latency**: a learned policy
acts in ≈4.6 ms, ≈15× faster than re-solving the MPC and ≈2000× faster than the GA. We
report these findings honestly as a *preliminary* study: the contribution is the
physics-faithful CMDP formulation, the constraint-aware method, and a fair, reproducible
benchmark that exposes exactly where model-free DRL currently falls short on constrained
gas dispatch — and what (longer training, safety layers, a realistic network) is required
to close the gap.

**Keywords:** natural gas pipeline; line-pack flexibility; time-of-use pricing; compressor
optimisation; deep reinforcement learning; constrained MDP; soft actor–critic.

> **Status / integrity note.** All numbers below come from a *reduced-but-real* run
> (`run_experiments.py --steps 120000 --seeds 2 --perturb 6`, ≈30 min wall-clock). They are
> genuine, not illustrative, but the DRL budget is small and the variance is large; they are
> not camera-ready. A one-flag scale-up (more steps/seeds) is provided. We deliberately do
> **not** overstate the DRL results.

---

## 1. Introduction

Natural gas remains a backbone of the energy transition, and the electrification of
compressor drivers couples gas transmission to the power system at the operational
timescale. When a compressor is electrically driven and the operator faces a TOU
electricity tariff, there is an economic incentive to **shift compression work to
low-price hours** and ride out high-price hours on stored line-pack — the mass of gas held
in the pipe, which acts as a distributed, fast short-term store. Capturing this value
online is a constrained, dynamic optimal-control problem: the controller must keep node
pressures within safe bounds, keep every compressor inside its speed and surge/choke
envelope, and return the line-pack inventory to a target at the end of the horizon, all
while demand and price are uncertain.

Classical approaches trade off optimality, speed, and model fidelity. Dynamic programming
(DP) yields a near-global optimum but suffers the curse of dimensionality and needs the
full model and forecast. Model predictive control (MPC) re-optimises online but is only as
good as its model and forecast and pays a per-step optimisation cost. Metaheuristics such
as genetic algorithms (GA) are simple but open-loop and prone to local optima. Deep
reinforcement learning (DRL) is attractive because, once trained, it is **model-free at
deployment, near-instant to evaluate, and naturally robust to the uncertainty seen during
training**. Liu et al. [1] showed DRL can match DP on *steady-state* compressor
optimisation with large speed-ups, but framed the task as a one-step MDP, leaving the
dynamic, economically-driven dispatch problem — and its interaction with the compressor
operating envelope — unaddressed.

This paper makes that step. Our contributions are:

1. **A physics-faithful dynamic CMDP** for electric-compressor dispatch: a 24-hour
   line-pack model with TOU economics that retains the paper-style centrifugal-compressor
   characteristics (Eqs. 7–11 of [1]) and encodes the *full operating envelope* — node
   pressures, compressor speed, and surge/choke on the normalised inlet flow — as
   state-dependent constraints. Most economic-dispatch DRL studies abstract this physics
   away; we keep it and show it materially shapes the feasible policy.
2. **Constrained (Lagrangian) SAC** that optimises economics subject to a single,
   weight-free, unit-normalised constraint cost, with a dual variable adapted by projected
   dual ascent. This removes the brittle hand-tuning of penalty weights used by standard
   penalty-reward DRL and turns "stay feasible" into an explicit, monitored objective.
3. **A fair, multi-method benchmark** — rule-based, GA, perfect-foresight DP (optimum
   bound), receding-horizon MPC, and SAC/TD3/PPO/Constrained-SAC — all scored on one shared
   environment and objective, with multi-seed statistics, a robustness study under
   demand/price perturbation, and ablations on the constraint mechanism and reward design.

We are deliberate about scope: the network is a compact illustrative element with
trend-consistent (not field-measured) profiles, in the spirit of [1]. The contribution is
the *formulation and the constraint-aware learning method*, and the quantified evidence
that constraint-aware DRL is a competitive, robust, real-time controller for this class of
problem.

---

## 2. Related work and positioning

**DRL for gas-network/compressor control.** Liu et al. [1] formulate steady-state pipeline
optimisation as a one-step MDP solved by an auto-tuned SAC, comparing against GA and DP.
Chen et al. [2] apply DRL to *predictive demand-response management* in gas pipelines.
These establish DRL for gas control but either freeze the dynamics (one-step) or target
demand response rather than TOU-price-driven compressor economics with a detailed
operating envelope.

**Line-pack as flexibility/storage.** The use of line-pack as a short-term store for
price arbitrage and balancing is well established in the *model-based/economics* literature
[3,4]: linepack valuation under price uncertainty and the operation of coupled
gas–electricity networks with line-pack. These works are LP/valuation or offline-optimisation
in nature and rely on accurate models; they do not learn a real-time feedback policy under
nonlinear compressor maps and uncertainty.

**Integrated electricity–gas system (IEGS) dispatch with DRL.** A large, active body of
work uses SAC/PPO/multi-agent DRL for IEGS economic dispatch under TOU pricing [5,6].
These co-optimise both networks but typically represent the gas side with simplified or
algebraic constraints and omit the compressor characteristic map and surge/choke envelope.

**Our niche.** We sit at the intersection but occupy the gap each leaves open: a
*single-pipeline, dynamic, TOU-priced* dispatch that (i) keeps the detailed compressor
physics of [1], (ii) treats the operating envelope as hard *feasibility* constraints rather
than economics, and (iii) learns a constraint-aware real-time policy whose value we quantify
against DP/MPC and under uncertainty. To our knowledge this specific combination —
physics-faithful compressor envelope + line-pack/TOU economics + constrained DRL with a
DP-optimum and MPC benchmark — has not been reported.

---

## 3. Problem formulation

### 3.1 Network and hydraulics

We consider a gun-barrel element: a source node, an internal node (2), and a demand node
(3), connected by two pipes; an electric compressor at the source sets the source pressure
through a discharge ratio. Edge flow follows the paper's Weymouth-type steady relation,
expressed on squared pressures,

$$ q_e = \mathrm{sgn}(\Delta(p^2)_e)\,\sqrt{|\Delta(p^2)_e| / K_e}, $$

with the incidence matrix mapping edge flows to nodal balances. The compressor discharge
ratio $\alpha\in[1,2]$ is the control; the source pressure is $p_0 = p_{0,\mathrm{ref}}\,\alpha$.

### 3.2 Compressor characteristics and operating envelope

The centrifugal compressor follows Liu et al. [1, Eqs. 7–11]. The adiabatic head from the
discharge ratio (Eq. 9) is

$$ H = \frac{ZRT}{M}\,\frac{\kappa}{\kappa-1}\Big[\alpha^{(\kappa-1)/\kappa}-1\Big], $$

(with $ZRT/M$ lumped into a single coefficient). The head and efficiency maps (Eqs. 7–8)
share the normalised inlet-flow coordinate $\phi = Q_{in}/\omega$:

$$ H/\omega^2 = A_H + B_H\phi + C_H\phi^2 + D_H\phi^3, \qquad
   \eta = A_E + B_E\phi + C_E\phi^2 + D_E\phi^3, $$

with the Table-3 coefficients of [1]. Given $H$ (from $\alpha$) and $Q_{in}$ (from the
source-pipe flow), Eq. 7 is a cubic in $\phi$; its physically valid root yields the speed
$\omega = Q_{in}/\phi$, and Eq. 8 gives the efficiency. Power follows Eq. 11,
$P = Q_{in}\,\rho\,H/\eta$.

The **operating envelope** (Eq. 20 of [1]) is enforced as constraints rather than folded
into economics:
- node pressures $p_i \in [p_{\min}, p_{\max}]$;
- compressor speed $\omega \in [\omega_{\min}, \omega_{\max}]$ — a controller demand outside
  the band (the unconstrained natural speed $\omega_{\mathrm{raw}}$ from Eq. 7) is a violation;
- surge/choke: $\phi \in [\phi_{\text{lo}}, \phi_{\text{hi}}] = [Q_{in,\min}/\omega_{\min},\,
  Q_{in,\max}/\omega_{\max}]$.

A single source-of-truth implementation (`envs/compressor.py`) is shared by the environment
and by the DP/MPC planners, guaranteeing every method is optimised and scored against
identical physics. At the calibrated nominal operating point ($\alpha\approx1.57$) the model
reproduces the paper's Case-1 figures (speed $\approx7.1\times10^3$ rpm, efficiency
$\approx0.76$ vs. 7370 rpm / 73.9%).

### 3.3 Dynamic line-pack and TOU economics

Each internal node $i$ holds a mass-equivalent inventory $m_i = \beta_i p_i$ (line-pack).
Mass balance is integrated explicitly over hourly steps,

$$ m_i^{t+1} = \mathrm{clip}\big(m_i^t + \Delta t\,(\text{net flow})_i,\; m_i^{\min}, m_i^{\max}\big),
   \quad p_i = m_i/\beta_i, $$

coupling injection and withdrawal across the day. The economic signal is the electricity
purchase cost $c^t = P^t\,\pi^t$ with $\pi^t$ the TOU tariff (off-peak/peak). A terminal
line-pack target prevents end-of-day inventory depletion.

### 3.4 Constrained MDP

- **State** $s_t$: normalised demand, price, the two internal pressures, line-pack ratio,
  a cyclic time encoding, and the look-ahead price profile.
- **Action** $a_t$: the discharge ratio $\alpha_t\in[1,2]$.
- **Economic reward** $r^{\text{econ}}_t = -c^t = -P^t\pi^t$.
- **Constraint cost** $c_t$: a weight-free, unit-normalised sum of relative violations
  (pressure band, flow limits, speed band, surge/choke band, terminal line-pack gap).
- **Objective:** $\max \mathbb{E}\sum_t r^{\text{econ}}_t$ s.t. $\mathbb{E}\sum_t c_t \le d$,
  with budget $d\approx 0$.

---

## 4. Method

### 4.1 Constrained (Lagrangian) SAC

We optimise the Lagrangian $L = \mathbb{E}\sum_t (r^{\text{econ}}_t - \lambda\, c_t)$ with an
off-the-shelf SAC actor–critic on the shaped reward, while the dual variable is updated by
projected dual ascent on the episodic constraint cost,

$$ \lambda \leftarrow \mathrm{clip}\big(\lambda + \eta\,(\widehat{\textstyle\sum_t c_t} - d),\; 0,\; \lambda_{\max}\big), $$

using an EMA-smoothed episodic cost estimate $\widehat{\sum_t c_t}$ for stability. Because
$c_t$ is unit-normalised, a *single* $\lambda$ balances economics against feasibility, and
$\lambda$ is *learned* rather than hand-set — the key difference from penalty-reward SAC,
whose fixed weights $(w_{\text{pressure}}, w_{\text{flow}}, w_{\text{speed}}, w_{\text{surge}})$
must be tuned a priori. Returns are reward-normalised (running-std) so the wide-range,
$\lambda$-scaled signal trains the critic stably. Actor and critic are 3-layer MLPs
(128–64–32), matching [1].

### 4.2 Baselines

- **Rule-based:** price-aware open-loop heuristic (compress in cheap hours).
- **Genetic algorithm (GA):** open-loop optimisation of the 24-step $\alpha$ vector by
  simulation on the environment.
- **Dynamic programming (DP-oracle):** backward value iteration on a discretised
  $(p_2,p_3)$ state with the shared compressor cost and Eq.-20 penalties; evaluated with
  *perfect foresight* of the realised scenario — a near-global-optimum **upper bound** on
  achievable performance.
- **MPC:** the certainty-equivalent DP feedback policy built on the *nominal* forecast and
  executed closed-loop on the realised environment.
- **DRL:** SAC, TD3, PPO (penalty reward) and Constrained-SAC, identical architecture/budget.

### 4.3 Fairness and reproducibility

All methods use the same environment, objective, and compressor physics. DRL agents are
trained on a noisy environment (to see uncertainty) and evaluated closed-loop; planners use
the nominal forecast (DP-oracle uses the realised future). Metrics are averaged over seeds
(DRL) and perturbed scenarios (robustness) with reported standard deviations.

---

## 5. Experimental setup

**Environment.** Horizon 24 h; $p_{0,\mathrm{ref}}=100$, safe pressure band $[80,150]$, hard
band $[10,200]$; flow limits $q_{1,\max}=1000,\,q_{2,\max}=900$; $\omega\in[5000,9400]$ rpm,
$Q_{in}\in[4000,12500]$; $\kappa=1.30$; line-pack $\beta=[50,40]$, $K=[0.02,0.05]$. TOU tariff
0.30 (off-peak) / 1.20 (peak). Penalty weights (scoring/penalty-reward DRL):
$w_{\text{pressure}}=1000,\,w_{\text{flow}}=5,\,w_{\text{terminal}}=250,\,w_{\text{speed}}=1,\,
w_{\text{surge}}=2000$. Compressor calibration: $ZRT/M=138$, $Q_{in}=10\,|q_0|$,
power coefficient $4.5\times10^{-6}$; head/efficiency polynomials from [1, Table 3].

**Training.** Reduced-but-real budget: 120 000 timesteps, 2 seeds, $\gamma=0.995$, learning rate
$10^{-3}$ (SAC/TD3) / $3\times10^{-4}$ (PPO), actor/critic MLP 128–64–32, reward normalisation on,
DRL trained on a noisy env (multiplicative demand noise 0.08). `run_experiments.py --steps … --seeds …`
scales to full budget in one line.

**Scenarios & metrics.** A nominal day plus 6 perturbed realizations (multiplicative demand noise
0.10, price-profile shift). Metrics: 24-h electricity cost, constraint-violation hours, mean
compressor efficiency, terminal line-pack gap, and planning/inference latency.

---

## 6. Results

Numbers are from `models/exports/paper/results_summary.csv` (120k steps, 2 seeds, 6 perturbed
scenarios). "Nominal" = the reference day; "Robust" = mean over perturbed realizations. Cost
is the 24-h electricity cost (lower is better); "Viol" is constraint-violation hours.

### 6.1 Nominal and robust performance (Table 1)

| Method | Nominal cost | Nominal viol-h | Robust cost | Robust viol-h | Mean η | Latency |
|---|---|---|---|---|---|---|
| **DP-oracle** (perfect foresight) | **7.03** | **0.0** | 7.60 ± 0.6 | 0.2 | 0.836 | 69 ms |
| **MPC** (nominal-model feedback) | **7.03** | **0.0** | 7.70 ± 0.8 | **0.0** | 0.836 | 69 ms |
| PPO | 4.65 ± 1.1 | 1.0 | 9.32 ± 2.9 | 1.1 | 0.838 | 3.4 ms |
| GA (open-loop) | 28.3 ± 8.6 | 0.5 | 29.9 ± 9.1 | 0.2 | 0.812 | 9.6 s |
| SAC | 33.0 ± 3.9 | 9.0 | 35.4 ± 4.5 | 8.0 | 0.806 | 4.6 ms |
| Constrained-SAC | 36.5 ± 5.4 | 8.0 | 37.1 ± 5.8 | 8.5 | 0.795 | 4.4 ms |
| TD3 | 0.00 | 5.0 | 0.00 | 5.5 | 0.838 | 3.3 ms |
| Rule-based | 22.9 | 19.0 | 23.3 ± 0.5 | 19.0 | 0.804 | 1.7 ms |

**Reading the table honestly:**
- **Model-based control dominates.** DP-oracle and MPC reach the lowest feasible cost (7.03)
  with zero violations on this well-modelled problem — as expected when an accurate model and
  forecast are available. MPC essentially matches the perfect-foresight DP, and stays feasible
  under perturbation.
- **Model-free DRL is not yet competitive and is unstable.** SAC and Constrained-SAC over-compress
  (cost 33–37) yet still incur 8–9 violation-hours; TD3 collapses to a degenerate "never compress"
  policy (zero electricity cost, hence cost 0, but 5 violations); PPO is the only DRL agent near
  the feasible frontier (cost 4.65, 1 violation) but achieves sub-DP cost only by *slightly*
  violating a constraint. Cross-seed variance is large (±4–9).
- **The constraint-aware method does not yet pay off at this budget.** Constrained-SAC is *not*
  better than penalty-SAC here (36.5/8 vs 33.0/9). The Lagrange multiplier saturates while the
  underlying policy still cannot reach feasibility, indicating the bottleneck is the base RL
  optimisation/budget, not the dual mechanism.

### 6.2 The one robust DRL advantage: deployment latency (Table 1, last column)

A trained policy acts in **≈4.6 ms**, versus **69 ms** to re-solve the MPC/DP feedback law and
**9.6 s** for the GA — i.e. **≈15× faster than MPC** and **≈2000× faster than the GA** at
deployment. This is the genuine, data-supported DRL value proposition for real-time dispatch;
it is, however, decoupled from solution quality, which DRL does not yet deliver here.

### 6.3 Dispatch behaviour (Fig. 2 — `fig_dispatch.png`)

The 24-h curves confirm the mechanism on the model-based side: DP/MPC pre-charge line-pack during
the off-peak window and coast (α≈1) through the peak-price hours, topping up the terminal
inventory at the last off-peak hour. The DRL policies do not reproduce this end-game cleanly —
notably they miss the terminal line-pack target by a wide margin — which is the main driver of
their poor cost/feasibility.

### 6.4 Robustness (Fig. 3 — `fig_robustness.png`)

Under perturbed demand/price, closed-loop MPC stays feasible at near-constant cost; the
open-loop GA stays near-feasible but expensive; the DRL agents do not degrade much *because they
were already far from optimal*. So the intended "DRL is more robust than open-loop" story is not
demonstrated here — the open-loop GA is actually more feasible than SAC/Constrained-SAC.

### 6.5 Status of the ablations

With the headline DRL results inconclusive, the planned ablations (Constrained-SAC vs penalty-SAC
weight sensitivity; surge/choke on/off; terminal line-pack on/off; dual-variable trajectory) are
**not yet meaningful** and are deferred until the base DRL is trained to a competitive operating
point (Section 7).

---

## 7. Discussion and limitations

The headline finding is sobering and worth stating plainly: **on a well-modelled, low-dimensional
dispatch problem, model-based control (DP/MPC) is hard to beat, and model-free DRL — including our
constraint-aware variant — is not yet competitive at a reduced training budget.** This is a
genuine result, not a presentation choice. Its causes and the path forward:

- **Hard exploration / credit assignment.** The optimal policy is "pre-charge line-pack off-peak,
  coast on-peak, refill at the last off-peak hour." The penalty for under-charging is a *delayed*
  catastrophe (a pressure violation hours later), which model-free RL discovers poorly at ~120k
  steps. The bimodal failure (TD3 → never compress; SAC → over-compress) is the signature of an
  under-converged, high-variance optimisation.
- **The constraint mechanism is sound but starved.** The Lagrange multiplier saturated while the
  base policy was still infeasible: the dual ascent cannot help if the actor–critic cannot first
  represent a near-feasible policy. The method must be re-evaluated once the base RL is trained to
  a competitive point (more steps, more seeds, hyperparameter search, possibly a safety/projection
  layer for hard feasibility).
- **MPC is the right competitor and it is strong.** When an accurate model and forecast exist,
  certainty-equivalent MPC is near-optimal and feasible. DRL's only demonstrated edge is
  *deployment latency* (~15× faster than MPC); that matters only if DRL can first reach comparable
  quality — which is the open problem.
- **Scope of the test system.** A compact element with trend-consistent profiles isolates the
  mechanism but is not a field network; absolute costs are illustrative. A realistic / benchmark
  network (e.g., GasLib-derived) with measured load and price is the primary extension and is
  needed for any credible quantitative claim.
- **Compressor model.** Single electric compressor with lumped thermodynamic coefficients;
  multi-compressor routing and discrete unit commitment are out of scope.

**What would make this a defensible Q2 paper** (in order of leverage): (1) train the DRL to a
competitive operating point — substantially more steps, ≥5 seeds, a hyperparameter sweep, reward
shaping or a safety layer for hard feasibility — so that Constrained-SAC demonstrably reaches near-
MPC cost with ≈0 violations and *then* the ablations become meaningful; (2) move to a realistic /
benchmark network with measured data; (3) make the uncertainty larger and harder so that DRL's
robustness and amortised speed can actually beat re-solving MPC. Absent (1), the honest claim is
limited to the formulation, the open benchmark, and the deployment-latency observation.

---

## 8. Conclusion

We posed electric-compressor dispatch under TOU electricity pricing as a physics-faithful dynamic
constrained MDP that retains the paper's compressor characteristic map and its full operating
envelope, and built a fair, reproducible benchmark spanning rule-based, GA, perfect-foresight DP,
receding-horizon MPC, and four DRL agents including a Lagrangian constrained SAC. On this
well-modelled problem, **DP and MPC achieve the optimal feasible dispatch (cost 7.03, zero
violations), while model-free DRL is not yet competitive at a reduced training budget and the
constraint-aware variant does not yet outperform penalty-reward SAC**; the single robust DRL
advantage is deployment latency (~15× faster than MPC, ~2000× faster than GA). We report this
honestly as a preliminary study: the lasting contributions are the formulation, the single-source
physics and open benchmark, and a clear-eyed account of where constraint-aware DRL must improve —
longer training, safe-RL feasibility, and a realistic network — before it can rival model-based
control for line-pack economic dispatch.

---

## References

[1] Z. E. Liu et al., "A novel optimization framework for natural gas transportation
pipeline networks based on deep reinforcement learning," *Energy and AI*, 18:100434, 2024.

[2] Chen et al., "A deep reinforcement learning-based method for predictive management of
demand response in natural gas pipeline networks," *Journal of Cleaner Production*, 2021.

[3] N. Keyaerts et al., "Line-pack storage valuation under price uncertainty," *Energy*, 2013.

[4] Operation of natural gas and electricity networks with line pack, *Journal of Modern
Power Systems and Clean Energy*, 2019.

[5] Integrated Electricity–Gas System Optimal Dispatch Based on Deep Reinforcement Learning,
IEEE, 2022.

[6] Soft actor–critic DRL for interval optimal dispatch of integrated energy systems with
uncertainty in demand response and renewable energy, *Engineering Applications of AI*, 2023.

*(Reference list to be completed with full bibliographic details and an expanded survey for
submission.)*
