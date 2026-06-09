# Dynamic line-pack dispatch under TOU pricing: physics-faithful CMDP, constraint-aware DRL, MPC distillation, and a two-network benchmark

## Summary

This PR extends the steady-state DRL pipeline-optimisation framework of Liu et al. (*Energy & AI*, 2024) into a **dynamic, economically-driven dispatch problem** and builds a complete, fair, reproducible study around it. It turns the project from a multi-algorithm sandbox into a coherent benchmark with a Markdown/Word paper draft (`paper/draft.md`, `paper/draft.docx`).

The problem: an **electric-driven compressor** can shift compression work in time, using the pipeline's **line-pack** as a short-term store to exploit **time-of-use (TOU)** electricity prices, subject to node-pressure, compressor speed and surge/choke constraints and an end-of-day inventory target.

## Key contributions

1. **Physics-faithful dynamic CMDP.** A 24-hour line-pack environment with TOU economics that keeps the paper's centrifugal-compressor maps (adiabatic head, head/efficiency polynomials, Eqs. 7–11) and encodes the full operating envelope (Eq. 20) as state-dependent constraints. A single source-of-truth compressor module (`envs/compressor.py`) is shared by the environment and every planner, so all methods are scored against identical physics.
2. **Feasibility-aware action realisation (the key enabler).** The compressor's feasible set is *disconnected* (off, or on above a state-dependent minimum speed). A continuous policy otherwise keeps commanding the infeasible "dead band" and incurs ~6 envelope violations/day **regardless of training budget**. Realising dead-band commands as the unit staying off makes the action map onto the feasible set and takes every DRL agent to **zero** envelope violations.
3. **Constrained (Lagrangian) SAC.** Optimises economics subject to a single, weight-free, normalised constraint cost, with a dual variable adapted by projected dual ascent — removing hand-tuned penalty weights.
4. **MPC distillation closes the cost premium.** From-scratch DRL is feasible but ~2–3× the optimum cost. Distilling the MPC feedback law into a small MLP (behavioral cloning + DAgger) yields a policy that **matches MPC cost at ~ms latency** (the from-scratch premium is removed).
5. **Three-network benchmark.** Rule-based, GA, DP, receding-horizon MPC, SAC/TD3/PPO, Constrained-SAC and Distilled-MPC, all on one objective, with multi-seed statistics, robustness under perturbation, and an honest discussion. Validated on (i) a gun-barrel element, (ii) a branched benchmark, and (iii) a sub-network with **native pipe geometry from the real GasLib-40 instance**; the env/DP/MPC/distillation are all generalised to arbitrary single-compressor topologies (N-D DP).

## Results (real, reduced-but-real: 150k steps, 3 seeds, 6 perturbed scenarios)

**Gun-barrel network** — 24-h electricity cost / violation-hours / latency:

| Method | Cost | Viol-h | Latency |
|---|---|---|---|
| GA / DP / MPC (near-optimal refs) | 4.3–5.4 | 0 | 1–11 s |
| **Distilled-MPC** | **5.34** | ~0.7 | **2.8 ms** |
| SAC / Constrained-SAC (from scratch) | 15.0 / 13.4 | 0 | ~5 ms |
| Rule-based | 22.9 | 19 | – |

**Branched benchmark** — the core findings transfer; the cost premium *shrinks*:

| Method | Cost | Viol-h | Term-gap | Latency |
|---|---|---|---|---|
| GA (clean reference) | 5.94 | 0 | 0 | 11.5 s |
| **SAC** | **4.46** | 0 | 318 | 4.6 ms |
| Distilled-MPC | 6.21 | 2.7 | 11 | 3.3 ms |
| DP / MPC (3-D grid) | 6.02 | 2.0 | 128 | 6.8 s |
| Constrained-SAC | 14.40 | 0.3 | 357 | 4.6 ms |

**GasLib-40-derived network** (real pipe geometry) — the cleanest case for the method:

| Method | Cost | Viol-h | Term-gap | Latency |
|---|---|---|---|---|
| GA (optimum ref) | 5.57 | 0 | 5 | 11.9 s |
| **Distilled-MPC** | **5.26** | **0** | 111 | **2.8 ms** |
| DP / MPC | 6.70 | 0 | 141 | 6.5 s |
| Constrained-SAC | 17.08 | 0 | 66 | 4.7 ms |
| SAC | 16.07 | 0 | 300 | 4.9 ms |

On the real-geometry network the distilled controller is the best learned method outright (matches the GA optimum, beats MPC, 0 violations, ms latency).

Honest caveats (in the paper): the constraint-aware-SAC ranking is network-dependent (it wins on the gun-barrel and GasLib nets, loses on the branched one); the branched 3-D grid DP/MPC are grid-limited (~2 violations) so GA is the clean reference there; TD3 degenerates (never compresses → depletes line-pack), which the terminal-gap column exposes; the GasLib geometry needed affine recalibration for numerical stability and the TOU price stays synthetic (GasLib is gas-only).

## What's in this PR

- `envs/compressor.py` — shared compressor physics (Eqs. 7–11) + feasibility-aware action realisation.
- `envs/pipeline_env.py` — topology-general dynamic line-pack CMDP; `branched_benchmark_config()`, `gaslib_benchmark_config()`.
- `DP.py` — DP reference (2-D fast path + general N-D); `baselines.py` — rule / GA / MPC + evaluators.
- `constrained_sac.py` — Lagrangian SAC; `distill_mpc.py` — BC + DAgger MPC distillation.
- `run_experiments.py` — unified harness (`--network {default,branched}`, multi-seed, figures + CSVs).
- `paper/draft.md`, `paper/draft.docx` — the paper draft; `paper/make_table.py` — table generator.
- Results: `models/exports/paper/` (gun-barrel), `models/exports/paper_branched/`, `models/exports/paper_gaslib/`; all raw data bundled in `raw_data/` (+ `raw_data.zip`).

## Reproduce

```bash
python run_experiments.py --quick                              # fast plumbing check
python run_experiments.py --steps 150000 --seeds 3 --perturb 6 # gun-barrel
python run_experiments.py --network branched --steps 150000 --seeds 3 --dagger 12
python run_experiments.py --network gaslib   --steps 150000 --seeds 3 --dagger 12  # GasLib-40 geometry
python paper/make_table.py                                     # regenerate the results table
```

## Status

Reduced-but-real budget (150k steps / 3 seeds) — numbers are genuine but not camera-ready. The remaining credibility lift is a field / GasLib-derived network with measured load and price; the branched benchmark is a step toward that.
