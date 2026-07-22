# Raw experimental data

All raw data from the dispatch experiments, for both networks. Numbers are genuine
(reduced-but-real budget: 150k training steps, 3 seeds, 6 perturbed scenarios).

## Layout

```
raw_data/
├── gunbarrel/                     # 3-node gun-barrel element (paper §6.1–6.7, Table 1)
│   ├── results_raw.csv            # per method × seed × scenario record
│   └── results_summary.csv        # aggregated mean/std per method
├── branched/                      # 4-node branched benchmark (paper §6.8, Table 2)
│   ├── results_raw.csv
│   ├── results_summary.csv
│   └── fig_*.png
├── gaslib/                        # GasLib-40-derived network (paper §6.9, Table 3)
│   ├── results_raw.csv            #   native GasLib-40 pipe geometry, real demand chain
│   ├── results_summary.csv
│   └── fig_*.png
├── ablations.csv                  # Constrained-SAC component ablations (paper §6.7, Table 1b)
├── sensitivity_w_penalty.csv      # terminal-line-pack penalty-weight sweep (SAC)
├── fig2_eta_curve.csv             # Figure 2(a) data: phi, eta(phi), H/omega^2 (compressor map)
├── fig2_alpha_realization.csv     # Figure 2(b) data: commanded->realized alpha + full operating point
├── fig2_constants.csv             # Figure 2 scalars: alpha_on, phi/omega/Qin bounds, polynomials
├── fig2a_prism.csv                # Prism-ready: eta curve split into surge/feasible/choke columns
└── fig2b_prism.csv                # Prism-ready: realized alpha split into OFF/ON + identity
```

## `results_raw.csv` schema (one row per evaluation)

| column   | meaning |
|----------|---------|
| `method` | Rule, GA, DP-oracle, MPC, SAC, TD3, PPO, Constrained-SAC, Distilled-MPC |
| `seed`   | training seed (DRL/GA); planners are deterministic (seed 0) |
| `kind`   | `nominal` (reference day) or `perturb` (a perturbed realization) |
| `scenario` | `-1` for nominal; `0..5` for the 6 perturbed scenarios |
| `cost`   | 24-h electricity cost (lower is better) |
| `viol`   | constraint-violation hours (pressure / flow / compressor speed / surge-choke) |
| `eff`    | mean compressor efficiency over the day |
| `tlp_gap`| terminal line-pack gap vs target (gun-barrel target 9000; branched 13000). NOT in `cost`/`viol`, so read it alongside `cost` — a policy can look cheap by depleting storage (e.g. degenerate TD3) |
| `time`   | wall-clock latency for the 24-h dispatch (DRL = inference; MPC/DP = build+run; GA = full optimisation) |

## `results_summary.csv` schema (one row per method)

`method, nominal_cost_mean, nominal_cost_std, nominal_viol, robust_cost_mean,
robust_cost_std, robust_viol_mean, mean_eff` — "nominal" = the reference day;
"robust" = aggregated over the 6 perturbed scenarios (and seeds for DRL).

## Reproduce / regenerate

```bash
python run_experiments.py --steps 150000 --seeds 3 --perturb 6 --lr 3e-4 --dagger 12               # gun-barrel
python run_experiments.py --network branched --steps 150000 --seeds 3 --perturb 6 --dagger 12       # branched
python paper/make_table.py                                                                          # render the table
```

Gun-barrel writes to `models/exports/paper/`, branched to `models/exports/paper_branched/`.
(Only the per-episode records above are persisted; per-hour trajectories are computed in
memory for the dispatch figure and not saved to CSV.)
