# Project status / handoff

**Project.** Paper: *Safety-Constrained DRL for Cost-Optimal Dynamic Compressor Dispatch under
TOU Prices.* Repo `zhuochenglee/DRL`, branch `refactor/multi-algo-pipeline-env`, PR #1 (update PR
body via `git credential` + GitHub API, gh not installed).

**Drafts (complete, real numbers).** `paper/draft.md` (EN) + `paper/draft_zh.md` (ZH); build docx:
`cd paper && pandoc draft.md -o draft.docx --toc --toc-depth=2`. Both have 12 figures, Tables
1/1b/2/3, 43 refs, Eqs (1)–(10). Chinese docx CJK may need an East-Asian font in Word (fallback
usually fine).

**Key results (real, 5-seed gun-barrel / 3-seed benchmarks).** DP/MPC 5.38, **Distilled-MPC 5.13**
(cheapest, ~0.4 viol, 2.5 ms), GA 5.51±2.3, PPO 7.81, SAC 15.48, Constrained-SAC 13.67, TD3 21.4,
Rule 22.9/19viol. Three networks: gun-barrel, branched, GasLib-40 (real geometry). Ablation: −feasibility
realization → 6 viol; −soft margin → cost 5.89 (premium cause). `alpha_on=1.29`.

**Data:** `raw_data/` — per-record CSVs for 3 networks + ablations + sensitivity + fig2/fig4/fig5
Prism-ready tables + `raw_data.zip`.

## Figure scripts (Python/matplotlib, run from repo root)
- `paper/make_figures.py` — schematic Figs 1,2,3 (matplotlib originals; Fig 3 arrow fix applied).
- `paper/make_analysis_figures.py` — ablation / cost-latency Pareto / cross-network (Figs 7,11,12).
- `paper/export_fig2_data.py`, `export_fig2_prism.py` — Fig 2 numbers → CSV.
- **Nature-style versions (via nature-figure skill, Python backend):**
  - `paper/fig2_nature.py` ✅ done — compressor map + disconnected feasible set
  - `paper/fig3_nature.py` ✅ done — method overview
  - `paper/fig4_nature.py` ✅ done — gun-barrel nominal cost bars
  - `paper/fig5_nature.py` ✅ done — 24-h dispatch (retrains C-SAC + distill seed 0; also writes
    `raw_data/fig5_dispatch.csv`)

## Nature-style palette (keep consistent)
neutral `#4d4d4d/#8c8c8c/#c2c2c2`; Distilled-MPC `#2c6fbb` (blue); Constrained-SAC `#e8923b` (orange);
infeasible/bad `#c44e58` (red); feasible `#3aaa64` (green). 7 pt, editable text
(`svg.fonttype=none`, `pdf.fonttype=42`), no top/right spines, export svg+pdf+tiff(600)+png.

## TODO (next session)
1. (optional) Nature-style versions of remaining figs: Fig 1 (networks), 6 (robustness), 7 (ablation),
   8/9 (branched), 10 (gaslib), 11 (Pareto), 12 (cross-network) — reuse the palette above.
2. Swap the finished `figN_nature.png` into `draft.md`/`draft_zh.md` (replace the matplotlib
   embeds), update captions (drafts above have suggested captions in chat), regenerate both docx,
   commit.
3. Remaining paper gaps: verify domain refs [15,16,19,20]; the TOU price stays synthetic (GasLib is
   gas-only) — the remaining credibility lift is a coupled gas–electricity dataset.
