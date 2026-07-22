"""Ablation study on the gun-barrel network.

Trains Constrained-SAC with one component removed at a time and measures the effect
on the (true) 24-h cost, constraint-violation hours, and terminal line-pack gap.
The headline ablation is `no_feasibility_realization`: disabling the dead-band->off
realization should bring envelope violations back, confirming it is the enabler.

Usage:
  python ablations.py                       # 100k steps, 3 seeds
  python ablations.py --steps 150000 --seeds 5
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from envs.pipeline_env import EnvConfig, PaperInspiredDynamicLinepackEnv
from baselines import evaluate_model
from constrained_sac import train_constrained_sac

OUT = Path("models/exports")

# Each variant removes ONE component relative to the full method.
VARIANTS = {
	"full":                       {},
	"no_feasibility_realization": {"enable_feasibility_realization": False},
	"no_soft_margin":             {"w_soft_margin": 0.0},
	"no_surge_penalty":           {"w_surge": 0.0},
	"no_terminal_linepack":       {"w_terminal_linepack": 0.0},
}


def run(args):
	rows = []
	for name, ov in VARIANTS.items():
		for seed in range(args.seeds):
			t0 = time.time()
			cfg_train = EnvConfig(noise_scale=0.08, **ov)
			model, wrap = train_constrained_sac(
				cfg_train, total_timesteps=args.steps, seed=seed,
				lam_init=50.0, dual_lr=1.5, lam_max=2000.0, learning_rate=3e-4)
			eval_env = PaperInspiredDynamicLinepackEnv(config=EnvConfig(noise_scale=0.0, **ov))
			m = evaluate_model(eval_env, model)
			rows.append(dict(variant=name, seed=seed, cost=m["total_cost"],
			                 viol=m["violation_hours"], tlp=m["terminal_linepack_gap"],
			                 lam=wrap.lam, t=time.time() - t0))
			print(f"  {name:<26} seed{seed}: cost={m['total_cost']:6.2f} "
			      f"viol={m['violation_hours']:2d} tlp={m['terminal_linepack_gap']:6.0f} "
			      f"({time.time()-t0:.0f}s)", flush=True)

	OUT.mkdir(parents=True, exist_ok=True)
	with open(OUT / "ablations.csv", "w", newline="") as f:
		w = csv.DictWriter(f, fieldnames=["variant", "seed", "cost", "viol", "tlp", "lam", "t"])
		w.writeheader()
		for r in rows:
			w.writerow(r)

	# aggregate
	print(f"\n{'variant':<28}{'cost':>14}{'viol-h':>10}{'term-gap':>12}")
	print("-" * 64)
	for name in VARIANTS:
		rs = [r for r in rows if r["variant"] == name]
		c = np.array([r["cost"] for r in rs]); v = np.array([r["viol"] for r in rs])
		g = np.array([r["tlp"] for r in rs])
		print(f"{name:<28}{c.mean():>8.2f}±{c.std():<4.1f}{v.mean():>10.1f}{g.mean():>12.0f}")
	print(f"\nSaved {OUT/'ablations.csv'}")


def parse_args():
	p = argparse.ArgumentParser()
	p.add_argument("--steps", type=int, default=100000)
	p.add_argument("--seeds", type=int, default=3)
	return p.parse_args()


if __name__ == "__main__":
	a = parse_args()
	print(f"Ablation study: steps={a.steps} seeds={a.seeds} | variants={list(VARIANTS)}")
	t = time.time()
	run(a)
	print(f"Total: {(time.time()-t)/60:.1f} min")
