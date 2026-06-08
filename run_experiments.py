"""Unified experiment harness for the paper.

Trains/plans every method on a shared scenario, evaluates them on a nominal
forecast and on a set of perturbed (demand/price) realizations, aggregates
metrics over seeds/scenarios, and writes CSV tables + figures.

Fairness:
  * one environment / objective for every method (envs/compressor.py physics);
  * planners (rule, GA, MPC) use the NOMINAL forecast; DP-oracle uses the
    realized future (perfect-foresight upper bound on performance);
  * DRL agents are trained once (on a noisy env) per seed, then evaluated
    closed-loop on the same scenarios.

Usage:
  python run_experiments.py --quick            # fast debug (tiny budget)
  python run_experiments.py                     # reduced-but-real defaults
  python run_experiments.py --steps 120000 --seeds 5 --perturb 16   # full scale
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.pipeline_env import EnvConfig, PaperInspiredDynamicLinepackEnv
from baselines import (evaluate_model, evaluate_schedule, ga_schedule,
                       rule_based_schedule, run_mpc)
from DP import solve_dp
from constrained_sac import train_constrained_sac

OUT_DIR = Path("models/exports/paper")
NET_ARCH = [128, 64, 32]   # shared actor/critic architecture (paper-aligned)
GAMMA = 0.995
LR = 1e-3
# DP/MPC discretization. Finer grids reduce cost but eventually hug the constraint
# boundary and become infeasible when executed on the continuous env; 121x101 is the
# finest grid that stays feasible here, and serves as the DP/receding-horizon reference.
DP_NP, DP_NA = 121, 101


# ── scenarios ─────────────────────────────────────────────────────────────────
def make_scenarios(env, n, noise_scale, seed):
	"""Perturbed (demand, price) realizations, matching the env's noise model."""
	rng = np.random.default_rng(seed)
	base_d = np.asarray(env.demand_series, dtype=np.float64)
	base_p = np.asarray(env.tou_price_series, dtype=np.float64)
	out = []
	for _ in range(n):
		d = np.clip(base_d * (1.0 + rng.normal(0.0, noise_scale, base_d.shape)), 50.0, 400.0)
		p = np.roll(base_p, int(rng.integers(-2, 3)))
		out.append((d, p))
	return out


# ── DRL training ────────────────────────────────────────────────────────────
def _norm_env(cfg):
	venv = DummyVecEnv([lambda: PaperInspiredDynamicLinepackEnv(config=cfg)])
	return VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=GAMMA)


def train_rl(name, cfg, steps, seed, lr=LR):
	"""Train one DRL agent (reward-normalized env); returns (model, train_seconds)."""
	t0 = time.time()
	env = _norm_env(cfg)
	common = dict(verbose=0, seed=seed, gamma=GAMMA, policy_kwargs=dict(net_arch=NET_ARCH))
	if name == "SAC":
		model = SAC("MlpPolicy", env, learning_rate=lr, **common)
	elif name == "TD3":
		noise = NormalActionNoise(mean=np.zeros(1), sigma=0.2 * np.ones(1))
		model = TD3("MlpPolicy", env, learning_rate=lr, action_noise=noise, **common)
	elif name == "PPO":
		model = PPO("MlpPolicy", env, learning_rate=3e-4, gamma=GAMMA, seed=seed,
		            verbose=0, policy_kwargs=dict(net_arch=NET_ARCH))
	else:
		raise ValueError(name)
	model.learn(total_timesteps=steps)
	return model, time.time() - t0


# ── evaluation record helper ──────────────────────────────────────────────────
def _rec(method, seed, kind, sid, m, t):
	return dict(method=method, seed=seed, kind=kind, scenario=sid,
	            cost=m["total_cost"], viol=m["violation_hours"],
	            eff=m["mean_eff"], tlp_gap=m["terminal_linepack_gap"], time=t)


def run(args):
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	cfg_eval = EnvConfig(noise_scale=0.0)          # deterministic env for scoring
	cfg_train = EnvConfig(noise_scale=args.train_noise)  # noisy env for DRL training
	eval_env = PaperInspiredDynamicLinepackEnv(config=cfg_eval)
	nominal_d = np.asarray(eval_env.demand_series, dtype=np.float64)
	nominal_p = np.asarray(eval_env.tou_price_series, dtype=np.float64)
	scenarios = make_scenarios(eval_env, args.perturb, args.scenario_noise, seed=123)

	records = []
	dispatch_curves = {}   # method -> records list (nominal, seed 0) for the dispatch figure

	# ---- classical planners (deterministic; seed only varies GA) ----
	# rule
	t0 = time.time(); sched = rule_based_schedule(eval_env, nominal_p)
	m = evaluate_schedule(eval_env, sched, nominal_d, nominal_p); t = time.time() - t0
	records.append(_rec("Rule", 0, "nominal", -1, m, t)); dispatch_curves["Rule"] = m["records"]
	for sid, (d, p) in enumerate(scenarios):
		records.append(_rec("Rule", 0, "perturb", sid, evaluate_schedule(eval_env, sched, d, p), t))

	# GA (plan on nominal, seed varies for error bars)
	for seed in range(args.seeds):
		t0 = time.time(); sched = ga_schedule(eval_env, nominal_d, nominal_p, pop=args.ga_pop, gens=args.ga_gens, seed=seed)
		t = time.time() - t0
		m = evaluate_schedule(eval_env, sched, nominal_d, nominal_p)
		records.append(_rec("GA", seed, "nominal", -1, m, t))
		if seed == 0: dispatch_curves["GA"] = m["records"]
		for sid, (d, p) in enumerate(scenarios):
			records.append(_rec("GA", seed, "perturb", sid, evaluate_schedule(eval_env, sched, d, p), t))

	# MPC (nominal-model DP feedback policy, closed-loop)
	t0 = time.time(); m = run_mpc(eval_env, nominal_d, nominal_p, nominal_d, nominal_p, n_p=DP_NP, n_a=DP_NA); t = time.time() - t0
	records.append(_rec("MPC", 0, "nominal", -1, m, t)); dispatch_curves["MPC"] = m["records"]
	for sid, (d, p) in enumerate(scenarios):
		records.append(_rec("MPC", 0, "perturb", sid, run_mpc(eval_env, nominal_d, nominal_p, d, p, n_p=DP_NP, n_a=DP_NA), t))

	# DP-oracle (perfect foresight on the realized scenario)
	t0 = time.time(); m = evaluate_schedule(eval_env, solve_dp(eval_env, nominal_d, nominal_p, n_p=DP_NP, n_a=DP_NA), nominal_d, nominal_p); t = time.time() - t0
	records.append(_rec("DP-oracle", 0, "nominal", -1, m, t)); dispatch_curves["DP-oracle"] = m["records"]
	for sid, (d, p) in enumerate(scenarios):
		records.append(_rec("DP-oracle", 0, "perturb", sid, evaluate_schedule(eval_env, solve_dp(eval_env, d, p, n_p=DP_NP, n_a=DP_NA), d, p), t))

	# ---- DRL agents (trained per seed) ----
	for seed in range(args.seeds):
		trained = {}
		for name in ["SAC", "TD3", "PPO"]:
			model, ttrain = train_rl(name, cfg_train, args.steps, seed, lr=args.lr)
			trained[name] = (model, ttrain)
		csac, wrap = train_constrained_sac(cfg_train, total_timesteps=args.steps, seed=seed,
		                                   lam_init=args.lam_init, dual_lr=args.dual_lr, lam_max=args.lam_max,
		                                   net_arch=NET_ARCH, learning_rate=args.lr, gamma=GAMMA)
		# include final lambda for logging
		trained["Constrained-SAC"] = (csac, None)

		for name, (model, ttrain) in trained.items():
			t0 = time.time(); m = evaluate_model(eval_env, model, nominal_d, nominal_p); infer = time.time() - t0
			rec = _rec(name, seed, "nominal", -1, m, infer)
			rec["train_time"] = ttrain
			if name == "Constrained-SAC": rec["lambda"] = wrap.lam
			records.append(rec)
			if seed == 0: dispatch_curves[name] = m["records"]
			for sid, (d, p) in enumerate(scenarios):
				records.append(_rec(name, seed, "perturb", sid, evaluate_model(eval_env, model, d, p), infer))

	_save_and_plot(records, dispatch_curves, eval_env, args)
	return records


# ── aggregation, CSV, figures ─────────────────────────────────────────────────
METHOD_ORDER = ["Rule", "GA", "PPO", "TD3", "SAC", "Constrained-SAC", "MPC", "DP-oracle"]


def _agg(records, kind):
	out = {}
	for method in METHOD_ORDER:
		rows = [r for r in records if r["method"] == method and r["kind"] == kind]
		if not rows:
			continue
		costs = np.array([r["cost"] for r in rows]); viols = np.array([r["viol"] for r in rows])
		effs = np.array([r["eff"] for r in rows])
		out[method] = dict(cost_mean=costs.mean(), cost_std=costs.std(),
		                   viol_mean=viols.mean(), viol_max=int(viols.max()),
		                   eff_mean=effs.mean(), n=len(rows))
	return out


def _save_and_plot(records, dispatch_curves, env, args):
	import csv
	# raw
	keys = ["method", "seed", "kind", "scenario", "cost", "viol", "eff", "tlp_gap", "time"]
	with open(OUT_DIR / "results_raw.csv", "w", newline="") as f:
		w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
		for r in records:
			w.writerow({k: r.get(k, "") for k in keys})

	nom = _agg(records, "nominal"); rob = _agg(records, "perturb")
	with open(OUT_DIR / "results_summary.csv", "w", newline="") as f:
		w = csv.writer(f)
		w.writerow(["method", "nominal_cost_mean", "nominal_cost_std", "nominal_viol",
		            "robust_cost_mean", "robust_cost_std", "robust_viol_mean", "mean_eff"])
		for method in METHOD_ORDER:
			if method not in nom:
				continue
			n = nom[method]; r = rob.get(method, {})
			w.writerow([method, f"{n['cost_mean']:.3f}", f"{n['cost_std']:.3f}", f"{n['viol_mean']:.2f}",
			            f"{r.get('cost_mean', float('nan')):.3f}", f"{r.get('cost_std', float('nan')):.3f}",
			            f"{r.get('viol_mean', float('nan')):.2f}", f"{n['eff_mean']:.3f}"])

	# Fig 1: nominal cost bar with error + violation annotation
	methods = [m for m in METHOD_ORDER if m in nom]
	costs = [nom[m]["cost_mean"] for m in methods]; errs = [nom[m]["cost_std"] for m in methods]
	viols = [nom[m]["viol_mean"] for m in methods]
	fig, ax = plt.subplots(figsize=(9, 5))
	bars = ax.bar(methods, costs, yerr=errs, capsize=4,
	              color=["#bbb" if m != "Constrained-SAC" else "#d1495b" for m in methods])
	for b, v in zip(bars, viols):
		ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"viol={v:.1f}",
		        ha="center", va="bottom", fontsize=7)
	ax.set_ylabel("24h electricity cost"); ax.set_title("Nominal dispatch cost (lower better)")
	plt.xticks(rotation=20); plt.tight_layout(); fig.savefig(OUT_DIR / "fig_cost_nominal.png", dpi=130); plt.close(fig)

	# Fig 2: dispatch curves, Constrained-SAC vs DP-oracle
	_plot_dispatch(env, dispatch_curves)

	# Fig 3: robustness — cost mean±std under perturbation
	methods_r = [m for m in METHOD_ORDER if m in rob]
	rc = [rob[m]["cost_mean"] for m in methods_r]; re_ = [rob[m]["cost_std"] for m in methods_r]
	rv = [rob[m]["viol_mean"] for m in methods_r]
	fig, ax = plt.subplots(figsize=(9, 5))
	bars = ax.bar(methods_r, rc, yerr=re_, capsize=4,
	              color=["#bbb" if m != "Constrained-SAC" else "#d1495b" for m in methods_r])
	for b, v in zip(bars, rv):
		ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"viol={v:.1f}",
		        ha="center", va="bottom", fontsize=7)
	ax.set_ylabel("24h electricity cost"); ax.set_title(f"Robustness under {args.perturb} perturbed scenarios")
	plt.xticks(rotation=20); plt.tight_layout(); fig.savefig(OUT_DIR / "fig_robustness.png", dpi=130); plt.close(fig)

	print(f"\nSaved CSVs + figures to {OUT_DIR}/")
	_print_table(nom, rob)


def _plot_dispatch(env, curves):
	want = [m for m in ["Constrained-SAC", "DP-oracle", "MPC"] if m in curves]
	if not want:
		return
	fig, axes = plt.subplots(2, 2, figsize=(12, 8), tight_layout=True)
	for m in want:
		recs = curves[m]; hours = [r["hour"] for r in recs]
		axes[0, 0].plot(hours, [r["alpha"] for r in recs], marker="o", ms=3, label=m)
		axes[0, 1].plot(hours, [r["p3"] for r in recs], marker="o", ms=3, label=m)
		axes[1, 0].plot(hours, [r["linepack"] for r in recs], marker="o", ms=3, label=m)
		axes[1, 1].plot(hours, np.cumsum([r["purchase_cost"] for r in recs]), marker="o", ms=3, label=m)
	axes[0, 1].axhline(env.p_min_safe, color="r", ls="--", lw=.8); axes[0, 1].axhline(env.p_max_safe, color="g", ls="--", lw=.8)
	for ax, ttl, yl in [(axes[0, 0], "Compressor action alpha", "alpha"),
	                    (axes[0, 1], "Demand-node pressure p3", "bar"),
	                    (axes[1, 0], "Linepack", "mass"),
	                    (axes[1, 1], "Cumulative electricity cost", "cost")]:
		ax.set_title(ttl); ax.set_xlabel("Hour"); ax.set_ylabel(yl); ax.grid(True); ax.legend(fontsize=7)
	fig.savefig(OUT_DIR / "fig_dispatch.png", dpi=130); plt.close(fig)


def _print_table(nom, rob):
	print(f"\n{'method':<17}{'nom_cost':>12}{'nom_viol':>10}{'rob_cost':>14}{'rob_viol':>10}{'eff':>7}")
	print("-" * 70)
	for m in METHOD_ORDER:
		if m not in nom:
			continue
		n = nom[m]; r = rob.get(m, {})
		print(f"{m:<17}{n['cost_mean']:>8.2f}±{n['cost_std']:<3.1f}{n['viol_mean']:>10.1f}"
		      f"{r.get('cost_mean', float('nan')):>10.2f}±{r.get('cost_std', 0):<3.1f}{r.get('viol_mean', float('nan')):>10.1f}{n['eff_mean']:>7.3f}")


def parse_args():
	p = argparse.ArgumentParser()
	p.add_argument("--steps", type=int, default=60000)
	p.add_argument("--seeds", type=int, default=3)
	p.add_argument("--perturb", type=int, default=8)
	p.add_argument("--lr", type=float, default=3e-4)
	p.add_argument("--train-noise", type=float, default=0.08)
	p.add_argument("--scenario-noise", type=float, default=0.10)
	p.add_argument("--ga-pop", type=int, default=80)
	p.add_argument("--ga-gens", type=int, default=80)
	p.add_argument("--lam-init", type=float, default=50.0)
	p.add_argument("--dual-lr", type=float, default=1.0)
	p.add_argument("--lam-max", type=float, default=1500.0)
	p.add_argument("--quick", action="store_true", help="tiny budget for debugging the harness")
	a = p.parse_args()
	if a.quick:
		a.steps, a.seeds, a.perturb, a.ga_pop, a.ga_gens = 4000, 1, 3, 30, 20
	return a


if __name__ == "__main__":
	args = parse_args()
	print(f"Experiment config: steps={args.steps} seeds={args.seeds} perturb={args.perturb}")
	t0 = time.time()
	run(args)
	print(f"\nTotal wall time: {(time.time() - t0) / 60:.1f} min")
