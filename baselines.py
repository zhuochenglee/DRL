"""Baseline controllers and shared evaluation utilities.

All methods are scored on the same environment so the comparison is fair:
  - rule_based_schedule : naive price-aware open-loop heuristic
  - ga_schedule         : genetic-algorithm open-loop optimizer
  - solve_dp (DP.py)    : perfect-foresight DP benchmark (near-global optimum)
  - run_mpc             : receding-horizon control = nominal-model DP feedback
                          policy executed closed-loop on the realized environment
  - evaluate_schedule / evaluate_model : roll out and collect metrics

Planning is done on a *forecast* (nominal) demand/price; evaluation can be on a
*realized* (possibly perturbed) demand/price passed through reset(options=...).
"""

from __future__ import annotations

import numpy as np

from DP import build_dp_policy


def _metrics(records, total_reward):
	return {
		"total_cost": float(sum(r["purchase_cost"] for r in records)),
		"total_power": float(sum(r["power_kwh"] for r in records)),
		"violation_hours": int(sum(int(r["violation"]) for r in records)),
		"mean_eff": float(np.mean([r["compressor_efficiency"] for r in records])),
		"mean_alpha": float(np.mean([r["alpha"] for r in records])),
		"terminal_linepack_gap": float(records[-1]["terminal_linepack_gap"]),
		"total_reward": float(total_reward),
		"records": records,
	}


def _reset_options(demand, prices):
	options = {}
	if demand is not None:
		options["demand"] = np.asarray(demand, dtype=np.float64)
	if prices is not None:
		options["prices"] = np.asarray(prices, dtype=np.float64)
	return options or None


def evaluate_schedule(env, schedule, demand=None, prices=None):
	"""Open-loop: apply a fixed alpha schedule on the (realized) environment."""
	obs, _ = env.reset(options=_reset_options(demand, prices))
	records, total_reward = [], 0.0
	for hour in range(env.horizon):
		a = float(np.clip(schedule[hour], env.action_space.low[0], env.action_space.high[0]))
		obs, reward, terminated, truncated, info = env.step(np.array([a], dtype=np.float32))
		records.append(info)
		total_reward += float(reward)
		if terminated or truncated:
			break
	return _metrics(records, total_reward)


def evaluate_model(env, model, demand=None, prices=None):
	"""Closed-loop: roll out a trained SB3 policy (deterministic) on the environment."""
	obs, _ = env.reset(options=_reset_options(demand, prices))
	records, total_reward = [], 0.0
	for _ in range(env.horizon):
		action, _ = model.predict(obs, deterministic=True)
		obs, reward, terminated, truncated, info = env.step(action)
		records.append(info)
		total_reward += float(reward)
		if terminated or truncated:
			break
	return _metrics(records, total_reward)


def rule_based_schedule(env, prices):
	"""Naive reference: compress hard in cheap hours, ease off in expensive hours."""
	prices = np.asarray(prices, dtype=np.float64)
	threshold = float(np.median(prices))
	schedule = np.where(prices <= threshold, 1.55, 1.10)
	schedule[-1] = 1.55  # top up terminal linepack
	return schedule.astype(np.float64)


def ga_schedule(env, demand, prices, pop=60, gens=60, seed=0,
                elite=4, mut_rate=0.18, mut_std=0.08, tour=4):
	"""Genetic-algorithm open-loop optimizer over the 24-step alpha vector."""
	rng = np.random.default_rng(seed)
	lo, hi = float(env.action_space.low[0]), float(env.action_space.high[0])
	H = env.horizon
	population = rng.uniform(lo, hi, size=(pop, H))

	def fitness(ind):
		return evaluate_schedule(env, ind, demand, prices)["total_reward"]

	best, best_fit = None, -np.inf
	for _ in range(gens):
		fits = np.array([fitness(ind) for ind in population])
		order = np.argsort(fits)[::-1]
		if fits[order[0]] > best_fit:
			best_fit = float(fits[order[0]])
			best = population[order[0]].copy()
		next_pop = [population[i].copy() for i in order[:elite]]
		while len(next_pop) < pop:
			ia = rng.integers(0, pop, tour); a = population[ia[np.argmax(fits[ia])]]
			ib = rng.integers(0, pop, tour); b = population[ib[np.argmax(fits[ib])]]
			mask = rng.random(H) < 0.5
			child = np.where(mask, a, b)
			child = child + (rng.random(H) < mut_rate) * rng.normal(0.0, mut_std, H)
			next_pop.append(np.clip(child, lo, hi))
		population = np.asarray(next_pop)
	return best


def run_mpc(env, nominal_demand, nominal_prices, demand=None, prices=None, n_p=45, n_a=31):
	"""Receding-horizon control via the certainty-equivalent (nominal-model) DP
	feedback policy, executed closed-loop on the realized environment."""
	p_grid, a_grid, best_action = build_dp_policy(env, nominal_demand, nominal_prices, n_p=n_p, n_a=n_a)
	dp_val = float(p_grid[1] - p_grid[0])
	n_p = len(p_grid)

	obs, _ = env.reset(options=_reset_options(demand, prices))
	records, total_reward = [], 0.0
	for hour in range(env.horizon):
		p2, p3 = float(env.P_internal[0]), float(env.P_internal[1])
		i2 = int(np.clip(round((p2 - p_grid[0]) / dp_val), 0, n_p - 1))
		i3 = int(np.clip(round((p3 - p_grid[0]) / dp_val), 0, n_p - 1))
		a = float(a_grid[int(best_action[hour, i2, i3])])
		obs, reward, terminated, truncated, info = env.step(np.array([a], dtype=np.float32))
		records.append(info)
		total_reward += float(reward)
		if terminated or truncated:
			break
	return _metrics(records, total_reward)


if __name__ == "__main__":
	import time
	from envs.pipeline_env import EnvConfig, PaperInspiredDynamicLinepackEnv
	from DP import solve_dp

	env = PaperInspiredDynamicLinepackEnv(config=EnvConfig())
	d, pr = env.demand_series, env.tou_price_series

	t = time.time(); rule = evaluate_schedule(env, rule_based_schedule(env, pr), d, pr); t_rule = time.time() - t
	t = time.time(); ga = evaluate_schedule(env, ga_schedule(env, d, pr), d, pr); t_ga = time.time() - t
	t = time.time(); dp = evaluate_schedule(env, solve_dp(env, d, pr), d, pr); t_dp = time.time() - t
	t = time.time(); mpc = run_mpc(env, d, pr, d, pr); t_mpc = time.time() - t

	print(f"{'method':<8} {'cost':>8} {'viol_h':>7} {'mean_eff':>9} {'time_s':>8}")
	for name, m, tt in [("rule", rule, t_rule), ("ga", ga, t_ga), ("dp", dp, t_dp), ("mpc", mpc, t_mpc)]:
		print(f"{name:<8} {m['total_cost']:>8.2f} {m['violation_hours']:>7d} {m['mean_eff']:>9.3f} {tt:>8.2f}")
