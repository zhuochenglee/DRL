"""Dynamic-programming benchmark (near-global optimum) for the 24h dispatch.

Backward value iteration over a discretized (p2, p3) pressure state with a
discretized discharge-ratio action. The compressor power and the Eq. 20
speed/surge penalties are precomputed with the shared compressor model
(envs/compressor.py), so DP optimizes exactly the objective the environment
scores. The resulting alpha schedule is then executed on the real env.
"""

from __future__ import annotations

import numpy as np

from envs import compressor
from envs.pipeline_env import EnvConfig, PaperInspiredDynamicLinepackEnv


def _compressor_tables(env, p_grid, a_grid):
	"""Precompute (n_p, n_a) tables of power_kwh, omega_raw, phi, head.

	Power and the compressor operating point depend only on the *current* p2 and
	the action alpha (not on p3), so a 2-D table is exact and cheap.
	"""
	p = compressor.CompressorParams.from_env(env)
	k0 = float(env.K[int(getattr(env, "source_pipe", 0))])
	dt = float(env.dt_hour)
	n_p, n_a = len(p_grid), len(a_grid)
	POW = np.zeros((n_p, n_a))
	ORAW = np.zeros((n_p, n_a))
	PHI = np.zeros((n_p, n_a))
	HEAD = np.zeros((n_p, n_a))
	for i, p2 in enumerate(p_grid):
		for j, a in enumerate(a_grid):
			p_src = env.p0_ref * a
			dp20 = p_src * p_src - p2 * p2
			q0 = np.sign(dp20) * np.sqrt(abs(dp20) / k0)
			pw, om, et, hd, ph, oraw = compressor.power_kwh(abs(q0), float(a), p, dt)
			POW[i, j] = pw
			ORAW[i, j] = oraw
			PHI[i, j] = ph
			HEAD[i, j] = hd
	return POW, ORAW, PHI, HEAD


def build_dp_policy_nd(env, demand, prices, n_p=31, n_a=31):
	"""General N-dimensional DP for an arbitrary single-compressor topology.

	State = the internal-node pressures (N = env.n_internal). The compressor cost
	depends only on the source-downstream node's pressure and the action, so it stays a
	2-D precomputed table; only the value function is interpolated in N-D. `demand` is
	the TOTAL hourly profile, split across nodes by env._spatial_frac (as in the env).
	Returns (p_grid, a_grid, best_action) with best_action shape (horizon,)+(n_p,)*N.
	"""
	from scipy.interpolate import RegularGridInterpolator

	cfg = env
	N, H, n_pipes = env.n_internal, env.horizon, env.n_pipes
	demand_total = np.asarray(demand, dtype=np.float64)
	prices = np.asarray(prices, dtype=np.float64)
	p_grid = np.linspace(cfg.p_hard_min, cfg.p_hard_max, n_p)
	a_grid = np.linspace(float(cfg.action_space.low[0]), float(cfg.action_space.high[0]), n_a)
	beta, K, A = np.asarray(env.beta), np.asarray(env.K), np.asarray(env.A)
	A_int, dt, down = env.A_internal, float(env.dt_hour), env._source_down_node
	froms = [int(np.where(A[:, e] == -1)[0][0]) for e in range(n_pipes)]
	tos = [int(np.where(A[:, e] == +1)[0][0]) for e in range(n_pipes)]
	spatial = env._spatial_frac
	target = float(np.sum(beta * 100.0))

	POW, ORAW, PHI, HEAD = _compressor_tables(env, p_grid, a_grid)   # (n_p, n_a) over down-node p

	def along_down(vec):                       # reshape length-n_p vector to vary along `down` axis
		shape = [1] * N; shape[down] = n_p
		return vec.reshape(shape)

	grids = np.meshgrid(*([p_grid] * N), indexing="ij")
	V_next = np.zeros((n_p,) * N)
	best_action = np.zeros((H,) + (n_p,) * N, dtype=np.int16)

	for hour in range(H - 1, -1, -1):
		dvec = spatial[:, hour] * demand_total[hour]
		price = float(prices[hour])
		rgi = RegularGridInterpolator(tuple([p_grid] * N), V_next, bounds_error=False, fill_value=None)
		vals = np.empty((n_a,) + (n_p,) * N)
		for ja, a in enumerate(a_grid):
			p_src = cfg.p0_ref * a
			node_p = lambda idx: (p_src if idx == 0 else grids[idx - 1])
			q = []
			for e in range(n_pipes):
				dp2 = node_p(froms[e]) ** 2 - node_p(tos[e]) ** 2
				q.append(np.sign(dp2) * np.sqrt(np.abs(dp2) / K[e]))
			pnext = []
			for i in range(N):
				net = sum(A_int[i, e] * q[e] for e in range(n_pipes)) - dvec[i]
				m = np.clip(beta[i] * grids[i] + dt * net, beta[i] * cfg.p_hard_min, beta[i] * cfg.p_hard_max)
				pnext.append(m / beta[i])
			running = along_down(HEAD[:, ja]) > 0.0
			stage = along_down(POW[:, ja]) * price
			stage = stage + cfg.w_speed * (np.maximum(env.omega_min - along_down(ORAW[:, ja]), 0.0)
			                               + np.maximum(along_down(ORAW[:, ja]) - env.omega_max, 0.0)) * running
			stage = stage + cfg.w_surge * (np.maximum(env.phi_lower - along_down(PHI[:, ja]), 0.0)
			                               + np.maximum(along_down(PHI[:, ja]) - env.phi_upper, 0.0)) * running
			stage = stage + sum(np.where(pn < cfg.p_min_safe, cfg.w_pressure * (cfg.p_min_safe - pn), 0.0)
			                    + np.where(pn > cfg.p_max_safe, cfg.w_pressure * (pn - cfg.p_max_safe), 0.0) for pn in pnext)
			stage = stage + sum(np.where(np.abs(q[e]) > env.q_max[e], cfg.w_flow * (np.abs(q[e]) - env.q_max[e]), 0.0)
			                    for e in range(n_pipes))
			if hour == H - 1:
				lp = sum(beta[i] * pnext[i] for i in range(N))
				stage = stage + cfg.w_terminal_linepack * np.abs(lp - target) / max(target, 1e-6)
			pts = np.stack([pn.ravel() for pn in pnext], axis=-1)
			vals[ja] = np.broadcast_to(stage, (n_p,) * N) + rgi(pts).reshape((n_p,) * N)
		best_action[hour] = np.argmin(vals, axis=0).astype(np.int16)
		V_next = np.min(vals, axis=0)
	return p_grid, a_grid, best_action


def greedy_alpha(env, p_grid, a_grid, best_action, hour):
	"""Look up the DP greedy action for the env's current internal-node pressures (any N)."""
	dp_val = float(p_grid[1] - p_grid[0]); n_p = len(p_grid)
	idx = tuple(int(np.clip(round((float(p) - p_grid[0]) / dp_val), 0, n_p - 1)) for p in env.P_internal)
	return float(a_grid[int(best_action[(hour,) + idx])])


def build_dp_policy(env, demand, prices, n_p=45, n_a=31):
	"""Backward value iteration; returns (p_grid, a_grid, best_action[horizon,n_p,n_p]).

	The greedy action table is a feedback policy over the discretized state, reused
	open-loop (perfect-foresight DP benchmark) and closed-loop (certainty-equivalent
	MPC executed against a possibly-perturbed environment).

	For non-2-node topologies, dispatches to the general N-D DP (capped grid for tractability).
	"""
	if env.n_internal != 2:
		return build_dp_policy_nd(env, demand, prices, n_p=min(n_p, 31), n_a=n_a)
	cfg = env
	horizon = int(cfg.horizon)
	demand = np.asarray(demand, dtype=np.float64)
	prices = np.asarray(prices, dtype=np.float64)

	p_grid = np.linspace(cfg.p_hard_min, cfg.p_hard_max, n_p)
	a_grid = np.linspace(float(cfg.action_space.low[0]), float(cfg.action_space.high[0]), n_a)
	dp_val = float(p_grid[1] - p_grid[0])

	beta2, beta3 = float(cfg.beta[0]), float(cfg.beta[1])
	k0, k1 = float(cfg.K[0]), float(cfg.K[1])
	dt = float(cfg.dt_hour)
	linepack_target = float(np.sum(cfg.beta * np.array([100.0, 100.0])))

	# State/action broadcast grids: P2 (n_p,1,1), P3 (1,n_p,1), ALPHA (1,1,n_a)
	P2 = p_grid[:, None, None]
	P3 = p_grid[None, :, None]
	ALPHA = a_grid[None, None, :]
	P_SRC = cfg.p0_ref * ALPHA

	# Precomputed compressor tables -> broadcast over the p3 axis.
	POW, ORAW, PHI, HEAD = _compressor_tables(env, p_grid, a_grid)
	pow_b = POW[:, None, :]
	running = (HEAD[:, None, :] > 0.0)
	speed_pen = cfg.w_speed * (
		np.maximum(cfg.omega_min - ORAW[:, None, :], 0.0)
		+ np.maximum(ORAW[:, None, :] - cfg.omega_max, 0.0)
	) * running
	surge_pen = cfg.w_surge * (
		np.maximum(cfg.phi_lower - PHI[:, None, :], 0.0)
		+ np.maximum(PHI[:, None, :] - cfg.phi_upper, 0.0)
	) * running

	V_next = np.zeros((n_p, n_p), dtype=np.float64)
	best_action = np.zeros((horizon, n_p, n_p), dtype=np.int32)

	for hour in range(horizon - 1, -1, -1):
		d = float(demand[hour])
		price = float(prices[hour])

		dp20 = P_SRC * P_SRC - P2 * P2
		dp21 = P2 * P2 - P3 * P3
		q0 = np.sign(dp20) * np.sqrt(np.abs(dp20) / k0)
		q1 = np.sign(dp21) * np.sqrt(np.abs(dp21) / k1)

		m2_next = np.clip(beta2 * P2 + dt * (q0 - q1), beta2 * cfg.p_hard_min, beta2 * cfg.p_hard_max)
		m3_next = np.clip(beta3 * P3 + dt * (q1 - d), beta3 * cfg.p_hard_min, beta3 * cfg.p_hard_max)
		p2_next = m2_next / beta2
		p3_next = m3_next / beta3

		purchase_cost = pow_b * price

		pressure_penalty = (
			np.where(p2_next < cfg.p_min_safe, cfg.w_pressure * (cfg.p_min_safe - p2_next), 0.0)
			+ np.where(p2_next > cfg.p_max_safe, cfg.w_pressure * (p2_next - cfg.p_max_safe), 0.0)
			+ np.where(p3_next < cfg.p_min_safe, cfg.w_pressure * (cfg.p_min_safe - p3_next), 0.0)
			+ np.where(p3_next > cfg.p_max_safe, cfg.w_pressure * (p3_next - cfg.p_max_safe), 0.0)
		)
		flow_penalty = (
			np.where(np.abs(q0) > cfg.q1_max, cfg.w_flow * (np.abs(q0) - cfg.q1_max), 0.0)
			+ np.where(np.abs(q1) > cfg.q2_max, cfg.w_flow * (np.abs(q1) - cfg.q2_max), 0.0)
		)
		stage_cost = purchase_cost + pressure_penalty + flow_penalty + speed_pen + surge_pen

		if hour == horizon - 1:
			linepack = m2_next + m3_next
			terminal_pen = cfg.w_terminal_linepack * np.abs(linepack - linepack_target) / max(linepack_target, 1e-6)
			stage_cost = stage_cost + terminal_pen

		idx2 = np.clip((p2_next - p_grid[0]) / dp_val, 0.0, n_p - 1.001)
		idx3 = np.clip((p3_next - p_grid[0]) / dp_val, 0.0, n_p - 1.001)
		i2 = np.clip(idx2.astype(int), 0, n_p - 2)
		i3 = np.clip(idx3.astype(int), 0, n_p - 2)
		f2 = idx2 - i2
		f3 = idx3 - i3

		future = (
			(1.0 - f2) * (1.0 - f3) * V_next[i2, i3]
			+ f2 * (1.0 - f3) * V_next[i2 + 1, i3]
			+ (1.0 - f2) * f3 * V_next[i2, i3 + 1]
			+ f2 * f3 * V_next[i2 + 1, i3 + 1]
		)

		total_cost = stage_cost + future
		best_action[hour] = np.argmin(total_cost, axis=2)
		V_next = np.min(total_cost, axis=2)

	return p_grid, a_grid, best_action


def solve_dp(env, demand, prices, n_p=45, n_a=31):
	"""Perfect-foresight DP benchmark: optimal open-loop alpha schedule for the given
	(realized) demand/prices. Builds the DP feedback policy on that scenario and rolls
	it out greedily through the env (topology-general), returning the alpha schedule."""
	p_grid, a_grid, best_action = build_dp_policy(env, demand, prices, n_p=n_p, n_a=n_a)
	env.reset(options={"demand": np.asarray(demand, dtype=np.float64),
	                   "prices": np.asarray(prices, dtype=np.float64)})
	schedule = []
	for hour in range(env.horizon):
		a = greedy_alpha(env, p_grid, a_grid, best_action, hour)
		schedule.append(a)
		_, _, terminated, truncated, _ = env.step(np.array([a], dtype=np.float32))
		if terminated or truncated:
			break
	return np.array(schedule, dtype=np.float64)


if __name__ == "__main__":
	env = PaperInspiredDynamicLinepackEnv(config=EnvConfig())
	schedule = solve_dp(env, env.demand_series, env.tou_price_series)

	obs, _ = env.reset()
	total_cost = 0.0
	viol = 0
	for hour in range(env.horizon):
		obs, reward, terminated, truncated, info = env.step(np.array([schedule[hour]], dtype=np.float32))
		total_cost += info["purchase_cost"]
		viol += int(info["violation"])
		if terminated:
			break
	print(f"DP schedule alpha: {np.round(schedule, 3)}")
	print(f"DP total electricity cost: {total_cost:.2f} | violation hours: {viol}")
