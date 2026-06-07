import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from envs.pipeline_env import EnvConfig, PaperInspiredDynamicLinepackEnv


TB_LOG_DIR = "models/tensorboard/dp_papercode"


def dp_optimize_schedule(env, demand, prices, n_p=36, n_a=31):
	"""Discrete-state DP on (p2, p3) with bilinear interpolation of value function."""
	cfg = env
	horizon = int(cfg.horizon)
	p_grid = np.linspace(cfg.p_hard_min, cfg.p_hard_max, n_p)
	a_grid = np.linspace(cfg.action_space.low[0], cfg.action_space.high[0], n_a)
	dp_val = float(p_grid[1] - p_grid[0])

	beta2, beta3 = float(cfg.beta[0]), float(cfg.beta[1])
	k0, k1 = float(cfg.K[0]), float(cfg.K[1])
	dt = float(cfg.dt_hour)
	linepack_target = float(np.sum(cfg.beta * np.array([100.0, 100.0], dtype=np.float64)))

	P2 = p_grid[:, None, None]
	P3 = p_grid[None, :, None]
	ALPHA = a_grid[None, None, :]
	P_SRC = cfg.p0_ref * ALPHA

	V_next = np.zeros((n_p, n_p), dtype=np.float64)
	best_action = np.zeros((horizon, n_p, n_p), dtype=np.int32)

	for hour in range(horizon - 1, -1, -1):
		d = float(demand[hour])
		price = float(prices[hour])

		dp20 = P_SRC * P_SRC - P2 * P2
		dp21 = P2 * P2 - P3 * P3
		q0 = np.sign(dp20) * np.sqrt(np.abs(dp20) / k0)
		q1 = np.sign(dp21) * np.sqrt(np.abs(dp21) / k1)

		m2 = beta2 * P2
		m3 = beta3 * P3
		m2_next = np.clip(m2 + dt * (q0 - q1), beta2 * cfg.p_hard_min, beta2 * cfg.p_hard_max)
		m3_next = np.clip(m3 + dt * (q1 - d), beta3 * cfg.p_hard_min, beta3 * cfg.p_hard_max)
		p2_next = m2_next / beta2
		p3_next = m3_next / beta3

		exp_term = (cfg.kappa - 1.0) / cfg.kappa
		head_factor = np.maximum(np.power(ALPHA, exp_term) - 1.0, 0.0)
		norm_speed = np.clip(0.58 + 1.25 * head_factor, 0.0, 1.0)
		omega = cfg.omega_min + norm_speed * (cfg.omega_max - cfg.omega_min)
		n = omega / cfg.omega_max
		eff = np.clip(np.polyval(cfg.poly_eta, n) / 100.0, 0.60, 0.88)
		q0_abs = np.abs(q0)
		power_kw = cfg.power_coeff * q0_abs * head_factor * (omega / cfg.omega_max) / np.maximum(eff, 1e-6)
		purchase_cost = power_kw * cfg.dt_hour * price

		pressure_penalty = (
			np.where(p2_next < cfg.p_min_safe, cfg.w_pressure * (cfg.p_min_safe - p2_next), 0.0)
			+ np.where(p2_next > cfg.p_max_safe, cfg.w_pressure * (p2_next - cfg.p_max_safe), 0.0)
			+ np.where(p3_next < cfg.p_min_safe, cfg.w_pressure * (cfg.p_min_safe - p3_next), 0.0)
			+ np.where(p3_next > cfg.p_max_safe, cfg.w_pressure * (p3_next - cfg.p_max_safe), 0.0)
		)
		flow_penalty = (
			np.where(q0_abs > cfg.q1_max, cfg.w_flow * (q0_abs - cfg.q1_max), 0.0)
			+ np.where(np.abs(q1) > cfg.q2_max, cfg.w_flow * (np.abs(q1) - cfg.q2_max), 0.0)
		)
		stage_cost = purchase_cost + pressure_penalty + flow_penalty

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

	schedule = []
	p2, p3 = 100.0, 100.0
	for hour in range(horizon):
		i2 = int(np.clip(round((p2 - p_grid[0]) / dp_val), 0, n_p - 1))
		i3 = int(np.clip(round((p3 - p_grid[0]) / dp_val), 0, n_p - 1))
		a = float(a_grid[best_action[hour, i2, i3]])
		schedule.append(a)

		dp20 = (cfg.p0_ref * a) ** 2 - p2 ** 2
		dp21 = p2 ** 2 - p3 ** 2
		q0 = np.sign(dp20) * np.sqrt(np.abs(dp20) / k0)
		q1 = np.sign(dp21) * np.sqrt(np.abs(dp21) / k1)
		m2 = np.clip(beta2 * p2 + dt * (q0 - q1), beta2 * cfg.p_hard_min, beta2 * cfg.p_hard_max)
		m3 = np.clip(beta3 * p3 + dt * (q1 - float(demand[hour])), beta3 * cfg.p_hard_min, beta3 * cfg.p_hard_max)
		p2 = float(m2 / beta2)
		p3 = float(m3 / beta3)

	return np.array(schedule, dtype=np.float64)


def run_schedule(env, action_schedule):
	obs, _ = env.reset()
	records = []
	total_cost = 0.0
	total_power = 0.0
	total_reward = 0.0
	violation_hours = 0

	for hour in range(env.horizon):
		action_value = float(np.clip(action_schedule[hour], env.action_space.low[0], env.action_space.high[0]))
		obs, reward, terminated, truncated, info = env.step(np.array([action_value], dtype=np.float32))
		records.append(info)
		total_cost += float(info["purchase_cost"])
		total_power += float(info["power_kwh"])
		total_reward += float(reward)
		if info["violation"]:
			violation_hours += 1
		if terminated or truncated:
			break

	return {
		"records": records,
		"total_cost": total_cost,
		"total_power": total_power,
		"total_reward": total_reward,
		"violation_hours": violation_hours,
	}


def build_figure(env, records):
	hours = [r["hour"] for r in records]
	fig, axes = plt.subplots(2, 2, figsize=(12, 8), tight_layout=True)
	fig.suptitle("DP deterministic 24h evaluation", fontsize=11)

	axes[0, 0].plot(hours, [r["alpha"] for r in records], marker="o", markersize=3)
	axes[0, 0].set_title("Compressor action alpha")
	axes[0, 0].set_xlabel("Hour")
	axes[0, 0].set_ylabel("alpha")
	axes[0, 0].grid(True)

	axes[0, 1].plot(hours, [r["p2"] for r in records], label="p2", marker="o", markersize=3)
	axes[0, 1].plot(hours, [r["p3"] for r in records], label="p3", marker="s", markersize=3)
	axes[0, 1].axhline(env.p_min_safe, color="red", linestyle="--", linewidth=0.8, label="p_min_safe")
	axes[0, 1].axhline(env.p_max_safe, color="green", linestyle="--", linewidth=0.8, label="p_max_safe")
	axes[0, 1].set_title("Node pressures")
	axes[0, 1].set_xlabel("Hour")
	axes[0, 1].set_ylabel("Pressure")
	axes[0, 1].legend(fontsize=7)
	axes[0, 1].grid(True)

	axes[1, 0].plot(hours, [r["linepack"] for r in records], color="purple", marker="o", markersize=3)
	axes[1, 0].set_title("Pipeline linepack")
	axes[1, 0].set_xlabel("Hour")
	axes[1, 0].set_ylabel("Mass")
	axes[1, 0].grid(True)

	axes[1, 1].bar(hours, [r["purchase_cost"] for r in records], color="orange")
	axes[1, 1].set_title("Electricity cost per hour")
	axes[1, 1].set_xlabel("Hour")
	axes[1, 1].set_ylabel("Cost")
	axes[1, 1].grid(True, axis="y")

	return fig


def print_dispatch_table(result):
	print("\n24h dispatch simulation")
	print("hour | price | demand | alpha | p2    | p3    | power_kwh | cost")
	print("-" * 78)
	for info in result["records"]:
		print(
			f"{info['hour']:02d}   | {info['price']:>5.2f} | {info['demand']:>6.1f} | {info['alpha']:>5.3f} | "
			f"{info['p2']:>5.1f} | {info['p3']:>5.1f} | {info['power_kwh']:>9.2f} | {info['purchase_cost']:>7.2f}"
		)
	print("-" * 78)
	print(f"Total purchased electricity: {result['total_power']:.2f} kWh")
	print(f"Total electricity cost: {result['total_cost']:.2f}")
	print(f"Total reward: {result['total_reward']:.2f}")
	print(f"Pressure/flow violation hours: {result['violation_hours']}")


def log_to_tensorboard(writer, result, figure):
	step = 0
	writer.add_scalar("dp/episode/total_cost", result["total_cost"], step)
	writer.add_scalar("dp/episode/total_power_kwh", result["total_power"], step)
	writer.add_scalar("dp/episode/total_reward", result["total_reward"], step)
	writer.add_scalar("dp/episode/violation_hours", result["violation_hours"], step)
	writer.add_scalar("dp/episode/mean_alpha", float(np.mean([r["alpha"] for r in result["records"]])), step)
	writer.add_scalar("dp/episode/mean_p2_bar", float(np.mean([r["p2"] for r in result["records"]])), step)
	writer.add_scalar("dp/episode/mean_p3_bar", float(np.mean([r["p3"] for r in result["records"]])), step)
	writer.add_scalar("dp/episode/mean_linepack", float(np.mean([r["linepack"] for r in result["records"]])), step)
	writer.add_figure("dp/eval/24h_curves", figure, step)

	for info in result["records"]:
		h = int(info["hour"])
		writer.add_scalar("dp/hourly/alpha", float(info["alpha"]), h)
		writer.add_scalar("dp/hourly/p2_bar", float(info["p2"]), h)
		writer.add_scalar("dp/hourly/p3_bar", float(info["p3"]), h)
		writer.add_scalar("dp/hourly/linepack", float(info["linepack"]), h)
		writer.add_scalar("dp/hourly/purchase_cost", float(info["purchase_cost"]), h)


if __name__ == "__main__":
	config = EnvConfig(
		noise_scale=0.0,
		horizon=24,
		w_pressure=1000.0,
		w_flow=5.0,
		w_terminal_linepack=250.0,
	)
	env_plan = PaperInspiredDynamicLinepackEnv(config=config)
	env_eval = PaperInspiredDynamicLinepackEnv(config=config)

	schedule = dp_optimize_schedule(
		env=env_plan,
		demand=env_plan.demand_series,
		prices=env_plan.tou_price_series,
		n_p=36,
		n_a=31,
	)
	result = run_schedule(env_eval, schedule)
	fig = build_figure(env_eval, result["records"])

	writer = SummaryWriter(log_dir=TB_LOG_DIR)
	log_to_tensorboard(writer, result, fig)
	writer.flush()
	writer.close()
	plt.close(fig)

	print_dispatch_table(result)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from envs.pipeline_env import EnvConfig, PaperInspiredDynamicLinepackEnv


DP_PRESSURE_GRID_SIZE = 45
DP_ACTION_GRID_SIZE = 31
DP_LOG_DIR = "models/tensorboard/dp_papercode"


def compressor_power_kwh(q0, alpha, cfg):
	"""Vectorized compressor power model aligned with the environment."""
	exp_term = (cfg.kappa - 1.0) / cfg.kappa
	head_factor = np.maximum(np.power(alpha, exp_term) - 1.0, 0.0)
	norm_speed = np.clip(0.58 + 1.25 * head_factor, 0.0, 1.0)
	omega = cfg.omega_min + norm_speed * (cfg.omega_max - cfg.omega_min)
	n = omega / cfg.omega_max
	eff_percent = np.polyval(cfg.poly_eta, n)
	eta = np.clip(eff_percent / 100.0, 0.60, 0.88)
	power_kw = cfg.power_coeff * np.maximum(q0, 0.0) * np.maximum(head_factor, 0.0) * (omega / cfg.omega_max) / np.maximum(eta, 1e-6)
	return power_kw * cfg.dt_hour


def build_dp_policy_table(cfg, n_p=DP_PRESSURE_GRID_SIZE, n_a=DP_ACTION_GRID_SIZE):
	"""Backward dynamic programming on discretized (p2, p3) states."""
	p_grid = np.linspace(cfg.p_hard_min, cfg.p_hard_max, n_p)
	a_grid = np.linspace(1.0, 2.0, n_a)
	dp_val = p_grid[1] - p_grid[0]

	# Broadcast tensors: (n_p, 1, 1), (1, n_p, 1), (1, 1, n_a)
	P2 = p_grid[:, None, None]
	P3 = p_grid[None, :, None]
	ALPHA = a_grid[None, None, :]
	P_SRC = cfg.p0_ref * ALPHA

	beta0 = float(cfg.beta[0])
	beta1 = float(cfg.beta[1])
	m_min0 = beta0 * cfg.p_hard_min
	m_min1 = beta1 * cfg.p_hard_min
	m_max0 = beta0 * cfg.p_hard_max
	m_max1 = beta1 * cfg.p_hard_max

	V_next = np.zeros((n_p, n_p), dtype=np.float64)
	best_action = np.zeros((cfg.horizon, n_p, n_p), dtype=np.int16)
	linepack_target = float(beta0 * 100.0 + beta1 * 100.0)

	for hour in range(cfg.horizon - 1, -1, -1):
		demand = float(cfg.demand_series[hour])
		price = float(cfg.tou_price_series[hour])

		dP0 = P_SRC * P_SRC - P2 * P2
		dP1 = P2 * P2 - P3 * P3
		Q0 = np.sign(dP0) * np.sqrt(np.abs(dP0) / cfg.K[0])
		Q1 = np.sign(dP1) * np.sqrt(np.abs(dP1) / cfg.K[1])

		# m_next = clip(m + dt * net_flow, hard bounds), p_next = m_next / beta
		net2 = Q0 - Q1
		net3 = Q1 - demand
		M2_next = np.clip(beta0 * P2 + cfg.dt_hour * net2, m_min0, m_max0)
		M3_next = np.clip(beta1 * P3 + cfg.dt_hour * net3, m_min1, m_max1)
		P2_next = M2_next / beta0
		P3_next = M3_next / beta1

		power_kwh = compressor_power_kwh(np.abs(Q0), ALPHA, cfg)
		purchase_cost = power_kwh * price

		pressure_penalty = (
			np.where(P2_next < cfg.p_min_safe, cfg.w_pressure * (cfg.p_min_safe - P2_next), 0.0)
			+ np.where(P2_next > cfg.p_max_safe, cfg.w_pressure * (P2_next - cfg.p_max_safe), 0.0)
			+ np.where(P3_next < cfg.p_min_safe, cfg.w_pressure * (cfg.p_min_safe - P3_next), 0.0)
			+ np.where(P3_next > cfg.p_max_safe, cfg.w_pressure * (P3_next - cfg.p_max_safe), 0.0)
		)
		flow_penalty = (
			np.maximum(np.abs(Q0) - cfg.q1_max, 0.0) * cfg.w_flow
			+ np.maximum(np.abs(Q1) - cfg.q2_max, 0.0) * cfg.w_flow
		)
		stage_cost = purchase_cost + pressure_penalty + flow_penalty

		if hour == cfg.horizon - 1:
			linepack = M2_next + M3_next
			terminal_gap = np.abs(linepack - linepack_target)
			stage_cost = stage_cost + cfg.w_terminal_linepack * terminal_gap / max(linepack_target, 1e-6)

		idx2 = np.clip((P2_next - p_grid[0]) / dp_val, 0, n_p - 1.001)
		idx3 = np.clip((P3_next - p_grid[0]) / dp_val, 0, n_p - 1.001)
		i2 = np.clip(idx2.astype(int), 0, n_p - 2)
		i3 = np.clip(idx3.astype(int), 0, n_p - 2)
		f2 = idx2 - i2
		f3 = idx3 - i3

		future = (
			(1 - f2) * (1 - f3) * V_next[i2, i3]
			+ f2 * (1 - f3) * V_next[i2 + 1, i3]
			+ (1 - f2) * f3 * V_next[i2, i3 + 1]
			+ f2 * f3 * V_next[i2 + 1, i3 + 1]
		)

		total = stage_cost + future
		best_action[hour] = np.argmin(total, axis=2)
		V_next = np.min(total, axis=2)

	return p_grid, a_grid, best_action


def run_dp_episode(env, p_grid, a_grid, best_action):
	obs, _ = env.reset()
	dp_val = p_grid[1] - p_grid[0]
	records = []
	total_reward = 0.0
	total_cost = 0.0
	total_power = 0.0
	violation_hours = 0

	for hour in range(env.horizon):
		p2, p3 = float(env.P_internal[0]), float(env.P_internal[1])
		i2 = int(np.clip(round((p2 - p_grid[0]) / dp_val), 0, len(p_grid) - 1))
		i3 = int(np.clip(round((p3 - p_grid[0]) / dp_val), 0, len(p_grid) - 1))
		alpha = float(a_grid[int(best_action[hour, i2, i3])])

		obs, reward, terminated, truncated, info = env.step(np.array([alpha], dtype=np.float32))
		total_reward += float(reward)
		total_cost += float(info["purchase_cost"])
		total_power += float(info["power_kwh"])
		if info["violation"]:
			violation_hours += 1
		records.append(info)

		if terminated or truncated:
			break

	return {
		"records": records,
		"total_reward": total_reward,
		"total_cost": total_cost,
		"total_power": total_power,
		"violation_hours": violation_hours,
	}


def make_figure(env, records):
	hours = [r["hour"] for r in records]
	fig, axes = plt.subplots(2, 2, figsize=(12, 8), tight_layout=True)
	fig.suptitle("DP deterministic 24h evaluation", fontsize=11)

	axes[0, 0].plot(hours, [r["alpha"] for r in records], marker="o", markersize=3)
	axes[0, 0].set_title("Compressor action alpha")
	axes[0, 0].set_xlabel("Hour")
	axes[0, 0].set_ylabel("alpha")
	axes[0, 0].grid(True)

	axes[0, 1].plot(hours, [r["p2"] for r in records], label="p2", marker="o", markersize=3)
	axes[0, 1].plot(hours, [r["p3"] for r in records], label="p3", marker="s", markersize=3)
	axes[0, 1].axhline(env.p_min_safe, color="red", linestyle="--", linewidth=0.8, label="p_min_safe")
	axes[0, 1].axhline(env.p_max_safe, color="green", linestyle="--", linewidth=0.8, label="p_max_safe")
	axes[0, 1].set_title("Node pressures")
	axes[0, 1].set_xlabel("Hour")
	axes[0, 1].set_ylabel("Pressure")
	axes[0, 1].legend(fontsize=7)
	axes[0, 1].grid(True)

	axes[1, 0].plot(hours, [r["linepack"] for r in records], color="purple", marker="o", markersize=3)
	axes[1, 0].set_title("Pipeline linepack")
	axes[1, 0].set_xlabel("Hour")
	axes[1, 0].set_ylabel("Mass")
	axes[1, 0].grid(True)

	axes[1, 1].bar(hours, [r["purchase_cost"] for r in records], color="orange")
	axes[1, 1].set_title("Electricity cost per hour")
	axes[1, 1].set_xlabel("Hour")
	axes[1, 1].set_ylabel("Cost")
	axes[1, 1].grid(True, axis="y")
	return fig


def print_episode_table(result):
	print("\n24h dispatch simulation")
	print("hour | price | demand | alpha | p2    | p3    | power_kwh | cost")
	print("-" * 78)
	for info in result["records"]:
		print(
			f"{info['hour']:02d}   | {info['price']:>5.2f} | {info['demand']:>6.1f} | {info['alpha']:>5.3f} | "
			f"{info['p2']:>5.1f} | {info['p3']:>5.1f} | {info['power_kwh']:>9.2f} | {info['purchase_cost']:>7.2f}"
		)
	print("-" * 78)
	print(f"Total purchased electricity: {result['total_power']:.2f} kWh")
	print(f"Total electricity cost: {result['total_cost']:.2f}")
	print(f"Pressure/flow violation hours: {result['violation_hours']}")


def log_to_tensorboard(writer, result):
	records = result["records"]
	writer.add_scalar("dp/episode/total_cost", result["total_cost"], 0)
	writer.add_scalar("dp/episode/total_reward", result["total_reward"], 0)
	writer.add_scalar("dp/episode/total_power_kwh", result["total_power"], 0)
	writer.add_scalar("dp/episode/violation_hours", result["violation_hours"], 0)
	writer.add_scalar("dp/episode/mean_alpha", float(np.mean([r["alpha"] for r in records])), 0)
	writer.add_scalar("dp/episode/mean_p2_bar", float(np.mean([r["p2"] for r in records])), 0)
	writer.add_scalar("dp/episode/mean_p3_bar", float(np.mean([r["p3"] for r in records])), 0)
	writer.add_scalar("dp/episode/mean_linepack", float(np.mean([r["linepack"] for r in records])), 0)

	for i, r in enumerate(records):
		writer.add_scalar("dp/hourly/alpha", r["alpha"], i)
		writer.add_scalar("dp/hourly/p2_bar", r["p2"], i)
		writer.add_scalar("dp/hourly/p3_bar", r["p3"], i)
		writer.add_scalar("dp/hourly/linepack", r["linepack"], i)
		writer.add_scalar("dp/hourly/cost", r["purchase_cost"], i)


if __name__ == "__main__":
	config = EnvConfig(
		noise_scale=0.0,
		horizon=24,
		w_pressure=1000.0,
		w_flow=5.0,
		w_terminal_linepack=250.0,
	)
	env = PaperInspiredDynamicLinepackEnv(config=config)

	p_grid, a_grid, best_action = build_dp_policy_table(config)
	result = run_dp_episode(env, p_grid, a_grid, best_action)
	print_episode_table(result)

	writer = SummaryWriter(log_dir=DP_LOG_DIR)
	log_to_tensorboard(writer, result)
	fig = make_figure(env, result["records"])
	writer.add_figure("dp/eval/24h_curves", fig, 0)
	plt.close(fig)
	writer.close()
