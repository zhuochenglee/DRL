import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise

from envs.pipeline_env import EnvConfig, PaperInspiredDynamicLinepackEnv


class TrainingProgressBarCallback(BaseCallback):
	def __init__(self, total_timesteps, bar_width=30):
		super().__init__()
		self.total_timesteps = int(total_timesteps)
		self.bar_width = int(bar_width)
		self.update_interval = max(1, self.total_timesteps // 100)
		self.last_displayed_step = -1
		self.start_time = None

	def _on_training_start(self):
		self.start_time = time.time()
		print("\nTraining progress:")
		self._render_progress(0)

	def _on_step(self):
		current_step = min(self.num_timesteps, self.total_timesteps)
		if current_step >= self.total_timesteps or current_step - self.last_displayed_step >= self.update_interval:
			self.last_displayed_step = current_step
			self._render_progress(current_step)
		return True

	def _on_training_end(self):
		self._render_progress(self.total_timesteps)
		print()

	def _render_progress(self, current_step):
		progress = min(current_step / max(self.total_timesteps, 1), 1.0)
		filled = int(self.bar_width * progress)
		bar = "#" * filled + "-" * (self.bar_width - filled)
		elapsed = 0.0 if self.start_time is None else time.time() - self.start_time
		print(f"\r[{bar}] {progress:6.2%} ({current_step}/{self.total_timesteps}) | {elapsed:6.1f}s", end="", flush=True)


class EpisodeStatsCallback(BaseCallback):
	"""Logs per-episode aggregate metrics to TensorBoard during training."""

	def __init__(self, verbose=0):
		super().__init__(verbose)
		self._reset_buffers()

	def _reset_buffers(self):
		self._alpha = []
		self._p2 = []
		self._p3 = []
		self._linepack = []
		self._cost = []
		self._violations = 0

	def _on_step(self) -> bool:
		infos = self.locals.get("infos", [{}])
		dones = self.locals.get("dones", [False])
		for info, done in zip(infos, dones):
			if not info:
				continue
			self._alpha.append(info.get("alpha", 0.0))
			self._p2.append(info.get("p2", 0.0))
			self._p3.append(info.get("p3", 0.0))
			self._linepack.append(info.get("linepack", 0.0))
			self._cost.append(info.get("purchase_cost", 0.0))
			if info.get("violation", False):
				self._violations += 1
			if done and self._cost:
				self.logger.record("td3/episode/total_cost", float(sum(self._cost)))
				self.logger.record("td3/episode/violation_hours", float(self._violations))
				self.logger.record("td3/episode/mean_alpha", float(np.mean(self._alpha)))
				self.logger.record("td3/episode/mean_p2_bar", float(np.mean(self._p2)))
				self.logger.record("td3/episode/mean_p3_bar", float(np.mean(self._p3)))
				self.logger.record("td3/episode/mean_linepack", float(np.mean(self._linepack)))
				self.logger.dump(self.num_timesteps)
				self._reset_buffers()
		return True


class CurveLoggingCallback(BaseCallback):
	"""Periodically runs a deterministic 24h episode and logs curves as TensorBoard figures."""

	def __init__(self, eval_env, eval_freq=20000, verbose=0):
		super().__init__(verbose)
		self.eval_env = eval_env
		self.eval_freq = eval_freq
		self._eval_count = 0
		self._last_eval_step = 0

	def _on_step(self) -> bool:
		if self.num_timesteps - self._last_eval_step >= self.eval_freq:
			self._last_eval_step = self.num_timesteps
			self._run_and_log()
		return True

	def _run_and_log(self):
		env = self.eval_env
		obs, _ = env.reset()
		records = []
		for _ in range(env.horizon):
			action, _ = self.model.predict(obs, deterministic=True)
			obs, _, terminated, truncated, info = env.step(action)
			records.append(info)
			if terminated or truncated:
				break

		hours = [r["hour"] for r in records]

		fig, axes = plt.subplots(2, 2, figsize=(12, 8), tight_layout=True)
		fig.suptitle(f"Deterministic 24h Eval - step {self.num_timesteps}", fontsize=11)

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

		writer = None
		for fmt in self.logger.output_formats:
			if hasattr(fmt, "writer"):
				writer = fmt.writer
				break
		if writer is not None:
			writer.add_figure("td3/eval/24h_curves", fig, self._eval_count)

		plt.close(fig)
		self._eval_count += 1


def run_deterministic_day(env, model):
	obs, _ = env.reset()
	total_cost = 0.0
	total_power = 0.0
	violation_hours = 0

	print("\n24h dispatch simulation")
	print("hour | price | demand | alpha | p2    | p3    | power_kwh | cost")
	print("-" * 78)

	for _ in range(env.horizon):
		action, _ = model.predict(obs, deterministic=True)
		obs, reward, terminated, truncated, info = env.step(action)

		total_cost += info["purchase_cost"]
		total_power += info["power_kwh"]
		if info["violation"]:
			violation_hours += 1

		print(
			f"{info['hour']:02d}   | {info['price']:>5.2f} | {info['demand']:>6.1f} | {info['alpha']:>5.3f} | "
			f"{info['p2']:>5.1f} | {info['p3']:>5.1f} | {info['power_kwh']:>9.2f} | {info['purchase_cost']:>7.2f}"
		)
		if terminated or truncated:
			break

	print("-" * 78)
	print(f"Total purchased electricity: {total_power:.2f} kWh")
	print(f"Total electricity cost: {total_cost:.2f}")
	print(f"Pressure/flow violation hours: {violation_hours}")


if __name__ == "__main__":
	config = EnvConfig(
		noise_scale=0.0,
		horizon=24,
		w_pressure=1000.0,
		w_flow=5.0,
		w_terminal_linepack=250.0,
	)
	env = PaperInspiredDynamicLinepackEnv(config=config)
	eval_env = PaperInspiredDynamicLinepackEnv(config=config)
	check_env(env, warn=True)

	total_timesteps = 120000
	action_noise = NormalActionNoise(mean=np.zeros(1), sigma=0.10 * np.ones(1))
	model = TD3(
		"MlpPolicy",
		env,
		verbose=0,
		learning_rate=1e-3,
		gamma=0.995,
		action_noise=action_noise,
		tensorboard_log="models/tensorboard",
	)
	callback = CallbackList([
		TrainingProgressBarCallback(total_timesteps=total_timesteps),
		EpisodeStatsCallback(),
		CurveLoggingCallback(eval_env=eval_env, eval_freq=20000),
	])
	model.learn(total_timesteps=total_timesteps, callback=callback, tb_log_name="td3")

	run_deterministic_day(env, model)
