import time
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from envs.pipeline_env import EnvConfig, PaperInspiredDynamicLinepackEnv


GA_POPULATION_SIZE = 100
GA_GENERATIONS = 120
GA_ELITE_SIZE = 4
GA_MUTATION_RATE = 0.18
GA_TOURNAMENT_SIZE = 4
GA_MUTATION_STD = 0.08
GA_LOG_DIR = "models/tensorboard/ga_papercode"


@dataclass
class EpisodeEvaluation:
	fitness: float
	total_cost: float
	total_reward: float
	total_power: float
	violation_hours: int
	records: list


class TrainingProgressBar:
	def __init__(self, total_generations, bar_width=30):
		self.total_generations = int(total_generations)
		self.bar_width = int(bar_width)
		self.start_time = None

	def start(self):
		self.start_time = time.time()
		print("\nGA progress:")
		self.render(0)

	def render(self, current_generation):
		progress = min(current_generation / max(self.total_generations, 1), 1.0)
		filled = int(self.bar_width * progress)
		bar = "#" * filled + "-" * (self.bar_width - filled)
		elapsed = 0.0 if self.start_time is None else time.time() - self.start_time
		print(f"\r[{bar}] {progress:6.2%} ({current_generation}/{self.total_generations}) | {elapsed:6.1f}s", end="", flush=True)

	def finish(self):
		self.render(self.total_generations)
		print()


def _simulate_action_schedule(env, action_schedule):
	obs, _ = env.reset()
	total_reward = 0.0
	total_cost = 0.0
	total_power = 0.0
	violation_hours = 0
	records = []

	for hour in range(env.horizon):
		action_value = float(np.clip(action_schedule[hour], env.action_space.low[0], env.action_space.high[0]))
		obs, reward, terminated, truncated, info = env.step(np.array([action_value], dtype=np.float32))
		total_reward += float(reward)
		total_cost += float(info["purchase_cost"])
		total_power += float(info["power_kwh"])
		if info["violation"]:
			violation_hours += 1
		records.append(info)
		if terminated or truncated:
			break

	fitness = total_reward
	return EpisodeEvaluation(
		fitness=fitness,
		total_cost=total_cost,
		total_reward=total_reward,
		total_power=total_power,
		violation_hours=violation_hours,
		records=records,
	)


def _initialize_population(population_size, horizon, action_low, action_high, rng):
	return rng.uniform(action_low, action_high, size=(population_size, horizon)).astype(np.float64)


def _tournament_select(population, fitnesses, tournament_size, rng):
	candidate_indices = rng.integers(0, len(population), size=tournament_size)
	best_index = candidate_indices[np.argmax(fitnesses[candidate_indices])]
	return population[best_index].copy()


def _uniform_crossover(parent_a, parent_b, rng):
	mask = rng.random(parent_a.shape) < 0.5
	child = np.where(mask, parent_a, parent_b)
	return child


def _mutate(individual, action_low, action_high, mutation_rate, mutation_std, rng):
	mutation_mask = rng.random(individual.shape) < mutation_rate
	noise = rng.normal(0.0, mutation_std, size=individual.shape)
	individual = individual + mutation_mask * noise
	return np.clip(individual, action_low, action_high)


def _render_curves_figure(env, records, generation_index):
	hours = [record["hour"] for record in records]
	fig, axes = plt.subplots(2, 2, figsize=(12, 8), tight_layout=True)
	fig.suptitle(f"GA deterministic 24h eval - generation {generation_index}", fontsize=11)

	axes[0, 0].plot(hours, [record["alpha"] for record in records], marker="o", markersize=3)
	axes[0, 0].set_title("Compressor action alpha")
	axes[0, 0].set_xlabel("Hour")
	axes[0, 0].set_ylabel("alpha")
	axes[0, 0].grid(True)

	axes[0, 1].plot(hours, [record["p2"] for record in records], label="p2", marker="o", markersize=3)
	axes[0, 1].plot(hours, [record["p3"] for record in records], label="p3", marker="s", markersize=3)
	axes[0, 1].axhline(env.p_min_safe, color="red", linestyle="--", linewidth=0.8, label="p_min_safe")
	axes[0, 1].axhline(env.p_max_safe, color="green", linestyle="--", linewidth=0.8, label="p_max_safe")
	axes[0, 1].set_title("Node pressures")
	axes[0, 1].set_xlabel("Hour")
	axes[0, 1].set_ylabel("Pressure")
	axes[0, 1].legend(fontsize=7)
	axes[0, 1].grid(True)

	axes[1, 0].plot(hours, [record["linepack"] for record in records], color="purple", marker="o", markersize=3)
	axes[1, 0].set_title("Pipeline linepack")
	axes[1, 0].set_xlabel("Hour")
	axes[1, 0].set_ylabel("Mass")
	axes[1, 0].grid(True)

	axes[1, 1].bar(hours, [record["purchase_cost"] for record in records], color="orange")
	axes[1, 1].set_title("Electricity cost per hour")
	axes[1, 1].set_xlabel("Hour")
	axes[1, 1].set_ylabel("Cost")
	axes[1, 1].grid(True, axis="y")

	return fig


def _log_generation(writer, env, generation_index, evaluation):
	writer.add_scalar("ga/generation/best_fitness", evaluation.fitness, generation_index)
	writer.add_scalar("ga/generation/total_cost", evaluation.total_cost, generation_index)
	writer.add_scalar("ga/generation/total_reward", evaluation.total_reward, generation_index)
	writer.add_scalar("ga/generation/total_power_kwh", evaluation.total_power, generation_index)
	writer.add_scalar("ga/generation/violation_hours", evaluation.violation_hours, generation_index)
	writer.add_scalar("ga/generation/mean_alpha", float(np.mean([record["alpha"] for record in evaluation.records])), generation_index)
	writer.add_scalar("ga/generation/mean_p2_bar", float(np.mean([record["p2"] for record in evaluation.records])), generation_index)
	writer.add_scalar("ga/generation/mean_p3_bar", float(np.mean([record["p3"] for record in evaluation.records])), generation_index)
	writer.add_scalar("ga/generation/mean_linepack", float(np.mean([record["linepack"] for record in evaluation.records])), generation_index)
	figure = _render_curves_figure(env, evaluation.records, generation_index)
	writer.add_figure("ga/eval/24h_curves", figure, generation_index)
	plt.close(figure)


def run_ga_optimization(env, population_size, generations, elite_size, mutation_rate, tournament_size, mutation_std, writer):
	rng = np.random.default_rng(42)
	action_low = float(env.action_space.low[0])
	action_high = float(env.action_space.high[0])
	population = _initialize_population(population_size, env.horizon, action_low, action_high, rng)
	best_individual = None
	best_evaluation = None
	progress = TrainingProgressBar(generations)
	progress.start()

	for generation_index in range(generations):
		evaluations = [_simulate_action_schedule(env, individual) for individual in population]
		fitnesses = np.array([evaluation.fitness for evaluation in evaluations], dtype=np.float64)
		sorted_indices = np.argsort(fitnesses)[::-1]

		current_best = evaluations[int(sorted_indices[0])]
		if best_evaluation is None or current_best.fitness > best_evaluation.fitness:
			best_evaluation = current_best
			best_individual = population[int(sorted_indices[0])].copy()

		_log_generation(writer, env, generation_index, current_best)

		progress.render(generation_index + 1)

		next_population = [population[index].copy() for index in sorted_indices[:elite_size]]
		while len(next_population) < population_size:
			parent_a = _tournament_select(population, fitnesses, tournament_size, rng)
			parent_b = _tournament_select(population, fitnesses, tournament_size, rng)
			child = _uniform_crossover(parent_a, parent_b, rng)
			child = _mutate(child, action_low, action_high, mutation_rate, mutation_std, rng)
			next_population.append(child)

		population = np.asarray(next_population, dtype=np.float64)

	progress.finish()
	return best_individual, best_evaluation


def run_deterministic_day(env, model_action_schedule):
	obs, _ = env.reset()
	total_cost = 0.0
	total_power = 0.0
	violation_hours = 0

	print("\n24h dispatch simulation")
	print("hour | price | demand | alpha | p2    | p3    | power_kwh | cost")
	print("-" * 78)

	for hour in range(env.horizon):
		action_value = float(np.clip(model_action_schedule[hour], env.action_space.low[0], env.action_space.high[0]))
		obs, reward, terminated, truncated, info = env.step(np.array([action_value], dtype=np.float32))

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
	writer = SummaryWriter(log_dir=GA_LOG_DIR)

	best_actions, best_evaluation = run_ga_optimization(
		env=env,
		population_size=GA_POPULATION_SIZE,
		generations=GA_GENERATIONS,
		elite_size=GA_ELITE_SIZE,
		mutation_rate=GA_MUTATION_RATE,
		tournament_size=GA_TOURNAMENT_SIZE,
		mutation_std=GA_MUTATION_STD,
		writer=writer,
	)

	if best_evaluation is not None:
		writer.add_scalar("ga/final/best_fitness", best_evaluation.fitness, GA_GENERATIONS)
		writer.add_scalar("ga/final/total_cost", best_evaluation.total_cost, GA_GENERATIONS)
		writer.add_scalar("ga/final/total_reward", best_evaluation.total_reward, GA_GENERATIONS)
		writer.add_scalar("ga/final/total_power_kwh", best_evaluation.total_power, GA_GENERATIONS)
		writer.add_scalar("ga/final/violation_hours", best_evaluation.violation_hours, GA_GENERATIONS)
		figure = _render_curves_figure(env, best_evaluation.records, GA_GENERATIONS)
		writer.add_figure("ga/final/24h_curves", figure, GA_GENERATIONS)
		plt.close(figure)

	run_deterministic_day(env, best_actions)
	writer.close()
