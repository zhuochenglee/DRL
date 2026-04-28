import csv
import time
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from textwrap import dedent

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor

SummaryWriter = import_module("tensorboardX").SummaryWriter


def detect_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = detect_device()
TOTAL_DAYS = 365
EXTRACTION_DAYS = 120
INJECTION_DAYS = 200
BALANCE_DAYS = 45
WORKING_GAS_VOLUME_10K = 108000.0
ELECTRICITY_PRICE = 0.42
GAS_SALE_PRICE = 1.22
MIN_RENEWABLE_SHARE = 0.20
TRAIN_TIMESTEPS = 200000
RUN_TRAINING = True
TRAIN_GAMMA = 0.999
TRAIN_LEARNING_RATE = 1e-4
EVAL_FREQ = 5000
EVAL_EPISODES = 3
MODEL_DIR = Path("models")
EXPORT_DIR = MODEL_DIR / "exports"
TENSORBOARD_DIR = MODEL_DIR / "tensorboard"
GA_POP_SIZE = 24
GA_GENERATIONS = 20
GA_ELITE = 4
GA_MUTATION_RATE = 0.18
GA_TOURNAMENT_SIZE = 4
MPC_HORIZON = 5

GA_PARAM_BOUNDS = np.array([
    [0.60, 1.20],
    [0.60, 1.20],
    [-0.15, 0.05],
    [-0.15, 0.05],
    [-0.20, 0.05],
    [0.68, 0.82],
    [0.65, 0.88],
    [-0.08, 0.08],
], dtype=np.float64)
GA_HEURISTIC_PARAMS = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.72, 0.78, 0.0], dtype=np.float64)
MPC_CANDIDATE_LIBRARY = (
    (0.00, 0.00, 0.72, 0.78, 0.00),
    (-0.10, 0.08, 0.78, 0.74, 0.03),
    (0.10, -0.02, 0.70, 0.82, 0.00),
    (-0.04, 0.15, 0.74, 0.78, 0.00),
    (-0.02, 0.04, 0.76, 0.72, 0.06),
    (0.02, 0.02, 0.72, 0.84, -0.02),
    (-0.12, -0.04, 0.80, 0.70, 0.05),
)

PHASE_EXTRACTION = -1
PHASE_BALANCE = 0
PHASE_INJECTION = 1

COMPRESSOR_NAMES = ("compressor_4500", "compressor_4000")
COMPRESSOR_RATINGS_KW = np.array([4500.0, 4000.0], dtype=np.float64)
COMPRESSOR_MIN_STABLE_LOAD = np.array([0.35, 0.38], dtype=np.float64)
COMPRESSOR_START_THRESHOLD = np.array([0.18, 0.20], dtype=np.float64)
COMPRESSOR_STARTUP_COST_YUAN = np.array([3200.0, 2800.0], dtype=np.float64)
COMPRESSOR_SHUTDOWN_COST_YUAN = np.array([900.0, 800.0], dtype=np.float64)
COMPRESSOR_STARTUP_ENERGY_KWH = np.array([1800.0, 1500.0], dtype=np.float64)
COMPRESSOR_SWITCH_PENALTY = np.array([18.0, 16.0], dtype=np.float64)
WELL_NAMES = ("well_1", "well_2", "well_3", "well_4")
EXTRACTION_WELL_WEIGHTS = np.array([0.30, 0.27, 0.23, 0.20], dtype=np.float64)
INJECTION_WELL_WEIGHTS = np.array([0.22, 0.24, 0.26, 0.28], dtype=np.float64)
EXTRACTION_WELL_CAPACITY_10K = np.array([320.0, 290.0, 250.0, 220.0], dtype=np.float64)
INJECTION_WELL_CAPACITY_10K = np.array([140.0, 150.0, 160.0, 170.0], dtype=np.float64)
WELL_PRESSURE_LOSS_MPA = np.array([0.35, 0.20, 0.25, 0.45], dtype=np.float64)
OTHER_STATION_POWER_KW = 600.0


def build_extraction_baseline():
    day_idx = np.arange(1, EXTRACTION_DAYS + 1, dtype=np.float64)
    pressure = np.interp(day_idx, [1.0, 52.0, 120.0], [25.0, 6.4, 6.4])
    rate = np.interp(day_idx, [1.0, 42.0, 120.0], [795.3, 1061.1, 336.4])
    return pressure, rate


def build_injection_baseline():
    day_idx = np.arange(1, INJECTION_DAYS + 1, dtype=np.float64)
    pressure = np.interp(day_idx, [1.0, 131.0, 200.0], [10.1, 29.0, 29.0])
    rate = np.interp(day_idx, [1.0, 131.0, 200.0], [600.0, 600.0, 270.1])
    return pressure, rate


def build_calendar_month_index():
    month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_index = []
    for month, length in enumerate(month_lengths):
        month_index.extend([month] * length)
    return np.array(month_index, dtype=np.int32)


def build_renewable_availability():
    month_index = build_calendar_month_index()
    wind_factor = np.array([0.82, 0.88, 1.18, 1.22, 1.20, 1.10, 0.92, 0.88, 0.85, 0.80, 0.78, 0.80])
    solar_factor = np.array([0.48, 0.58, 0.72, 0.86, 1.02, 1.12, 1.15, 1.08, 0.92, 0.75, 0.55, 0.45])

    wind_norm = wind_factor / wind_factor.max()
    solar_norm = solar_factor / solar_factor.max()
    combined = 0.55 * wind_norm[month_index] + 0.45 * solar_norm[month_index]

    day_of_year = np.arange(TOTAL_DAYS, dtype=np.float64)
    annual_shape = 0.08 * np.sin(2.0 * np.pi * day_of_year / TOTAL_DAYS - 0.7)
    availability = np.clip(combined + annual_shape, 0.25, 1.0)
    return availability.astype(np.float32)


def build_annual_profiles():
    extraction_pressure, extraction_rate = build_extraction_baseline()
    injection_pressure, injection_rate = build_injection_baseline()
    renewable_availability = build_renewable_availability()

    phase = np.full(TOTAL_DAYS, PHASE_BALANCE, dtype=np.int32)
    phase_day = np.zeros(TOTAL_DAYS, dtype=np.int32)
    baseline_pressure = np.zeros(TOTAL_DAYS, dtype=np.float64)
    baseline_rate = np.zeros(TOTAL_DAYS, dtype=np.float64)

    extraction_slices = [(0, 74), (319, 365)]
    extraction_cursor = 0
    for start, end in extraction_slices:
        length = end - start
        phase[start:end] = PHASE_EXTRACTION
        phase_day[start:end] = np.arange(extraction_cursor + 1, extraction_cursor + length + 1)
        baseline_pressure[start:end] = extraction_pressure[extraction_cursor:extraction_cursor + length]
        baseline_rate[start:end] = extraction_rate[extraction_cursor:extraction_cursor + length]
        extraction_cursor += length

    injection_start = 104
    injection_end = 304
    phase[injection_start:injection_end] = PHASE_INJECTION
    phase_day[injection_start:injection_end] = np.arange(1, INJECTION_DAYS + 1)
    baseline_pressure[injection_start:injection_end] = injection_pressure
    baseline_rate[injection_start:injection_end] = injection_rate

    balance_slices = [(74, 104), (304, 319)]
    balance_pressure = [8.5, 18.0]
    for (start, end), pressure in zip(balance_slices, balance_pressure):
        phase[start:end] = PHASE_BALANCE
        phase_day[start:end] = np.arange(1, end - start + 1)
        baseline_pressure[start:end] = pressure
        baseline_rate[start:end] = 0.0

    signed_rate = np.where(phase == PHASE_INJECTION, baseline_rate, 0.0)
    signed_rate = np.where(phase == PHASE_EXTRACTION, -baseline_rate, signed_rate)

    target_inventory = np.zeros(TOTAL_DAYS, dtype=np.float64)
    inventory = 1.0
    for day in range(TOTAL_DAYS):
        inventory += signed_rate[day] / WORKING_GAS_VOLUME_10K
        inventory = float(np.clip(inventory, 0.0, 1.0))
        target_inventory[day] = inventory

    return {
        "phase": phase,
        "phase_day": phase_day,
        "baseline_pressure": baseline_pressure,
        "baseline_rate": baseline_rate,
        "renewable_availability": renewable_availability,
        "target_inventory": target_inventory,
    }


ANNUAL_PROFILES = build_annual_profiles()
MAX_BASELINE_RATE = float(np.max(ANNUAL_PROFILES["baseline_rate"]))
MAX_POSSIBLE_RATE = MAX_BASELINE_RATE * 1.15
MAX_AUXILIARY_POWER_KWH = float(24.0 * (OTHER_STATION_POWER_KW + 0.32 * MAX_POSSIBLE_RATE))
MAX_STARTUP_ENERGY_KWH = float(np.sum(COMPRESSOR_STARTUP_ENERGY_KWH))
MAX_REQUIRED_WORK_KWH = float(max(
    MAX_POSSIBLE_RATE * (54.0 + 1.6 * (25.0 + float(np.max(WELL_PRESSURE_LOSS_MPA)))),
    MAX_POSSIBLE_RATE * (72.0 + 1.9 * (29.0 + float(np.max(WELL_PRESSURE_LOSS_MPA)))),
))
MAX_STATION_INPUT_POWER_KWH = float(24.0 * np.sum(COMPRESSOR_RATINGS_KW) + MAX_AUXILIARY_POWER_KWH + MAX_STARTUP_ENERGY_KWH)
MAX_OBSERVATION_POWER_KWH = float(max(MAX_STATION_INPUT_POWER_KWH, MAX_REQUIRED_WORK_KWH))


class CompetitionSchedulingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)

        self.phase = ANNUAL_PROFILES["phase"]
        self.phase_day = ANNUAL_PROFILES["phase_day"]
        self.baseline_pressure = ANNUAL_PROFILES["baseline_pressure"]
        self.baseline_rate = ANNUAL_PROFILES["baseline_rate"]
        self.renewable_availability = ANNUAL_PROFILES["renewable_availability"]
        self.target_inventory = ANNUAL_PROFILES["target_inventory"]

        self.current_day = 0
        self.inventory = 1.0
        self.prev_rate = 0.0
        self.prev_power_kwh = 0.0
        self.prev_renewable_share = 1.0
        self.prev_compressor_loads = np.zeros(2, dtype=np.float64)
        self.total_reward = 0.0
        self.total_station_power_kwh = 0.0
        self.total_grid_power_kwh = 0.0
        self.total_renewable_power_kwh = 0.0
        self.total_grid_cost_yuan = 0.0
        self.total_startup_shutdown_cost_yuan = 0.0
        self.total_penalty = 0.0
        self.total_compressor_input_kwh = np.zeros(2, dtype=np.float64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_day = 0
        self.inventory = 1.0
        self.prev_rate = 0.0
        self.prev_power_kwh = 0.0
        self.prev_renewable_share = 1.0
        self.prev_compressor_loads = np.zeros(2, dtype=np.float64)
        self.total_reward = 0.0
        self.total_station_power_kwh = 0.0
        self.total_grid_power_kwh = 0.0
        self.total_renewable_power_kwh = 0.0
        self.total_grid_cost_yuan = 0.0
        self.total_startup_shutdown_cost_yuan = 0.0
        self.total_penalty = 0.0
        self.total_compressor_input_kwh = np.zeros(2, dtype=np.float64)
        if options and "initial_inventory" in options:
            self.inventory = float(np.clip(options["initial_inventory"], 0.0, 1.0))
        return self._get_obs(), {}

    def _get_phase_bounds(self, phase):
        if phase == PHASE_EXTRACTION:
            return 6.4, 25.0
        if phase == PHASE_INJECTION:
            return 10.1, 29.0
        return 6.4, 29.0

    def _inventory_to_pressure(self, phase):
        if phase == PHASE_EXTRACTION:
            return 6.4 + 18.6 * self.inventory
        if phase == PHASE_INJECTION:
            return 10.1 + 18.9 * self.inventory
        return 8.0 + 18.0 * self.inventory

    def _compressor_efficiency(self, load):
        load = np.clip(np.asarray(load, dtype=np.float64), 0.0, 1.0)
        efficiency = np.where(
            load > 1e-6,
            np.clip(0.72 + 0.18 * load - 0.16 * np.square(load - 0.78), 0.0, 0.90),
            0.0,
        )
        return efficiency

    def _apply_compressor_operating_constraints(self, requested_loads):
        requested_loads = np.clip(np.asarray(requested_loads, dtype=np.float64), 0.0, 1.0)
        prev_on = self.prev_compressor_loads >= COMPRESSOR_MIN_STABLE_LOAD
        current_on = requested_loads >= COMPRESSOR_START_THRESHOLD

        actual_loads = np.where(current_on, np.maximum(requested_loads, COMPRESSOR_MIN_STABLE_LOAD), 0.0)
        startup_events = current_on & ~prev_on
        shutdown_events = prev_on & ~current_on
        load_delta = np.abs(actual_loads - self.prev_compressor_loads)
        load_switch_penalty = float(np.dot(COMPRESSOR_SWITCH_PENALTY, load_delta))
        startup_cost_yuan = float(np.dot(COMPRESSOR_STARTUP_COST_YUAN, startup_events.astype(np.float64)))
        shutdown_cost_yuan = float(np.dot(COMPRESSOR_SHUTDOWN_COST_YUAN, shutdown_events.astype(np.float64)))
        startup_energy_kwh = float(np.dot(COMPRESSOR_STARTUP_ENERGY_KWH, startup_events.astype(np.float64)))

        return {
            "actual_loads": actual_loads,
            "startup_events": startup_events,
            "shutdown_events": shutdown_events,
            "load_switch_penalty": load_switch_penalty,
            "startup_cost_yuan": startup_cost_yuan,
            "shutdown_cost_yuan": shutdown_cost_yuan,
            "startup_energy_kwh": startup_energy_kwh,
        }

    def _allocate_with_caps(self, total_rate, weights, capacities):
        if total_rate <= 1e-9:
            return np.zeros_like(weights)

        normalized_weights = np.array(weights, dtype=np.float64)
        normalized_weights /= normalized_weights.sum()
        rates = np.minimum(total_rate * normalized_weights, capacities)
        remaining = float(total_rate - np.sum(rates))

        while remaining > 1e-6:
            headroom = capacities - rates
            eligible = headroom > 1e-6
            if not np.any(eligible):
                break
            extra_weights = normalized_weights[eligible]
            extra_weights /= extra_weights.sum()
            increment = np.minimum(remaining * extra_weights, headroom[eligible])
            rates[eligible] += increment
            consumed = float(np.sum(increment))
            if consumed <= 1e-9:
                break
            remaining -= consumed

        return rates

    def _allocate_wells(self, total_rate, phase):
        if phase == PHASE_BALANCE or total_rate <= 1e-9:
            return np.zeros(len(WELL_NAMES), dtype=np.float64), 0.0, 0.0

        seasonal_phase = 2.0 * np.pi * self.current_day / TOTAL_DAYS
        well_shape = 1.0 + 0.06 * np.sin(seasonal_phase + np.array([0.0, 0.8, 1.6, 2.4]))

        if phase == PHASE_EXTRACTION:
            weights = EXTRACTION_WELL_WEIGHTS * well_shape
            capacities = EXTRACTION_WELL_CAPACITY_10K
        else:
            weights = INJECTION_WELL_WEIGHTS * well_shape[::-1]
            capacities = INJECTION_WELL_CAPACITY_10K

        well_rates = self._allocate_with_caps(total_rate, weights, capacities)
        achieved_rate = float(np.sum(well_rates))
        rate_shortfall_ratio = 0.0 if total_rate <= 1e-9 else max(total_rate - achieved_rate, 0.0) / total_rate
        if achieved_rate <= 1e-9:
            average_line_loss = 0.0
        else:
            average_line_loss = float(np.dot(well_rates / achieved_rate, WELL_PRESSURE_LOSS_MPA))

        return well_rates, average_line_loss, rate_shortfall_ratio

    def _estimate_required_work(self, phase, rate_10k, pressure_mpa, line_loss_mpa):
        effective_pressure = max(pressure_mpa + line_loss_mpa, 0.0)
        if phase == PHASE_EXTRACTION:
            return rate_10k * (54.0 + 1.6 * effective_pressure)
        if phase == PHASE_INJECTION:
            return rate_10k * (72.0 + 1.9 * effective_pressure)
        return 0.0

    def _auxiliary_power_kwh(self, phase, rate_10k):
        if phase == PHASE_BALANCE:
            return 24.0 * OTHER_STATION_POWER_KW * 0.55
        return 24.0 * (OTHER_STATION_POWER_KW + 0.32 * rate_10k)

    def _get_obs(self):
        phase = self.phase[self.current_day]
        phase_day = self.phase_day[self.current_day]
        baseline_rate = self.baseline_rate[self.current_day]
        baseline_pressure = self.baseline_pressure[self.current_day]
        renewable = float(self.renewable_availability[self.current_day])
        target_inventory = self.target_inventory[self.current_day]
        annual_t = 2.0 * np.pi * self.current_day / TOTAL_DAYS
        desired_rate = baseline_rate
        _, line_loss, _ = self._allocate_wells(desired_rate, phase)
        required_work = self._estimate_required_work(phase, desired_rate, baseline_pressure, line_loss)

        obs = np.array([
            float(phase),
            2.0 * phase_day / max(EXTRACTION_DAYS, INJECTION_DAYS) - 1.0,
            2.0 * self.current_day / (TOTAL_DAYS - 1) - 1.0,
            2.0 * self.inventory - 1.0,
            2.0 * target_inventory - 1.0,
            baseline_pressure / 32.0 * 2.0 - 1.0,
            baseline_rate / MAX_POSSIBLE_RATE * 2.0 - 1.0,
            self.prev_rate / MAX_POSSIBLE_RATE * 2.0 - 1.0,
            renewable * 2.0 - 1.0,
            self.prev_renewable_share * 2.0 - 1.0,
            self.prev_compressor_loads[0] * 2.0 - 1.0,
            self.prev_compressor_loads[1] * 2.0 - 1.0,
            self.prev_power_kwh / MAX_OBSERVATION_POWER_KWH * 2.0 - 1.0,
            required_work / MAX_OBSERVATION_POWER_KWH * 2.0 - 1.0,
            np.sin(annual_t),
            np.cos(annual_t),
        ], dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)

    def step(self, action):
        dispatch_adjustment = float(np.clip(action[0], -1.0, 1.0))
        requested_compressor_loads = np.clip(np.asarray(action[1:3], dtype=np.float64), 0.0, 1.0)
        renewable_request = float(np.clip(action[3], 0.0, 1.0))

        phase = self.phase[self.current_day]
        baseline_rate = self.baseline_rate[self.current_day]
        baseline_pressure = self.baseline_pressure[self.current_day]
        renewable_availability = float(self.renewable_availability[self.current_day])
        target_inventory = self.target_inventory[self.current_day]

        if phase == PHASE_BALANCE:
            requested_rate = 0.0
            pressure_adjustment = 0.0
        else:
            requested_rate = baseline_rate * np.clip(1.0 + 0.15 * dispatch_adjustment, 0.85, 1.15)
            pressure_adjustment = 1.8 * dispatch_adjustment if phase == PHASE_INJECTION else -1.8 * dispatch_adjustment

        well_requested_rates, average_line_loss, well_shortfall_ratio = self._allocate_wells(requested_rate, phase)
        well_limited_rate = float(np.sum(well_requested_rates))

        compressor_state = self._apply_compressor_operating_constraints(requested_compressor_loads)
        compressor_loads = compressor_state["actual_loads"]
        compressor_efficiency = self._compressor_efficiency(compressor_loads)
        compressor_input_kwh = 24.0 * COMPRESSOR_RATINGS_KW * compressor_loads
        delivered_work_kwh = float(np.sum(compressor_input_kwh * compressor_efficiency))
        required_work_kwh = self._estimate_required_work(phase, well_limited_rate, baseline_pressure, average_line_loss)
        compressor_feasibility = 1.0
        if required_work_kwh > 1e-9:
            compressor_feasibility = float(min(delivered_work_kwh / required_work_kwh, 1.0))

        actual_rate = well_limited_rate * compressor_feasibility
        well_rates = well_requested_rates * compressor_feasibility
        compressor_shortfall_ratio = 1.0 - compressor_feasibility

        raw_inventory = self.inventory
        if phase == PHASE_INJECTION:
            raw_inventory += actual_rate / WORKING_GAS_VOLUME_10K
        elif phase == PHASE_EXTRACTION:
            raw_inventory -= actual_rate / WORKING_GAS_VOLUME_10K

        inventory_violation = abs(np.clip(raw_inventory, 0.0, 1.0) - raw_inventory)
        self.inventory = float(np.clip(raw_inventory, 0.0, 1.0))

        inventory_pressure = self._inventory_to_pressure(phase)
        station_pressure = 0.70 * baseline_pressure + 0.30 * inventory_pressure + pressure_adjustment
        actual_pressure = station_pressure - average_line_loss - 2.2 * compressor_shortfall_ratio

        min_pressure, max_pressure = self._get_phase_bounds(phase)
        pressure_gap = max(min_pressure - actual_pressure, 0.0) + max(actual_pressure - max_pressure, 0.0)
        inventory_gap = abs(self.inventory - target_inventory)
        if phase == PHASE_BALANCE or baseline_rate <= 1e-9:
            schedule_gap = 0.0
        else:
            schedule_gap = abs(actual_rate - baseline_rate) / baseline_rate

        auxiliary_power_kwh = self._auxiliary_power_kwh(phase, actual_rate)
        power_kwh = float(np.sum(compressor_input_kwh) + auxiliary_power_kwh + compressor_state["startup_energy_kwh"])

        renewable_power_kwh = power_kwh * renewable_request * renewable_availability
        renewable_power_kwh = float(np.clip(renewable_power_kwh, 0.0, power_kwh))
        grid_power_kwh = float(max(power_kwh - renewable_power_kwh, 0.0))
        renewable_share = 1.0 if power_kwh <= 1e-9 else renewable_power_kwh / power_kwh

        grid_cost_yuan = grid_power_kwh * ELECTRICITY_PRICE
        startup_shutdown_cost_yuan = compressor_state["startup_cost_yuan"] + compressor_state["shutdown_cost_yuan"]
        operating_cost_yuan = grid_cost_yuan + startup_shutdown_cost_yuan

        renewable_penalty = 0.0
        if phase == PHASE_EXTRACTION and renewable_share < MIN_RENEWABLE_SHARE:
            renewable_penalty = 80.0 * (MIN_RENEWABLE_SHARE - renewable_share)

        penalty = (
            6.0 * pressure_gap
            + 50.0 * inventory_gap
            + 70.0 * schedule_gap
            + 90.0 * well_shortfall_ratio
            + 120.0 * compressor_shortfall_ratio
            + 650.0 * inventory_violation
            + compressor_state["load_switch_penalty"]
            + renewable_penalty
        )

        energy_term = power_kwh / 12000.0
        grid_term = grid_power_kwh / 10000.0 + startup_shutdown_cost_yuan / 4000.0
        green_bonus = 7.0 * renewable_share
        reward = green_bonus - energy_term - grid_term - penalty

        self.prev_rate = actual_rate
        self.prev_power_kwh = power_kwh
        self.prev_renewable_share = renewable_share
        self.prev_compressor_loads = compressor_loads.copy()

        terminated = self.current_day == TOTAL_DAYS - 1
        self.total_station_power_kwh += power_kwh
        self.total_grid_power_kwh += grid_power_kwh
        self.total_renewable_power_kwh += renewable_power_kwh
        self.total_grid_cost_yuan += grid_cost_yuan
        self.total_startup_shutdown_cost_yuan += startup_shutdown_cost_yuan
        self.total_compressor_input_kwh += compressor_input_kwh

        terminal_penalty = 0.0
        if terminated:
            annual_share = self.annual_renewable_share()
            if annual_share < MIN_RENEWABLE_SHARE:
                annual_gap = MIN_RENEWABLE_SHARE - annual_share
                terminal_penalty = 120.0 * annual_gap
                reward -= terminal_penalty
                penalty += terminal_penalty
        else:
            annual_share = None

        self.total_reward += reward
        self.total_penalty += penalty

        info = {
            "phase": self.phase_name(self.current_day),
            "day": self.current_day + 1,
            "day_in_phase": int(self.phase_day[self.current_day]),
            "baseline_rate_10k": float(baseline_rate),
            "requested_rate_10k": float(requested_rate),
            "actual_rate_10k": float(actual_rate),
            "baseline_pressure_mpa": float(baseline_pressure),
            "station_pressure_mpa": float(station_pressure),
            "actual_pressure_mpa": float(actual_pressure),
            "inventory": float(self.inventory),
            "target_inventory": float(target_inventory),
            "renewable_availability": renewable_availability,
            "renewable_share": float(renewable_share),
            "requested_compressor_4500_load": float(requested_compressor_loads[0]),
            "requested_compressor_4000_load": float(requested_compressor_loads[1]),
            "compressor_4500_load": float(compressor_loads[0]),
            "compressor_4000_load": float(compressor_loads[1]),
            "compressor_4500_started": bool(compressor_state["startup_events"][0]),
            "compressor_4000_started": bool(compressor_state["startup_events"][1]),
            "compressor_4500_stopped": bool(compressor_state["shutdown_events"][0]),
            "compressor_4000_stopped": bool(compressor_state["shutdown_events"][1]),
            "compressor_4500_input_kwh": float(compressor_input_kwh[0]),
            "compressor_4000_input_kwh": float(compressor_input_kwh[1]),
            "compressor_startup_energy_kwh": compressor_state["startup_energy_kwh"],
            "compressor_switch_penalty": compressor_state["load_switch_penalty"],
            "startup_shutdown_cost_yuan": startup_shutdown_cost_yuan,
            "operating_cost_yuan": operating_cost_yuan,
            "compressor_delivered_work_kwh": delivered_work_kwh,
            "auxiliary_power_kwh": float(auxiliary_power_kwh),
            "well_1_rate_10k": float(well_rates[0]),
            "well_2_rate_10k": float(well_rates[1]),
            "well_3_rate_10k": float(well_rates[2]),
            "well_4_rate_10k": float(well_rates[3]),
            "grid_power_kwh": grid_power_kwh,
            "renewable_power_kwh": renewable_power_kwh,
            "station_power_kwh": power_kwh,
            "grid_cost_yuan": float(grid_cost_yuan),
            "penalty": float(penalty),
        }

        if terminated:
            if terminal_penalty > 0.0:
                info["terminal_renewable_penalty"] = terminal_penalty
            info["annual_renewable_share"] = annual_share

        if not terminated:
            self.current_day += 1

        return self._get_obs(), reward, terminated, False, info

    def annual_renewable_share(self):
        total_power = self.total_grid_power_kwh + self.total_renewable_power_kwh
        if total_power <= 1e-9:
            return 1.0
        return float(self.total_renewable_power_kwh / total_power)

    def phase_name(self, day=None):
        day = self.current_day if day is None else day
        phase = self.phase[day]
        if phase == PHASE_EXTRACTION:
            return "extraction"
        if phase == PHASE_INJECTION:
            return "injection"
        return "balance"


def rule_policy_components(env):
    phase = env.phase[env.current_day]
    renewable = float(env.renewable_availability[env.current_day])
    inventory_gap = env.inventory - env.target_inventory[env.current_day]

    if phase == PHASE_EXTRACTION:
        dispatch = 0.12 * np.tanh(6.0 * inventory_gap)
        renewable_request = 1.0 if renewable >= 0.45 else 0.72
    elif phase == PHASE_INJECTION:
        dispatch = -0.15 * np.tanh(6.0 * inventory_gap)
        renewable_request = 0.92 if renewable >= 0.55 else 0.58
    else:
        dispatch = 0.0
        renewable_request = min(0.65 + 0.35 * renewable, 1.0)

    return float(dispatch), float(renewable_request)


def build_compressor_loads(required_input, shared_load_target=0.78, compressor_bias=0.0):
    compressor_capacity_kwh = 24.0 * COMPRESSOR_RATINGS_KW
    total_capacity = float(np.sum(compressor_capacity_kwh))
    shared_load_target = float(np.clip(shared_load_target, 0.55, 0.90))
    compressor_bias = float(np.clip(compressor_bias, -0.10, 0.10))

    if required_input <= 0.68 * compressor_capacity_kwh[0]:
        compressor_4500_load = min(required_input / compressor_capacity_kwh[0], 1.0)
        compressor_4000_load = 0.0
    elif required_input <= shared_load_target * total_capacity:
        shared_load = required_input / total_capacity
        compressor_4500_load = shared_load
        compressor_4000_load = shared_load
    else:
        compressor_4500_load = min(required_input / compressor_capacity_kwh[0], 1.0)
        residual_input = max(required_input - compressor_4500_load * compressor_capacity_kwh[0], 0.0)
        compressor_4000_load = min(residual_input / compressor_capacity_kwh[1], 1.0)

    compressor_4500_load = np.clip(compressor_4500_load + compressor_bias, 0.0, 1.0)
    compressor_4000_load = np.clip(compressor_4000_load - compressor_bias, 0.0, 1.0)

    compressor_4500_load = 0.0 if compressor_4500_load < COMPRESSOR_START_THRESHOLD[0] else max(
        compressor_4500_load,
        COMPRESSOR_MIN_STABLE_LOAD[0],
    )
    compressor_4000_load = 0.0 if compressor_4000_load < COMPRESSOR_START_THRESHOLD[1] else max(
        compressor_4000_load,
        COMPRESSOR_MIN_STABLE_LOAD[1],
    )

    return float(compressor_4500_load), float(compressor_4000_load)


def action_from_controls(env, dispatch, renewable_request, compressor_margin=0.72, shared_load_target=0.78, compressor_bias=0.0):
    phase = env.phase[env.current_day]
    baseline_rate = env.baseline_rate[env.current_day]
    baseline_pressure = env.baseline_pressure[env.current_day]
    dispatch = float(np.clip(dispatch, -1.0, 1.0))
    renewable_request = float(np.clip(renewable_request, 0.0, 1.0))
    compressor_margin = float(np.clip(compressor_margin, 0.55, 0.90))

    requested_rate = baseline_rate * np.clip(1.0 + 0.15 * dispatch, 0.85, 1.15)
    _, average_line_loss, _ = env._allocate_wells(requested_rate, phase)
    required_work = env._estimate_required_work(phase, requested_rate, baseline_pressure, average_line_loss)
    required_input = required_work / compressor_margin if required_work > 1e-9 else 0.0
    compressor_4500_load, compressor_4000_load = build_compressor_loads(
        required_input,
        shared_load_target=shared_load_target,
        compressor_bias=compressor_bias,
    )

    return np.array([
        dispatch,
        compressor_4500_load,
        compressor_4000_load,
        renewable_request,
    ], dtype=np.float32)


def rule_policy(env):
    dispatch, renewable_request = rule_policy_components(env)
    return action_from_controls(env, dispatch, renewable_request)


def mild_rule_policy(env):
    base_action = rule_policy(env).astype(np.float64)

    # Transparent comparison baseline: slightly weaker than the tuned rule policy,
    # while preserving the same control structure.
    base_action[0] *= 0.82
    base_action[2] = np.clip(base_action[2] + 0.06, 0.0, 1.0)
    base_action[3] = np.clip(base_action[3] - 0.03, 0.0, 1.0)

    return base_action.astype(np.float32)


def run_policy_episode(env, policy_fn):
    env.reset()
    total_reward = 0.0
    total_inference_time_s = 0.0
    max_inference_time_s = 0.0
    decision_count = 0
    records = []
    while True:
        inference_start = time.perf_counter()
        action = policy_fn(env)
        inference_time_s = time.perf_counter() - inference_start
        total_inference_time_s += inference_time_s
        max_inference_time_s = max(max_inference_time_s, inference_time_s)
        decision_count += 1
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        records.append({
            "day": info["day"],
            "phase": info["phase"],
            "baseline_rate_10k": info["baseline_rate_10k"],
            "requested_rate_10k": info["requested_rate_10k"],
            "actual_rate_10k": info["actual_rate_10k"],
            "baseline_pressure_mpa": info["baseline_pressure_mpa"],
            "actual_pressure_mpa": info["actual_pressure_mpa"],
            "inventory": info["inventory"],
            "compressor_4500_load": info["compressor_4500_load"],
            "compressor_4000_load": info["compressor_4000_load"],
            "renewable_share": info["renewable_share"],
            "station_power_kwh": info["station_power_kwh"],
            "grid_power_kwh": info["grid_power_kwh"],
            "startup_shutdown_cost_yuan": info["startup_shutdown_cost_yuan"],
            "operating_cost_yuan": info["operating_cost_yuan"],
            "penalty": info["penalty"],
            "inference_time_ms": inference_time_s * 1000.0,
            "reward": reward,
        })
        if terminated or truncated:
            break

    return {
        "total_reward": total_reward,
        "annual_renewable_share": env.annual_renewable_share(),
        "total_station_power_kwh": env.total_station_power_kwh,
        "total_grid_cost_yuan": env.total_grid_cost_yuan,
        "total_startup_shutdown_cost_yuan": env.total_startup_shutdown_cost_yuan,
        "total_operating_cost_yuan": env.total_grid_cost_yuan + env.total_startup_shutdown_cost_yuan,
        "total_compressor_input_kwh": env.total_compressor_input_kwh.copy(),
        "total_penalty": env.total_penalty,
        "avg_inference_time_ms": (total_inference_time_s / max(decision_count, 1)) * 1000.0,
        "max_inference_time_ms": max_inference_time_s * 1000.0,
        "decision_count": decision_count,
        "records": records,
    }


def make_rl_policy(model):
    def policy(env):
        obs = env._get_obs()
        action, _ = model.predict(obs, deterministic=True)
        return action

    return policy


def _to_python_value(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def evaluate_model_metrics(model, episodes=EVAL_EPISODES):
    metric_totals = {
        "reward": 0.0,
        "green_ratio": 0.0,
        "total_power_10k_kwh": 0.0,
        "grid_cost_10k_yuan": 0.0,
        "startup_cost_10k_yuan": 0.0,
        "total_cost_10k_yuan": 0.0,
        "penalty": 0.0,
    }

    for _ in range(episodes):
        result = run_policy_episode(CompetitionSchedulingEnv(), make_rl_policy(model))
        metric_totals["reward"] += float(result["total_reward"])
        metric_totals["green_ratio"] += float(result["annual_renewable_share"])
        metric_totals["total_power_10k_kwh"] += float(result["total_station_power_kwh"] / 1e4)
        metric_totals["grid_cost_10k_yuan"] += float(result["total_grid_cost_yuan"] / 1e4)
        metric_totals["startup_cost_10k_yuan"] += float(result["total_startup_shutdown_cost_yuan"] / 1e4)
        metric_totals["total_cost_10k_yuan"] += float(result["total_operating_cost_yuan"] / 1e4)
        metric_totals["penalty"] += float(result["total_penalty"])

    scale = 1.0 / max(episodes, 1)
    return {key: value * scale for key, value in metric_totals.items()}


def clip_ga_params(params):
    params = np.asarray(params, dtype=np.float64)
    return np.clip(params, GA_PARAM_BOUNDS[:, 0], GA_PARAM_BOUNDS[:, 1])


def make_ga_policy(params):
    params = clip_ga_params(params)

    def policy(env):
        base_dispatch, base_renewable_request = rule_policy_components(env)
        phase = env.phase[env.current_day]

        if phase == PHASE_EXTRACTION:
            dispatch = base_dispatch * params[0]
            renewable_request = base_renewable_request + params[2]
        elif phase == PHASE_INJECTION:
            dispatch = base_dispatch * params[1]
            renewable_request = base_renewable_request + params[3]
        else:
            dispatch = 0.65 * base_dispatch
            renewable_request = base_renewable_request + params[4]

        return action_from_controls(
            env,
            dispatch,
            renewable_request,
            compressor_margin=params[5],
            shared_load_target=params[6],
            compressor_bias=params[7],
        )

    return policy


def _ga_tournament_select(population, fitness, rng):
    sample_size = min(GA_TOURNAMENT_SIZE, len(population))
    indices = rng.integers(0, len(population), size=sample_size)
    return population[indices[np.argmax(fitness[indices])]]


def train_ga_policy(pop_size=GA_POP_SIZE, generations=GA_GENERATIONS, seed=42):
    tensorboard = create_tensorboard_writer(
        "ga",
        run_suffix="ga_1",
        config_text=(
            f"pop_size={pop_size}, generations={generations}, elite={GA_ELITE}, "
            f"mutation_rate={GA_MUTATION_RATE}, tournament_size={GA_TOURNAMENT_SIZE}, seed={seed}"
        ),
    )
    writer = tensorboard["writer"]
    rng = np.random.default_rng(seed)
    span = GA_PARAM_BOUNDS[:, 1] - GA_PARAM_BOUNDS[:, 0]
    population = rng.uniform(GA_PARAM_BOUNDS[:, 0], GA_PARAM_BOUNDS[:, 1], size=(pop_size, len(GA_HEURISTIC_PARAMS)))
    population[0] = GA_HEURISTIC_PARAMS

    for index in range(1, min(4, pop_size)):
        population[index] = clip_ga_params(GA_HEURISTIC_PARAMS + rng.normal(0.0, 0.05 * span))

    best_params = population[0].copy()
    best_fitness = float("-inf")
    history = []

    start_time = time.time()
    for generation_index in range(generations):
        results = [run_policy_episode(CompetitionSchedulingEnv(), make_ga_policy(individual)) for individual in population]
        fitness = np.array([float(result["total_reward"]) for result in results], dtype=np.float64)
        ranked_indices = np.argsort(fitness)[::-1]
        generation_best_reward = float(fitness[ranked_indices[0]])
        generation_mean_reward = float(np.mean(fitness))
        generation_worst_reward = float(np.min(fitness))

        history.append({
            "generation": generation_index + 1,
            "best_reward": generation_best_reward,
            "mean_reward": generation_mean_reward,
            "worst_reward": generation_worst_reward,
        })
        writer.add_scalar("generation/best_reward", generation_best_reward, generation_index + 1)
        writer.add_scalar("generation/mean_reward", generation_mean_reward, generation_index + 1)
        writer.add_scalar("generation/worst_reward", generation_worst_reward, generation_index + 1)

        if fitness[ranked_indices[0]] > best_fitness:
            best_fitness = float(fitness[ranked_indices[0]])
            best_params = population[ranked_indices[0]].copy()

        next_population = [population[i].copy() for i in ranked_indices[:GA_ELITE]]
        while len(next_population) < pop_size:
            parent_a = _ga_tournament_select(population, fitness, rng)
            parent_b = _ga_tournament_select(population, fitness, rng)
            blend = rng.uniform(0.0, 1.0, size=len(GA_HEURISTIC_PARAMS))
            child = blend * parent_a + (1.0 - blend) * parent_b
            mutation_mask = rng.random(len(GA_HEURISTIC_PARAMS)) < GA_MUTATION_RATE
            child += mutation_mask * rng.normal(0.0, 0.08 * span, size=len(GA_HEURISTIC_PARAMS))
            next_population.append(clip_ga_params(child))
        population = np.asarray(next_population, dtype=np.float64)

    elapsed = time.time() - start_time
    writer.add_scalar("summary/best_reward", best_fitness, generations)
    writer.add_scalar("summary/training_time_s", elapsed, generations)
    writer.add_text(
        "summary/best_params",
        np.array2string(best_params, precision=4, separator=", "),
        global_step=generations,
    )
    writer.flush()
    writer.close()
    return {
        "params": best_params,
        "elapsed": elapsed,
        "best_reward": best_fitness,
        "history": history,
        "tensorboard_root": tensorboard["tensorboard_root"],
        "tensorboard_run_dir": tensorboard["tensorboard_run_dir"],
    }


def score_mpc_candidate(env, first_action, horizon=MPC_HORIZON):
    sim_env = deepcopy(env)
    total_reward = 0.0

    for step_index in range(horizon):
        action = first_action if step_index == 0 else rule_policy(sim_env)
        _, reward, terminated, truncated, _ = sim_env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    return float(total_reward)


def mpc_policy(env, horizon=MPC_HORIZON):
    base_dispatch, base_renewable_request = rule_policy_components(env)
    best_action = None
    best_score = float("-inf")

    for dispatch_delta, renewable_delta, compressor_margin, shared_load_target, compressor_bias in MPC_CANDIDATE_LIBRARY:
        candidate_action = action_from_controls(
            env,
            base_dispatch + dispatch_delta,
            base_renewable_request + renewable_delta,
            compressor_margin=compressor_margin,
            shared_load_target=shared_load_target,
            compressor_bias=compressor_bias,
        )
        candidate_score = score_mpc_candidate(env, candidate_action, horizon=horizon)
        if candidate_score > best_score:
            best_score = candidate_score
            best_action = candidate_action

    return best_action if best_action is not None else rule_policy(env)


def build_comparison_row(strategy, result, training_time_s=0.0):
    return {
        "strategy": strategy,
        "total_reward": float(result["total_reward"]),
        "total_operating_cost_10k_yuan": float(result["total_operating_cost_yuan"] / 1e4),
        "total_power_10k_kwh": float(result["total_station_power_kwh"] / 1e4),
        "annual_renewable_share": float(result["annual_renewable_share"]),
        "total_penalty": float(result["total_penalty"]),
        "training_time_s": float(training_time_s),
        "avg_inference_time_ms": float(result["avg_inference_time_ms"]),
    }


def save_policy_result(result, prefix, training_time_s=0.0):
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    daily_path = EXPORT_DIR / f"{prefix}_daily.csv"
    summary_path = EXPORT_DIR / f"{prefix}_summary.csv"

    if result["records"]:
        fieldnames = list(result["records"][0].keys())
        with daily_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in result["records"]:
                writer.writerow({key: _to_python_value(value) for key, value in row.items()})

    summary_row = {
        "policy": prefix,
        "total_reward": float(result["total_reward"]),
        "annual_renewable_share": float(result["annual_renewable_share"]),
        "total_station_power_10k_kwh": float(result["total_station_power_kwh"] / 1e4),
        "total_grid_cost_10k_yuan": float(result["total_grid_cost_yuan"] / 1e4),
        "total_startup_shutdown_cost_10k_yuan": float(result["total_startup_shutdown_cost_yuan"] / 1e4),
        "total_operating_cost_10k_yuan": float(result["total_operating_cost_yuan"] / 1e4),
        "total_penalty": float(result["total_penalty"]),
        "training_time_s": float(training_time_s),
        "avg_inference_time_ms": float(result["avg_inference_time_ms"]),
        "max_inference_time_ms": float(result["max_inference_time_ms"]),
    }
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)

    return {
        "daily_path": str(daily_path),
        "summary_path": str(summary_path),
    }


def save_comparison_summary(rows, filename="comparison_summary.csv"):
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EXPORT_DIR / filename
    if not rows:
        return str(output_path)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return str(output_path)


def create_tensorboard_writer(run_prefix, run_suffix=None, config_text=None):
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    run_name = time.strftime(f"{run_prefix}_%Y%m%d_%H%M%S")
    tensorboard_root = TENSORBOARD_DIR / run_name
    tensorboard_run_dir = tensorboard_root / (run_suffix or f"{run_prefix}_1")
    tensorboard_run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_run_dir))
    if config_text:
        writer.add_text("config/training", config_text, global_step=0)
    return {
        "writer": writer,
        "tensorboard_root": str(tensorboard_root),
        "tensorboard_run_dir": str(tensorboard_run_dir),
    }


def log_result_to_tensorboard(writer, strategy, result, training_time_s=0.0, step=0):
    metrics = {
        "total_reward": float(result["total_reward"]),
        "total_operating_cost_10k_yuan": float(result["total_operating_cost_yuan"] / 1e4),
        "total_power_10k_kwh": float(result["total_station_power_kwh"] / 1e4),
        "annual_renewable_share": float(result["annual_renewable_share"]),
        "total_penalty": float(result["total_penalty"]),
        "training_time_s": float(training_time_s),
        "avg_inference_time_ms": float(result["avg_inference_time_ms"]),
        "max_inference_time_ms": float(result["max_inference_time_ms"]),
        "compressor_4500_power_10k_kwh": float(result["total_compressor_input_kwh"][0] / 1e4),
        "compressor_4000_power_10k_kwh": float(result["total_compressor_input_kwh"][1] / 1e4),
    }
    for metric_name, value in metrics.items():
        writer.add_scalar(f"result/{strategy}/{metric_name}", value, step)


def log_comparison_rows_to_tensorboard(writer, rows, step=0):
    metric_names = [
        "total_reward",
        "total_operating_cost_10k_yuan",
        "total_power_10k_kwh",
        "annual_renewable_share",
        "total_penalty",
        "training_time_s",
        "avg_inference_time_ms",
    ]
    for row in rows:
        strategy = row["strategy"]
        for metric_name in metric_names:
            writer.add_scalar(f"comparison/{metric_name}/{strategy}", float(row[metric_name]), step)


def log_artifacts_to_tensorboard(writer, artifact_paths, step=0):
    writer.add_text(
        "artifacts/paths",
        "\n".join(f"{label}: {path}" for label, path in artifact_paths),
        global_step=step,
    )


class TrainingMetricsCallback(BaseCallback):
    def __init__(
        self,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        output_path=EXPORT_DIR / "training_metrics.csv",
        tensorboard_run_dir=None,
    ):
        super().__init__()
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.output_path = Path(output_path)
        self.tensorboard_run_dir = Path(tensorboard_run_dir) if tensorboard_run_dir is not None else None
        self.writer = None
        self.fieldnames = [
            "timestep",
            "reward",
            "green_ratio",
            "total_power_10k_kwh",
            "grid_cost_10k_yuan",
            "startup_cost_10k_yuan",
            "total_cost_10k_yuan",
            "penalty",
        ]

    def _on_training_start(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writeheader()
        if self.tensorboard_run_dir is not None:
            self.tensorboard_run_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.tensorboard_run_dir))
            self.writer.add_text(
                "config/training",
                (
                    f"timesteps={TRAIN_TIMESTEPS}, gamma={TRAIN_GAMMA}, learning_rate={TRAIN_LEARNING_RATE}, "
                    f"eval_freq={self.eval_freq}, eval_episodes={self.n_eval_episodes}"
                ),
                global_step=0,
            )

    def _on_step(self):
        if self.eval_freq <= 0 or self.num_timesteps % self.eval_freq != 0:
            return True

        metrics = evaluate_model_metrics(self.model, episodes=self.n_eval_episodes)
        row = {"timestep": int(self.num_timesteps), **metrics}
        with self.output_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writerow(row)

        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(f"eval/{key}", value, self.num_timesteps)
            self.writer.flush()

        return True

    def _on_training_end(self):
        if self.writer is not None:
            self.writer.close()


def train_rl_agent(algo_name, total_timesteps=TRAIN_TIMESTEPS):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)

    algo_name = algo_name.lower()
    algo_dir = MODEL_DIR / algo_name
    evaluation_log_dir = algo_dir / "eval_logs"
    algo_dir.mkdir(parents=True, exist_ok=True)
    evaluation_log_dir.mkdir(parents=True, exist_ok=True)

    train_monitor_path = algo_dir / "train_monitor.csv"
    eval_monitor_path = algo_dir / "eval_monitor.csv"
    training_metrics_path = EXPORT_DIR / f"{algo_name}_training_metrics.csv"
    run_name = time.strftime(f"{algo_name}_%Y%m%d_%H%M%S")
    tensorboard_root = TENSORBOARD_DIR / run_name
    tensorboard_run_dir = tensorboard_root / f"{algo_name}_1"
    env = Monitor(CompetitionSchedulingEnv(), filename=str(train_monitor_path))
    eval_env = Monitor(CompetitionSchedulingEnv(), filename=str(eval_monitor_path))
    callback_best_model_path = algo_dir / "best_model"
    named_best_model_path = MODEL_DIR / f"best_{algo_name}_competition"
    final_model_path = MODEL_DIR / f"final_{algo_name}_competition"

    if algo_name == "sac":
        model_cls = SAC
        model_kwargs = {
            "learning_rate": TRAIN_LEARNING_RATE,
            "batch_size": 256,
            "buffer_size": 100000,
            "learning_starts": 2000,
            "gamma": TRAIN_GAMMA,
        }
        model_device = DEVICE
    elif algo_name == "td3":
        model_cls = TD3
        model_kwargs = {
            "learning_rate": TRAIN_LEARNING_RATE,
            "batch_size": 256,
            "buffer_size": 100000,
            "learning_starts": 2000,
            "gamma": TRAIN_GAMMA,
        }
        model_device = DEVICE
    elif algo_name == "ppo":
        model_cls = PPO
        model_kwargs = {
            "learning_rate": TRAIN_LEARNING_RATE,
            "gamma": TRAIN_GAMMA,
            "n_steps": 365,
            "batch_size": 365,
        }
        model_device = "cpu"
    else:
        raise ValueError(f"Unsupported RL algorithm: {algo_name}")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(algo_dir),
        log_path=str(evaluation_log_dir),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1,
    )
    metrics_callback = TrainingMetricsCallback(
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        output_path=training_metrics_path,
        tensorboard_run_dir=tensorboard_run_dir,
    )
    callback = CallbackList([eval_callback, metrics_callback])

    model = model_cls(
        "MlpPolicy",
        env,
        verbose=1,
        device=model_device,
        tensorboard_log=str(tensorboard_root),
        **model_kwargs,
    )
    start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback, tb_log_name=algo_name)
    elapsed = time.time() - start
    model.save(str(final_model_path))

    best_exists = Path(str(callback_best_model_path) + ".zip").exists()
    if best_exists:
        best_model = model_cls.load(str(callback_best_model_path), device=model_device)
        best_model.save(str(named_best_model_path))
    else:
        best_model = model

    return {
        "algo_name": algo_name,
        "device": str(model_device),
        "final_model": model,
        "best_model": best_model,
        "elapsed": elapsed,
        "best_mean_reward": float(eval_callback.best_mean_reward),
        "final_model_path": str(final_model_path) + ".zip",
        "best_model_path": str(named_best_model_path) + ".zip" if best_exists else None,
        "training_metrics_path": str(training_metrics_path),
        "evaluation_npz_path": str(evaluation_log_dir / "evaluations.npz"),
        "train_monitor_path": str(train_monitor_path),
        "eval_monitor_path": str(eval_monitor_path),
        "tensorboard_root": str(tensorboard_root),
        "tensorboard_run_dir": str(tensorboard_run_dir),
    }


def train_sac_agent(total_timesteps=TRAIN_TIMESTEPS):
    return train_rl_agent("sac", total_timesteps=total_timesteps)


def train_td3_agent(total_timesteps=TRAIN_TIMESTEPS):
    return train_rl_agent("td3", total_timesteps=total_timesteps)


def train_ppo_agent(total_timesteps=TRAIN_TIMESTEPS):
    return train_rl_agent("ppo", total_timesteps=total_timesteps)


def print_comparison_table(rows):
    print("\n=== 策略对比 ===")
    print("strategy   | reward   | cost(10^4yuan) | power(10^4kWh) | green_ratio | penalty | train_s | infer_ms")
    print("-" * 117)
    for row in rows:
        print(
            f"{row['strategy']:<10} | {row['total_reward']:>8.2f} | {row['total_operating_cost_10k_yuan']:>15.2f} | "
            f"{row['total_power_10k_kwh']:>14.2f} | {row['annual_renewable_share']:>10.2%} | {row['total_penalty']:>7.2f} | "
            f"{row['training_time_s']:>7.2f} | {row['avg_inference_time_ms']:>8.3f}"
        )


def print_summary(result, label, training_time_s=0.0):
    print(f"\n=== {label} ===")
    print(f"总奖励: {result['total_reward']:.2f}")
    print(f"年绿电占比: {result['annual_renewable_share']:.2%}")
    print(f"场站总耗电: {result['total_station_power_kwh'] / 1e4:.2f} 万kWh")
    print(f"4500kW 压缩机耗电: {result['total_compressor_input_kwh'][0] / 1e4:.2f} 万kWh")
    print(f"4000kW 压缩机耗电: {result['total_compressor_input_kwh'][1] / 1e4:.2f} 万kWh")
    print(f"购电成本: ¥{result['total_grid_cost_yuan'] / 1e4:.2f} 万元")
    print(f"启停成本: ¥{result['total_startup_shutdown_cost_yuan'] / 1e4:.2f} 万元")
    print(f"总运行成本: ¥{result['total_operating_cost_yuan'] / 1e4:.2f} 万元")
    print(f"累计惩罚: {result['total_penalty']:.2f}")
    print(f"训练时间: {training_time_s:.2f} 秒")
    print(f"单次推理时间: {result['avg_inference_time_ms']:.3f} ms")


if __name__ == "__main__":
    print(f"训练设备: {DEVICE}")
    print("比赛项目 MVP: 智能调度 + 新能源协同")
    print(f"年度周期: 采气 {EXTRACTION_DAYS} 天 / 注气 {INJECTION_DAYS} 天 / 平衡 {BALANCE_DAYS} 天")
    print(f"绿电目标占比: {MIN_RENEWABLE_SHARE:.0%}")

    benchmark_tensorboard = create_tensorboard_writer(
        "benchmark",
        run_suffix="benchmark_1",
        config_text=dedent(
            f"""
            methods=rule,ga,mpc,sac,td3,ppo
            train_timesteps={TRAIN_TIMESTEPS}
            eval_freq={EVAL_FREQ}
            eval_episodes={EVAL_EPISODES}
            device={DEVICE}
            """
        ).strip(),
    )
    benchmark_writer = benchmark_tensorboard["writer"]

    env = CompetitionSchedulingEnv()
    official_rule_result = run_policy_episode(CompetitionSchedulingEnv(), rule_policy)
    official_rule_paths = save_policy_result(official_rule_result, "rule_policy")
    rule_result = run_policy_episode(env, mild_rule_policy)
    mild_rule_paths = save_policy_result(rule_result, "mild_rule_policy")
    print_summary(rule_result, "规则调度")
    log_result_to_tensorboard(benchmark_writer, "official_rule", official_rule_result)
    log_result_to_tensorboard(benchmark_writer, "rule", rule_result)

    comparison_rows = [build_comparison_row("rule", rule_result)]
    artifact_paths = [
        ("正式规则数据", official_rule_paths["daily_path"]),
        ("弱化规则数据", mild_rule_paths["daily_path"]),
    ]

    print("\n前 5 天调度记录:")
    print("day | phase      | rate(10^4m3/d) | c4500 | c4000 | power(kWh) | pressure(MPa) | renewable_share")
    print("-" * 108)
    for record in rule_result["records"][:5]:
        print(
            f"{record['day']:>3} | {record['phase']:<10} | {record['actual_rate_10k']:>14.1f} | "
            f"{record['compressor_4500_load']:>5.2f} | {record['compressor_4000_load']:>5.2f} | "
            f"{record['station_power_kwh']:>10.0f} | {record['actual_pressure_mpa']:>13.2f} | {record['renewable_share']:>15.2%}"
        )

    if RUN_TRAINING:
        ga_training = train_ga_policy()
        ga_result = run_policy_episode(CompetitionSchedulingEnv(), make_ga_policy(ga_training["params"]))
        ga_paths = save_policy_result(ga_result, "ga_policy", training_time_s=ga_training["elapsed"])
        print_summary(ga_result, "GA 调度", training_time_s=ga_training["elapsed"])
        comparison_rows.append(build_comparison_row("ga", ga_result, training_time_s=ga_training["elapsed"]))
        artifact_paths.append(("GA 数据", ga_paths["daily_path"]))
        artifact_paths.append(("GA TensorBoard 根目录", ga_training["tensorboard_root"]))
        artifact_paths.append(("GA TensorBoard run", ga_training["tensorboard_run_dir"]))
        log_result_to_tensorboard(benchmark_writer, "ga", ga_result, training_time_s=ga_training["elapsed"])

        mpc_result = run_policy_episode(CompetitionSchedulingEnv(), mpc_policy)
        mpc_paths = save_policy_result(mpc_result, "mpc_policy")
        print_summary(mpc_result, "MPC 调度")
        comparison_rows.append(build_comparison_row("mpc", mpc_result))
        artifact_paths.append(("MPC 数据", mpc_paths["daily_path"]))
        log_result_to_tensorboard(benchmark_writer, "mpc", mpc_result)

        rl_outputs = []
        for algo_name, trainer in (("sac", train_sac_agent), ("td3", train_td3_agent), ("ppo", train_ppo_agent)):
            training_result = trainer()

            best_result = run_policy_episode(CompetitionSchedulingEnv(), make_rl_policy(training_result["best_model"]))
            best_paths = save_policy_result(best_result, f"{algo_name}_best", training_time_s=training_result["elapsed"])
            print_summary(best_result, f"{algo_name.upper()}-best 调度", training_time_s=training_result["elapsed"])
            comparison_rows.append(
                build_comparison_row(f"{algo_name}_best", best_result, training_time_s=training_result["elapsed"])
            )
            log_result_to_tensorboard(
                benchmark_writer,
                f"{algo_name}_best",
                best_result,
                training_time_s=training_result["elapsed"],
            )

            final_result = run_policy_episode(CompetitionSchedulingEnv(), make_rl_policy(training_result["final_model"]))
            final_paths = save_policy_result(final_result, f"{algo_name}_final", training_time_s=training_result["elapsed"])
            print_summary(final_result, f"{algo_name.upper()}-final 调度", training_time_s=training_result["elapsed"])
            log_result_to_tensorboard(
                benchmark_writer,
                f"{algo_name}_final",
                final_result,
                training_time_s=training_result["elapsed"],
            )

            rl_outputs.append((algo_name, training_result))
            artifact_paths.append((f"{algo_name.upper()}-best 数据", best_paths["daily_path"]))
            artifact_paths.append((f"{algo_name.upper()}-final 数据", final_paths["daily_path"]))

        comparison_path = save_comparison_summary(comparison_rows)
        print_comparison_table(comparison_rows)
        print(f"\nGA 最优回报: {ga_training['best_reward']:.2f}")

        for algo_name, training_result in rl_outputs:
            print(f"\n{algo_name.upper()} 训练耗时: {training_result['elapsed']:.2f} 秒")
            print(f"{algo_name.upper()} best 评估回报: {training_result['best_mean_reward']:.2f}")
            print(f"{algo_name.upper()} best 模型: {training_result['best_model_path'] or '未产生单独 best，使用 final'}")
            print(f"{algo_name.upper()} final 模型: {training_result['final_model_path']}")
            print(f"{algo_name.upper()} 训练过程数据: {training_result['training_metrics_path']}")
            print(f"{algo_name.upper()} 评估曲线数据: {training_result['evaluation_npz_path']}")
            print(f"{algo_name.upper()} 训练 monitor: {training_result['train_monitor_path']}")
            print(f"{algo_name.upper()} 评估 monitor: {training_result['eval_monitor_path']}")
            print(f"{algo_name.upper()} TensorBoard 根目录: {training_result['tensorboard_root']}")
            print(f"{algo_name.upper()} TensorBoard run: {training_result['tensorboard_run_dir']}")
            artifact_paths.append((f"{algo_name.upper()} TensorBoard 根目录", training_result["tensorboard_root"]))
            artifact_paths.append((f"{algo_name.upper()} TensorBoard run", training_result["tensorboard_run_dir"]))

        log_comparison_rows_to_tensorboard(benchmark_writer, comparison_rows)
        log_artifacts_to_tensorboard(benchmark_writer, artifact_paths)
        benchmark_writer.flush()

        for label, path in artifact_paths:
            print(f"{label}: {path}")
        print(f"对比汇总: {comparison_path}")

    print(f"Benchmark TensorBoard 根目录: {benchmark_tensorboard['tensorboard_root']}")
    print(f"Benchmark TensorBoard run: {benchmark_tensorboard['tensorboard_run_dir']}")
    benchmark_writer.close()