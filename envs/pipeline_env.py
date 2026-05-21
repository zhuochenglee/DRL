from dataclasses import dataclass, field, fields, replace
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class EnvConfig:
	# Time setup
	horizon: int = 24
	noise_scale: float = 0.0
	dt_hour: float = 1.0

	# Network topology and hydraulic coefficients
	A: np.ndarray = field(
		default_factory=lambda: np.array(
			[
				[-1.0, 0.0],
				[1.0, -1.0],
				[0.0, 1.0],
			],
			dtype=np.float64,
		)
	)
	K: np.ndarray = field(default_factory=lambda: np.array([0.02, 0.05], dtype=np.float64))
	beta: np.ndarray = field(default_factory=lambda: np.array([50.0, 40.0], dtype=np.float64))

	# Pressure and flow limits
	p0_ref: float = 100.0
	p_min_safe: float = 80.0
	p_max_safe: float = 150.0
	p_hard_min: float = 10.0
	p_hard_max: float = 200.0
	q1_max: float = 1000.0
	q2_max: float = 900.0

	# Reward weights
	w_pressure: float = 1000.0
	w_flow: float = 5.0
	w_terminal_linepack: float = 250.0

	# Compressor map and thermodynamic parameters
	kappa: float = 1.30
	power_coeff: float = 0.090
	omega_min: float = 5000.0
	omega_max: float = 9400.0
	poly_h: np.ndarray = field(
		default_factory=lambda: np.array([6.223e-6, -1.450e-5, 1.618e-5, -6.261e-6], dtype=np.float64)
	)
	poly_eta: np.ndarray = field(
		default_factory=lambda: np.array([134.806, -262.296, 390.048, -176.702], dtype=np.float64)
	)

	# Exogenous profiles
	demand_series: np.ndarray = field(
		default_factory=lambda: np.array(
			[
				120, 110, 110, 115, 120, 130, 150, 180,
				220, 250, 260, 250, 240, 240, 250, 260,
				270, 280, 280, 260, 220, 180, 150, 130,
			],
			dtype=np.float64,
		)
	)
	tou_price_series: np.ndarray = field(
		default_factory=lambda: np.array(
			[
				0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30,
				1.20, 1.20, 1.20, 1.20, 1.20, 1.20, 1.20, 1.20,
				1.20, 1.20, 1.20, 1.20, 1.20, 1.20, 0.30, 0.30,
			],
			dtype=np.float64,
		)
	)

	# Observation normalization constants
	demand_center: float = 200.0
	demand_scale: float = 120.0
	price_center: float = 0.75
	price_scale: float = 0.45
	pressure_center: float = 115.0
	pressure_scale: float = 35.0


class PaperInspiredDynamicLinepackEnv(gym.Env):
	"""
	DRL environment inspired by the paper's steady-state flow constraints,
	extended with dynamic linepack and time-of-use (TOU) electricity prices.

	Main objective: minimize electricity purchasing cost while satisfying
	pressure and compressor operating constraints.
	"""

	metadata = {"render_modes": []}

	def __init__(self, config: Optional[EnvConfig] = None, **overrides):
		super().__init__()
		self.config = self._build_config(config=config, overrides=overrides)
		cfg = self.config

		self.horizon = int(cfg.horizon)
		self.noise_scale = float(cfg.noise_scale)
		self.dt_hour = float(cfg.dt_hour)

		# Action: compressor discharge ratio alpha.
		self.action_space = spaces.Box(low=1.0, high=2.0, shape=(1,), dtype=np.float32)

		# Observation layout:
		# [norm_demand, norm_price, norm_p2, norm_p3, norm_linepack, sin_t, cos_t, horizon price profile]
		self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(7 + self.horizon,), dtype=np.float32)

		# Node-edge incidence matrix.
		# Rows: source node(1), internal node(2), demand node(3)
		# Cols: pipeline 1 (1->2), pipeline 2 (2->3)
		self.A = np.array(cfg.A, dtype=np.float64)
		self.A_source = self.A[0:1, :]
		self.A_internal = self.A[1:3, :]

		# Hydraulic resistance constants in q = sign(dp2) * sqrt(|dp2| / K)
		self.K = np.array(cfg.K, dtype=np.float64)

		# Linepack capacitance coefficients (mass-equivalent per pressure unit).
		# m_i = beta_i * p_i
		self.beta = np.array(cfg.beta, dtype=np.float64)

		self.p0_ref = float(cfg.p0_ref)
		self.p_min_safe = float(cfg.p_min_safe)
		self.p_max_safe = float(cfg.p_max_safe)
		self.p_hard_min = float(cfg.p_hard_min)
		self.p_hard_max = float(cfg.p_hard_max)

		self.q1_max = float(cfg.q1_max)
		self.q2_max = float(cfg.q2_max)

		self.w_pressure = float(cfg.w_pressure)
		self.w_flow = float(cfg.w_flow)
		self.w_terminal_linepack = float(cfg.w_terminal_linepack)

		# Compressor power model parameters and map coefficients.
		# Head and efficiency polynomial coefficients are aligned to the paper table style.
		self.kappa = float(cfg.kappa)
		self.power_coeff = float(cfg.power_coeff)
		self.omega_min = float(cfg.omega_min)
		self.omega_max = float(cfg.omega_max)
		self.poly_h = np.array(cfg.poly_h, dtype=np.float64)
		self.poly_eta = np.array(cfg.poly_eta, dtype=np.float64)

		self.current_hour = 0
		self.P_internal = np.array([100.0, 100.0], dtype=np.float64)
		self.M_internal = self.beta * self.P_internal
		self._linepack_target_end = float(np.sum(self.M_internal))

		self.demand_series = np.array(cfg.demand_series, dtype=np.float64)
		self.tou_price_series = np.array(cfg.tou_price_series, dtype=np.float64)
		if self.demand_series.shape != (self.horizon,) or self.tou_price_series.shape != (self.horizon,):
			raise ValueError("demand_series and tou_price_series must both match horizon length")

		self.obs_norm = {
			"demand_center": float(cfg.demand_center),
			"demand_scale": float(cfg.demand_scale),
			"price_center": float(cfg.price_center),
			"price_scale": float(cfg.price_scale),
			"pressure_center": float(cfg.pressure_center),
			"pressure_scale": float(cfg.pressure_scale),
		}

		self._episode_demand = self.demand_series.copy()
		self._episode_prices = self.tou_price_series.copy()

	def _build_config(self, config: Optional[EnvConfig], overrides):
		cfg = config if config is not None else EnvConfig()
		if not overrides:
			return cfg

		valid = {f.name for f in fields(EnvConfig)}
		unknown = sorted(set(overrides.keys()) - valid)
		if unknown:
			raise TypeError(f"Unknown EnvConfig override(s): {unknown}")
		return replace(cfg, **overrides)

	def _validate_series(self, values, name):
		arr = np.asarray(values, dtype=np.float64)
		if arr.shape != (self.horizon,):
			raise ValueError(f"{name} must have length {self.horizon}, got shape={arr.shape}")
		return arr

	def reset(self, seed=None, options=None):
		super().reset(seed=seed)

		self.current_hour = 0
		self.P_internal = np.array([100.0, 100.0], dtype=np.float64)
		self.M_internal = self.beta * self.P_internal
		self._linepack_target_end = float(np.sum(self.M_internal))

		self._episode_demand = self.demand_series.copy()
		self._episode_prices = self.tou_price_series.copy()

		if options is not None:
			if "demand" in options:
				self._episode_demand = self._validate_series(options["demand"], "demand")
			if "prices" in options:
				self._episode_prices = self._validate_series(options["prices"], "prices")

		if self.noise_scale > 0.0:
			noise = self.np_random.normal(0.0, self.noise_scale, size=self.horizon)
			self._episode_demand = np.clip(self._episode_demand * (1.0 + noise), 50.0, 400.0)
			shift = int(self.np_random.integers(-2, 3))
			self._episode_prices = np.roll(self._episode_prices, shift)

		return self._get_obs(hour=self.current_hour), {}

	def _get_obs(self, hour):
		h = int(np.clip(hour, 0, self.horizon - 1))
		demand = self._episode_demand[h]
		price = self._episode_prices[h]

		norm_demand = np.clip(
			(demand - self.obs_norm["demand_center"]) / max(self.obs_norm["demand_scale"], 1e-6),
			-1.0,
			1.0,
		)
		norm_price = np.clip(
			(price - self.obs_norm["price_center"]) / max(self.obs_norm["price_scale"], 1e-6),
			-1.0,
			1.0,
		)
		norm_p2 = np.clip(
			(self.P_internal[0] - self.obs_norm["pressure_center"]) / max(self.obs_norm["pressure_scale"], 1e-6),
			-1.0,
			1.0,
		)
		norm_p3 = np.clip(
			(self.P_internal[1] - self.obs_norm["pressure_center"]) / max(self.obs_norm["pressure_scale"], 1e-6),
			-1.0,
			1.0,
		)

		linepack_ratio = np.clip(np.sum(self.M_internal) / np.sum(self.beta * self.p_max_safe), 0.0, 1.0)
		norm_linepack = 2.0 * linepack_ratio - 1.0

		t = 2.0 * np.pi * h / self.horizon
		norm_prices = np.clip(
			(self._episode_prices - self.obs_norm["price_center"]) / max(self.obs_norm["price_scale"], 1e-6),
			-1.0,
			1.0,
		).astype(np.float32)

		head = np.array(
			[norm_demand, norm_price, norm_p2, norm_p3, norm_linepack, np.sin(t), np.cos(t)],
			dtype=np.float32,
		)
		return np.concatenate([head, norm_prices])

	def _compute_flow(self, p_source):
		# Paper-style steady flow relation (Weymouth-like):
		# q_e = sgn(Δ(p^2)_e) * sqrt(|Δ(p^2)_e| / K_e)
		p_source_sq = np.array([p_source * p_source], dtype=np.float64)
		p_internal_sq = self.P_internal * self.P_internal
		dp2 = -(self.A_source.T @ p_source_sq + self.A_internal.T @ p_internal_sq)
		q = np.sign(dp2) * np.sqrt(np.abs(dp2) / self.K)
		return q

	def _compressor_map(self, alpha):
		# Paper mapping idea: action(alpha) -> required head -> speed and efficiency.
		exp_term = (self.kappa - 1.0) / self.kappa
		head_factor = max(alpha ** exp_term - 1.0, 0.0)

		# Convert head demand into normalized speed ratio and then into rpm.
		norm_speed = np.clip(0.58 + 1.25 * head_factor, 0.0, 1.0)
		omega = self.omega_min + norm_speed * (self.omega_max - self.omega_min)

		n = omega / self.omega_max
		eff_percent = np.polyval(self.poly_eta, n)
		eff = np.clip(eff_percent / 100.0, 0.60, 0.88)
		return float(omega), float(eff), float(head_factor)

	def _compressor_power_kwh(self, q1, alpha):
		# Compressor electricity model in kWh for one step.
		omega, eta, head_factor = self._compressor_map(alpha)
		power_kw = self.power_coeff * max(q1, 0.0) * max(head_factor, 0.0) * (omega / self.omega_max) / max(eta, 1e-6)
		return power_kw * self.dt_hour, omega, eta

	def step(self, action):
		alpha = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
		hour = self.current_hour
		demand = float(self._episode_demand[hour])
		price = float(self._episode_prices[hour])

		p_source = self.p0_ref * alpha
		q = self._compute_flow(p_source)

		q_ext_internal = np.array([0.0, -demand], dtype=np.float64)
		# Nodal mass balance:
		# m_dot = A_internal @ q + q_ext
		net_flow_internal = self.A_internal @ q + q_ext_internal

		# Dynamic linepack:
		# m_{t+1} = m_t + Δt * m_dot, and m = beta * p
		self.M_internal = self.M_internal + self.dt_hour * net_flow_internal
		m_min = self.beta * self.p_hard_min
		m_max = self.beta * self.p_hard_max
		self.M_internal = np.clip(self.M_internal, m_min, m_max)
		self.P_internal = self.M_internal / self.beta

		q1 = float(abs((self.A_source @ q)[0]))
		power_kwh, omega, eta = self._compressor_power_kwh(q1=q1, alpha=alpha)
		purchase_cost = power_kwh * price

		reward = -purchase_cost
		violation = False
		penalty = 0.0

		for p in self.P_internal:
			if p < self.p_min_safe:
				violation = True
				penalty += self.w_pressure * (self.p_min_safe - p)
			if p > self.p_max_safe:
				violation = True
				penalty += self.w_pressure * (p - self.p_max_safe)

		if abs(q[0]) > self.q1_max:
			violation = True
			penalty += self.w_flow * (abs(q[0]) - self.q1_max)
		if abs(q[1]) > self.q2_max:
			violation = True
			penalty += self.w_flow * (abs(q[1]) - self.q2_max)

		# Economic objective with physical penalties:
		# r_t = - Cost_t - Penalty_t
		reward -= penalty

		self.current_hour += 1
		terminated = bool(self.current_hour >= self.horizon)
		if terminated:
			# Terminal inventory consistency to prevent end-of-day linepack depletion.
			terminal_linepack_gap = abs(float(np.sum(self.M_internal)) - self._linepack_target_end)
			penalty += self.w_terminal_linepack * terminal_linepack_gap / max(self._linepack_target_end, 1e-6)
			reward -= self.w_terminal_linepack * terminal_linepack_gap / max(self._linepack_target_end, 1e-6)
		else:
			terminal_linepack_gap = 0.0
		next_obs = self._get_obs(hour=self.horizon - 1 if terminated else self.current_hour)

		info = {
			"hour": hour,
			"alpha": alpha,
			"price": price,
			"demand": demand,
			"p_source": p_source,
			"p2": float(self.P_internal[0]),
			"p3": float(self.P_internal[1]),
			"linepack": float(np.sum(self.M_internal)),
			"q1": float(q[0]),
			"q2": float(q[1]),
			"compressor_speed_rpm": float(omega),
			"compressor_efficiency": float(eta),
			"power_kwh": float(power_kwh),
			"purchase_cost": float(purchase_cost),
			"penalty": float(penalty),
			"terminal_linepack_gap": float(terminal_linepack_gap),
			"violation": violation,
		}

		return next_obs, float(reward), terminated, False, info
