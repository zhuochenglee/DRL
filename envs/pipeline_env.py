from dataclasses import dataclass, field, fields, replace
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs import compressor


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
	# Index of the pipe leaving the (single) source compressor; its throughput is the
	# compressor inlet flow Qin. Default 0 = the first pipe.
	source_pipe: int = 0
	# Optional per-pipe flow limit (length m). If None, built from q1_max/q2_max for the
	# default 2-pipe network. Generalizes the flow constraint to arbitrary topologies.
	q_max: Optional[np.ndarray] = None
	# Optional per-internal-node demand matrix (n_internal x horizon). If None, the single
	# demand_series is placed on the LAST internal node (the default gun-barrel behavior).
	demand_matrix: Optional[np.ndarray] = None

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
	# Compressor constraint weights (paper Eq. 20): speed out of band (per rpm) and
	# surge/choke on the normalized flow phi = Qin/omega (per unit of phi).
	w_speed: float = 1.0
	w_surge: float = 2000.0
	# Soft pressure margin: added ONLY to the (learning-signal) constraint cost, giving a
	# dense gradient as pressure approaches the safe band before a hard violation. It does
	# NOT change the binary violation flag or the scoring penalty, so baselines are unaffected.
	p_soft_margin: float = 8.0
	w_soft_margin: float = 0.3
	# Ablation switch: realize the discharge-ratio command against the compressor's feasible
	# set (dead-band -> off). Disabling it lets the policy command the infeasible band, which
	# the ablation in the paper shows brings envelope violations back.
	enable_feasibility_realization: bool = True

	# Optional fixed terminal linepack target (kg equiv.).
	# If None, the target is derived from the initial steady-state pressures (default = 9000).
	linepack_target: Optional[float] = None

	# Compressor map and thermodynamic parameters (paper Eqs. 7-11).
	kappa: float = 1.30
	# Lumped density/units coefficient in the power relation P = Q * rho * H / eta (Eq. 11).
	# Calibrated so the per-step electricity power stays in the same range as before
	# (a few kW at a nominal operating point), keeping the reward weights meaningful.
	power_coeff: float = 4.5e-6
	# Lumped thermodynamic head coefficient Z*R*T/M (kJ/kg) for the adiabatic head (Eq. 9).
	head_thermo_coeff: float = 138.0
	# Converts the abstract pipe-1 flow into the compressor inlet volumetric flow Qin used
	# by the characteristic polynomials (Eqs. 7-8); lands Qin in the paper's ~4000-12500 band.
	qin_per_flow: float = 10.0
	omega_min: float = 5000.0
	omega_max: float = 9400.0
	# Compressor inlet volumetric-flow bounds (m3/h, paper case studies), used together
	# with the speed bounds to define the surge/choke band on phi = Qin/omega (Eq. 20).
	qin_min: float = 4000.0
	qin_max: float = 12500.0
	# Ascending-power coefficients [A, B, C, D] of the compressor characteristic maps, with
	# the normalized inlet-flow argument phi = Qin/omega (paper Table 3):
	#   H / omega^2 = AH + BH*phi + CH*phi^2 + DH*phi^3   (Eq. 7)
	#   eta         = AE + BE*phi + CE*phi^2 + DE*phi^3   (Eq. 8)
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
		# [norm_total_demand, norm_price, norm_p (one per internal node), norm_linepack,
		#  sin_t, cos_t, horizon price profile, horizon total-demand profile]
		# The full price+demand look-ahead gives the agent the same forecast information an
		# MPC controller would use, so the comparison reflects control quality, not access to
		# information. (n_internal is known after the incidence matrix is parsed below.)
		_n_internal = int(np.array(cfg.A).shape[0] - 1)
		self.observation_space = spaces.Box(
			low=-1.0, high=1.0, shape=(5 + _n_internal + 2 * self.horizon,), dtype=np.float32)

		# Node-edge incidence matrix. Row 0 is the source node (with the compressor),
		# rows 1.. are the internal/demand nodes that hold line-pack; columns are pipes.
		# This supports an arbitrary single-source topology (gun-barrel, branch, tree).
		self.A = np.array(cfg.A, dtype=np.float64)
		self.n_internal = self.A.shape[0] - 1
		self.n_pipes = self.A.shape[1]
		self.A_source = self.A[0:1, :]
		self.A_internal = self.A[1:, :]
		self.source_pipe = int(cfg.source_pipe)
		# Internal node immediately downstream of the source compressor (the node the
		# source pipe enters); its pressure sets the compressor's feasible operating band.
		_down = np.where(self.A_internal[:, self.source_pipe] > 0)[0]
		self._source_down_node = int(_down[0]) if len(_down) else 0

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
		# Per-pipe flow limits (length n_pipes). Default reproduces the 2-pipe network.
		if cfg.q_max is not None:
			self.q_max = np.array(cfg.q_max, dtype=np.float64)
		elif self.n_pipes == 2:
			self.q_max = np.array([self.q1_max, self.q2_max], dtype=np.float64)
		else:
			self.q_max = np.full(self.n_pipes, self.q1_max, dtype=np.float64)

		self.w_pressure = float(cfg.w_pressure)
		self.w_flow = float(cfg.w_flow)
		self.w_terminal_linepack = float(cfg.w_terminal_linepack)
		self.w_speed = float(cfg.w_speed)
		self.w_surge = float(cfg.w_surge)
		self.p_soft_margin = float(cfg.p_soft_margin)
		self.w_soft_margin = float(cfg.w_soft_margin)
		self.enable_feasibility_realization = bool(cfg.enable_feasibility_realization)

		# Compressor power model parameters and map coefficients.
		# Head and efficiency polynomial coefficients are aligned to the paper table style.
		self.kappa = float(cfg.kappa)
		self.power_coeff = float(cfg.power_coeff)
		self.head_thermo_coeff = float(cfg.head_thermo_coeff)
		self.qin_per_flow = float(cfg.qin_per_flow)
		self.omega_min = float(cfg.omega_min)
		self.omega_max = float(cfg.omega_max)
		self.qin_min = float(cfg.qin_min)
		self.qin_max = float(cfg.qin_max)
		# Surge/choke band on the normalized characteristic phi = Qin/omega (Eq. 20):
		#   phi_lower = Qin_min/omega_min,  phi_upper = Qin_max/omega_max
		self.phi_lower = self.qin_min / self.omega_min
		self.phi_upper = self.qin_max / self.omega_max
		self.poly_h = np.array(cfg.poly_h, dtype=np.float64)
		self.poly_eta = np.array(cfg.poly_eta, dtype=np.float64)
		# Cached parameter bundle for the shared compressor model (envs/compressor.py),
		# the single source of truth reused by the DP/MPC planning baselines.
		self._comp_params = compressor.CompressorParams.from_env(self)

		self.current_hour = 0
		self.P_internal = np.full(self.n_internal, 100.0, dtype=np.float64)
		self.M_internal = self.beta * self.P_internal
		# Use explicit target if provided, otherwise derive from initial steady state.
		self._linepack_target_end = (
			float(cfg.linepack_target)
			if cfg.linepack_target is not None
			else float(np.sum(self.M_internal))
		)

		# Demand: a per-internal-node matrix D (n_internal x horizon). For the default
		# gun-barrel, the single demand_series sits on the last internal node. The total
		# hourly demand (demand_series) and the per-node spatial fractions are cached so
		# perturbations/forecasts can rescale the whole profile while preserving where gas
		# is withdrawn.
		self.tou_price_series = np.array(cfg.tou_price_series, dtype=np.float64)
		base_series = np.array(cfg.demand_series, dtype=np.float64)
		if cfg.demand_matrix is not None:
			D = np.array(cfg.demand_matrix, dtype=np.float64)
			if D.shape != (self.n_internal, self.horizon):
				raise ValueError(f"demand_matrix must be ({self.n_internal}, {self.horizon}), got {D.shape}")
		else:
			D = np.zeros((self.n_internal, self.horizon), dtype=np.float64)
			D[-1, :] = base_series  # single demand on the last internal node
		self.demand_matrix = D
		self.demand_series = D.sum(axis=0)                       # total hourly demand
		self._spatial_frac = D / np.maximum(self.demand_series[None, :], 1e-9)
		if self.tou_price_series.shape != (self.horizon,):
			raise ValueError("tou_price_series must match horizon length")

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
		self.P_internal = np.full(self.n_internal, 100.0, dtype=np.float64)
		self.M_internal = self.beta * self.P_internal
		# Re-derive target each reset (respects fixed override from config).
		self._linepack_target_end = (
			float(self.config.linepack_target)
			if self.config.linepack_target is not None
			else float(np.sum(self.M_internal))
		)

		# _episode_demand is the TOTAL hourly demand profile; the per-node split uses the
		# cached spatial fractions (see step()).
		self._episode_demand = self.demand_series.copy()
		self._episode_prices = self.tou_price_series.copy()

		if options is not None:
			if "demand" in options:
				self._episode_demand = self._validate_series(options["demand"], "demand")
			if "prices" in options:
				self._episode_prices = self._validate_series(options["prices"], "prices")

		if self.noise_scale > 0.0:
			noise = self.np_random.normal(0.0, self.noise_scale, size=self.horizon)
			self._episode_demand = np.clip(self._episode_demand * (1.0 + noise), 1.0, None)
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
		norm_p = np.clip(
			(self.P_internal - self.obs_norm["pressure_center"]) / max(self.obs_norm["pressure_scale"], 1e-6),
			-1.0,
			1.0,
		).astype(np.float32)

		linepack_ratio = np.clip(np.sum(self.M_internal) / np.sum(self.beta * self.p_max_safe), 0.0, 1.0)
		norm_linepack = 2.0 * linepack_ratio - 1.0

		t = 2.0 * np.pi * h / self.horizon
		norm_prices = np.clip(
			(self._episode_prices - self.obs_norm["price_center"]) / max(self.obs_norm["price_scale"], 1e-6),
			-1.0,
			1.0,
		).astype(np.float32)
		norm_demands = np.clip(
			(self._episode_demand - self.obs_norm["demand_center"]) / max(self.obs_norm["demand_scale"], 1e-6),
			-1.0,
			1.0,
		).astype(np.float32)

		head = np.array(
			[norm_demand, norm_price] + list(norm_p) + [norm_linepack, np.sin(t), np.cos(t)],
			dtype=np.float32,
		)
		return np.concatenate([head, norm_prices, norm_demands])

	def _compute_flow(self, p_source):
		# Paper-style steady flow relation (Weymouth-like):
		# q_e = sgn(Δ(p^2)_e) * sqrt(|Δ(p^2)_e| / K_e)
		p_source_sq = np.array([p_source * p_source], dtype=np.float64)
		p_internal_sq = self.P_internal * self.P_internal
		dp2 = -(self.A_source.T @ p_source_sq + self.A_internal.T @ p_internal_sq)
		q = np.sign(dp2) * np.sqrt(np.abs(dp2) / self.K)
		return q

	def _compressor_power_kwh(self, q1, alpha):
		# Delegate to the shared compressor model (envs/compressor.py) so the env and the
		# DP/MPC baselines use identical physics. Returns
		# (power_kwh, omega, eta, head, phi, omega_raw).
		return compressor.power_kwh(q1, alpha, self._comp_params, self.dt_hour)

	def step(self, action):
		alpha = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
		# Realize the command against the compressor's feasible set: a discharge ratio in
		# the sub-minimum-speed dead band is physically infeasible and is taken as unit-off.
		if self.enable_feasibility_realization:
			alpha = compressor.effective_discharge_ratio(
				alpha, float(self.P_internal[self._source_down_node]), self.p0_ref,
				float(self.K[self.source_pipe]), self._comp_params)
		hour = self.current_hour
		demand = float(self._episode_demand[hour])                 # total hourly demand
		demand_vec = self._spatial_frac[:, hour] * self._episode_demand[hour]  # per-node
		price = float(self._episode_prices[hour])

		p_source = self.p0_ref * alpha
		q = self._compute_flow(p_source)

		# Nodal mass balance: m_dot = A_internal @ q - demand_per_node
		net_flow_internal = self.A_internal @ q - demand_vec

		# Dynamic linepack:
		# m_{t+1} = m_t + Δt * m_dot, and m = beta * p
		self.M_internal = self.M_internal + self.dt_hour * net_flow_internal
		m_min = self.beta * self.p_hard_min
		m_max = self.beta * self.p_hard_max
		self.M_internal = np.clip(self.M_internal, m_min, m_max)
		self.P_internal = self.M_internal / self.beta

		q1 = float(abs((self.A_source @ q)[0]))
		power_kwh, omega, eta, head, phi, omega_raw = self._compressor_power_kwh(q1=q1, alpha=alpha)
		purchase_cost = power_kwh * price

		reward = -purchase_cost
		violation = False
		penalty = 0.0
		# Weight-free, unit-normalized constraint cost c (the CMDP cost signal used by the
		# Lagrangian-SAC agent). Each term is the violation magnitude relative to its band,
		# so a single dual variable lambda can trade economics against feasibility.
		constraint_cost = 0.0
		p_band = max(self.p_max_safe - self.p_min_safe, 1e-6)

		for p in self.P_internal:
			if p < self.p_min_safe:
				violation = True
				penalty += self.w_pressure * (self.p_min_safe - p)
				constraint_cost += (self.p_min_safe - p) / p_band
			if p > self.p_max_safe:
				violation = True
				penalty += self.w_pressure * (p - self.p_max_safe)
				constraint_cost += (p - self.p_max_safe) / p_band
			# Soft margin (learning signal only): ramps up as p enters within p_soft_margin
			# of either safe bound, encouraging the policy to keep an operating margin.
			constraint_cost += self.w_soft_margin * max(self.p_min_safe + self.p_soft_margin - p, 0.0) / self.p_soft_margin
			constraint_cost += self.w_soft_margin * max(p - (self.p_max_safe - self.p_soft_margin), 0.0) / self.p_soft_margin

		for e in range(self.n_pipes):
			qe = abs(float(q[e]))
			if qe > self.q_max[e]:
				violation = True
				penalty += self.w_flow * (qe - self.q_max[e])
				constraint_cost += (qe - self.q_max[e]) / self.q_max[e]

		# Compressor operating-envelope constraints (paper Eq. 20), only when the unit is
		# actually running (head > 0, i.e. alpha > 1). Two checks:
		#   1. Rotational speed must stay within [omega_min, omega_max]. omega_raw is the
		#      natural speed solved from Eq. 7; demanding a speed outside the band is a violation.
		#   2. Surge/choke: the normalized flow phi = Qin/omega must stay within the band
		#      [phi_lower, phi_upper] = [Qin_min/omega_min, Qin_max/omega_max].
		if head > 0.0:
			omega_band = max(self.omega_max - self.omega_min, 1e-6)
			phi_band = max(self.phi_upper - self.phi_lower, 1e-6)
			if omega_raw < self.omega_min:
				violation = True
				penalty += self.w_speed * (self.omega_min - omega_raw)
				constraint_cost += (self.omega_min - omega_raw) / omega_band
			elif omega_raw > self.omega_max:
				violation = True
				penalty += self.w_speed * (omega_raw - self.omega_max)
				constraint_cost += (omega_raw - self.omega_max) / omega_band

			if phi < self.phi_lower:
				violation = True
				penalty += self.w_surge * (self.phi_lower - phi)
				constraint_cost += (self.phi_lower - phi) / phi_band
			elif phi > self.phi_upper:
				violation = True
				penalty += self.w_surge * (phi - self.phi_upper)
				constraint_cost += (phi - self.phi_upper) / phi_band

		# Economic objective with physical penalties:
		# r_t = - Cost_t - Penalty_t
		reward -= penalty

		self.current_hour += 1
		terminated = bool(self.current_hour >= self.horizon)
		if terminated:
			# Terminal inventory consistency to prevent end-of-day linepack depletion.
			terminal_linepack_gap = abs(float(np.sum(self.M_internal)) - self._linepack_target_end)
			rel_gap = terminal_linepack_gap / max(self._linepack_target_end, 1e-6)
			penalty += self.w_terminal_linepack * rel_gap
			reward -= self.w_terminal_linepack * rel_gap
			constraint_cost += rel_gap
		else:
			terminal_linepack_gap = 0.0
		next_obs = self._get_obs(hour=self.horizon - 1 if terminated else self.current_hour)

		info = {
			"hour": hour,
			"alpha": alpha,
			"price": price,
			"demand": demand,
			"p_source": p_source,
			# p2/p3 kept for figure compatibility (first two internal nodes); "pressures"
			# and "p_min" generalize to any topology. p3 is taken as the worst-case node so
			# the dispatch figure shows the binding (lowest-pressure) demand node.
			"p2": float(self.P_internal[0]),
			"p3": float(self.P_internal[-1]),
			"p_min": float(np.min(self.P_internal)),
			"pressures": [float(x) for x in self.P_internal],
			"linepack": float(np.sum(self.M_internal)),
			"q1": float(q[0]),
			"q2": float(q[self.n_pipes - 1]),
			"compressor_speed_rpm": float(omega),
			"compressor_speed_demand_rpm": float(omega_raw),
			"compressor_efficiency": float(eta),
			"compressor_head": float(head),
			"flow_coeff_phi": float(phi),
			"power_kwh": float(power_kwh),
			"purchase_cost": float(purchase_cost),
			"penalty": float(penalty),
			"econ_reward": float(-purchase_cost),
			"constraint_cost": float(constraint_cost),
			"terminal_linepack_gap": float(terminal_linepack_gap),
			"violation": violation,
		}

		return next_obs, float(reward), terminated, False, info


def branched_benchmark_config(**overrides) -> EnvConfig:
	"""A 3-internal-node branched transmission network (one source compressor).

	Topology:  source(0) --p0--> junction(1) --p1--> demand(2)
	                                         \--p2--> demand(3)
	Pipe resistances K are derived from Table-2-range lengths/diameters via a
	Weymouth-style K ∝ L / D^5 (scaled to the env's pressure regime). Total hourly
	demand matches the default single-node case but is split 55/45 across the two
	demand nodes, so the observation normalization and price profile carry over.
	DP/MPC remain tractable (3-D pressure state).
	"""
	base = EnvConfig()
	demand = np.asarray(base.demand_series, dtype=np.float64)
	A = np.array([
		[-1.0,  0.0,  0.0],   # source
		[ 1.0, -1.0, -1.0],   # junction (node 1)
		[ 0.0,  1.0,  0.0],   # demand node 2
		[ 0.0,  0.0,  1.0],   # demand node 3
	], dtype=np.float64)
	pipe_LD = [(100.0, 0.406), (68.0, 0.406), (80.0, 0.432)]  # (length km, diameter m)
	k_rel = np.array([L / (D ** 5) for L, D in pipe_LD], dtype=np.float64)
	K = k_rel / k_rel.max() * 0.05
	beta = np.array([50.0, 40.0, 40.0], dtype=np.float64)
	demand_matrix = np.vstack([np.zeros(base.horizon), 0.55 * demand, 0.45 * demand])
	return EnvConfig(
		A=A, K=K, beta=beta, source_pipe=0,
		demand_matrix=demand_matrix,
		q_max=np.array([1400.0, 900.0, 800.0], dtype=np.float64),
		**overrides,
	)


def gaslib_benchmark_config(**overrides) -> EnvConfig:
	"""Sub-network extracted from the real GasLib-40 instance (German low-calorific
	transmission network), with its NATIVE pipe geometry.

	We take the line from the compressor-station node ``innode_6`` through the demand
	chain ``sink_13 -> sink_14 -> sink_10`` (GasLib-40 pipes with their real lengths and
	diameters); the compressor station at ``innode_6`` provides the discharge-ratio
	control. This is a gun-barrel chain of three real demand nodes:

	    source/innode_6 (compressor) --p0--> sink_13 --p1--> sink_14 --p2--> sink_10
	      p0: 21.56 km, 1000 mm     p1: 7.00 km, 1000 mm     p2: 58.22 km, 800 mm

	Pipe resistances K ∝ L/D^5 and line-pack capacitances beta ∝ L·D^2 are computed from
	the GasLib-40 geometry (K scaled so the max sits in the env's pressure regime; beta
	scaled to the same total as the other configs). The three sinks carry equal demand in
	the GasLib scenario, so the daily total is split evenly across them. Electricity price
	is the same TOU profile (GasLib is gas-only and carries no electricity data).
	Provenance: GasLib-40-v1 (gaslib.zib.de), pipes 3/4/5.
	"""
	base = EnvConfig()
	demand = np.asarray(base.demand_series, dtype=np.float64)
	A = np.array([
		[-1.0,  0.0,  0.0],   # source / innode_6 (compressor)
		[ 1.0, -1.0,  0.0],   # sink_13
		[ 0.0,  1.0, -1.0],   # sink_14
		[ 0.0,  0.0,  1.0],   # sink_10
	], dtype=np.float64)
	# GasLib-40 native (length km, diameter m) for pipes innode_6->sink_13->sink_14->sink_10
	pipe_LD = [(21.56, 1.000), (7.00, 1.000), (58.22, 0.800)]
	# K ∝ L/D^5 and beta ∝ L·D^2 preserve the GasLib geometry's *relative ordering*, but are
	# affinely mapped into the env's numerically-stable band: the raw ratios (25x K spread,
	# a tiny mid-chain capacitance) make the simplified explicit-Euler line-pack model stiff,
	# so K is mapped to [0.012, 0.05] and beta is floored, keeping the ordering.
	k_rel = np.array([L / (D ** 5) for L, D in pipe_LD], dtype=np.float64)
	kn = (k_rel - k_rel.min()) / (k_rel.max() - k_rel.min())
	K = 0.012 + kn * (0.05 - 0.012)                         # -> [~0.015, 0.012, 0.05]
	vol = np.array([L * D ** 2 for L, D in pipe_LD], dtype=np.float64)
	beta = np.maximum(vol / vol.sum() * 130.0, 38.0)        # floor the stiff mid node
	demand_matrix = np.vstack([demand / 3.0, demand / 3.0, demand / 3.0])  # equal sinks
	return EnvConfig(
		A=A, K=K, beta=beta, source_pipe=0,
		demand_matrix=demand_matrix,
		q_max=np.array([1400.0, 900.0, 600.0], dtype=np.float64),
		**overrides,
	)
