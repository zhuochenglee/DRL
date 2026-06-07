"""Single source of truth for the centrifugal-compressor model.

The same physics is used by the environment (per-step, scalar) and by the
planning baselines (DP/MPC, which evaluate it over grids). Keeping one
implementation guarantees that every method optimizes and is scored against
exactly the same compressor characteristics.

Equations follow Liu et al. (Energy & AI, 2024):
  Eq. 9  adiabatic head from the discharge (pressure) ratio
  Eq. 7  H/omega^2 = polyH(phi),  phi = Qin/omega   -> solve for omega
  Eq. 8  eta = polyE(phi)
  Eq. 11 P = Q * rho * H / eta
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class CompressorParams:
	kappa: float
	head_thermo_coeff: float
	qin_per_flow: float
	power_coeff: float
	omega_min: float
	omega_max: float
	poly_h: Tuple[float, float, float, float]    # ascending [AH, BH, CH, DH]
	poly_eta: Tuple[float, float, float, float]  # ascending [AE, BE, CE, DE]

	@staticmethod
	def from_env(env) -> "CompressorParams":
		return CompressorParams(
			kappa=float(env.kappa),
			head_thermo_coeff=float(env.head_thermo_coeff),
			qin_per_flow=float(env.qin_per_flow),
			power_coeff=float(env.power_coeff),
			omega_min=float(env.omega_min),
			omega_max=float(env.omega_max),
			poly_h=tuple(np.asarray(env.poly_h, dtype=np.float64).tolist()),
			poly_eta=tuple(np.asarray(env.poly_eta, dtype=np.float64).tolist()),
		)


def adiabatic_head(alpha: float, p: CompressorParams) -> float:
	# Eq. 9: H = (Z R T / M) * kappa/(kappa-1) * [ alpha^((kappa-1)/kappa) - 1 ].
	exp_term = (p.kappa - 1.0) / p.kappa
	ratio_term = max(alpha ** exp_term - 1.0, 0.0)
	return p.head_thermo_coeff * (p.kappa / (p.kappa - 1.0)) * ratio_term


def solve_speed(head: float, qin: float, p: CompressorParams) -> Tuple[float, float, float]:
	# Solve Eq. 7 for phi = Qin/omega via the cubic
	#   DH*phi^3 + (CH - H/Qin^2)*phi^2 + BH*phi + AH = 0,
	# pick the real positive root whose speed omega = Qin/phi best fits the band,
	# then clip omega to [omega_min, omega_max]. Returns (omega, phi_eff, omega_raw).
	if qin <= 1e-9 or head <= 0.0:
		idle = p.omega_min
		return idle, (qin / idle if qin > 0.0 else 0.0), idle

	coeffs = np.array(p.poly_h[::-1], dtype=np.float64)  # [DH, CH, BH, AH]
	coeffs[1] -= head / (qin * qin)
	roots = np.roots(coeffs)
	candidates = [r.real for r in roots if abs(r.imag) < 1e-9 and r.real > 1e-9]

	if candidates:
		def speed_gap(phi):
			w = qin / phi
			return max(p.omega_min - w, 0.0) + max(w - p.omega_max, 0.0)
		phi_star = min(candidates, key=speed_gap)
		omega_raw = qin / phi_star
	else:
		omega_raw = 0.5 * (p.omega_min + p.omega_max)

	omega = float(np.clip(omega_raw, p.omega_min, p.omega_max))
	phi_eff = qin / omega
	return omega, phi_eff, float(omega_raw)


def compressor_map(alpha: float, qin: float, p: CompressorParams):
	# Returns (omega, eta, head, phi, omega_raw).
	head = adiabatic_head(alpha, p)
	omega, phi, omega_raw = solve_speed(head, qin, p)
	eff_percent = np.polyval(np.array(p.poly_eta[::-1], dtype=np.float64), phi)  # Eq. 8 at phi
	eff = float(np.clip(eff_percent / 100.0, 0.60, 0.88))
	return float(omega), eff, float(head), float(phi), float(omega_raw)


def power_kwh(q0_abs: float, alpha: float, p: CompressorParams, dt_hour: float):
	# Returns (power_kwh, omega, eta, head, phi, omega_raw).
	qin = p.qin_per_flow * max(q0_abs, 0.0)
	omega, eta, head, phi, omega_raw = compressor_map(alpha, qin, p)
	# Eq. 11: P = Q * rho * H / eta; gas density/units lumped into power_coeff.
	power_kw = p.power_coeff * qin * max(head, 0.0) / max(eta, 1e-6)
	return power_kw * dt_hour, omega, eta, head, phi, omega_raw
