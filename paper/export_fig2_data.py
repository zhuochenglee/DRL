"""Export the numerical data behind Figure 2 (computed from envs/compressor.py).

Writes three files into raw_data/:
  fig2_eta_curve.csv         : characteristic map  φ, η(φ), H/ω²(φ)   (Eq. 3)
  fig2_alpha_realization.csv : per commanded α at p2=100 — the full operating point and the
                               realized (dead-band-snapped) α  (Eqs. 2-5)
  fig2_constants.csv         : the scalar bounds (φ band, ω band, α_on, etc.)

Run from repo root:  python paper/export_fig2_data.py
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from envs.pipeline_env import EnvConfig, PaperInspiredDynamicLinepackEnv
from envs import compressor

OUT = os.path.join(ROOT, "raw_data")
P2 = 100.0   # representative downstream pressure used for the figure


def main():
	env = PaperInspiredDynamicLinepackEnv(config=EnvConfig())
	p = env._comp_params
	k0 = float(env.K[env.source_pipe])
	dt = float(env.dt_hour)
	poly_eta = np.array(p.poly_eta[::-1]); poly_h = np.array(p.poly_h[::-1])

	# (a) characteristic curves vs phi
	phi = np.round(np.linspace(0.5, 1.6, 221), 4)
	with open(os.path.join(OUT, "fig2_eta_curve.csv"), "w", newline="") as f:
		w = csv.writer(f); w.writerow(["phi", "eta", "H_over_omega2", "in_feasible_band"])
		for ph in phi:
			eta = float(np.clip(np.polyval(poly_eta, ph) / 100.0, 0.5, 0.95))
			h = float(np.polyval(poly_h, ph))
			w.writerow([ph, round(eta, 5), f"{h:.3e}", int(env.phi_lower <= ph <= env.phi_upper)])

	# (b) per commanded alpha: full operating point + realized alpha
	alphas = np.round(np.linspace(1.0, 2.0, 201), 4)
	rows = []
	for a in alphas:
		p_src = env.p0_ref * a
		dp20 = p_src * p_src - P2 * P2
		q0 = float(np.sign(dp20) * np.sqrt(abs(dp20) / k0))
		qin = p.qin_per_flow * abs(q0)
		omega, eta, head, phi_eff, omega_raw = compressor.compressor_map(a, qin, p)
		pw, *_ = compressor.power_kwh(abs(q0), float(a), p, dt)
		realized = compressor.effective_discharge_ratio(a, P2, env.p0_ref, k0, p)
		feasible = int(realized > 1.0 + 1e-6)
		rows.append([a, round(realized, 4), round(p_src, 2), round(abs(q0), 2), round(qin, 1),
		             round(head, 3), round(omega, 1), round(omega_raw, 1), round(phi_eff, 4),
		             round(eta, 4), round(pw, 4), feasible])
	with open(os.path.join(OUT, "fig2_alpha_realization.csv"), "w", newline="") as f:
		w = csv.writer(f)
		w.writerow(["commanded_alpha", "realized_alpha", "p_source", "q0", "Qin", "head",
		            "omega", "omega_raw", "phi", "eta", "power_kwh", "feasible_on"])
		w.writerows(rows)

	# alpha_on = smallest commanded alpha that stays ON
	on = [r[0] for r in rows if r[1] > 1.0 + 1e-6]
	alpha_on = min(on) if on else float("nan")

	with open(os.path.join(OUT, "fig2_constants.csv"), "w", newline="") as f:
		w = csv.writer(f); w.writerow(["quantity", "value", "note"])
		w.writerow(["p2_state", P2, "downstream pressure used for panel (b)"])
		w.writerow(["alpha_on", round(alpha_on, 4), "on-threshold at p2=100 (dead band = (1, alpha_on))"])
		w.writerow(["phi_lower", round(env.phi_lower, 4), "surge bound = Qin_min/omega_min"])
		w.writerow(["phi_upper", round(env.phi_upper, 4), "choke bound = Qin_max/omega_max"])
		w.writerow(["omega_min", env.omega_min, "rpm"]); w.writerow(["omega_max", env.omega_max, "rpm"])
		w.writerow(["qin_min", env.qin_min, ""]); w.writerow(["qin_max", env.qin_max, ""])
		w.writerow(["kappa", p.kappa, "specific heat ratio"])
		w.writerow(["poly_eta_ascending", list(p.poly_eta), "[AE,BE,CE,DE]"])
		w.writerow(["poly_h_ascending", list(p.poly_h), "[AH,BH,CH,DH]"])

	print(f"alpha_on (p2={P2}) = {alpha_on:.3f}")
	print("Saved fig2_eta_curve.csv, fig2_alpha_realization.csv, fig2_constants.csv to", OUT)


if __name__ == "__main__":
	main()
