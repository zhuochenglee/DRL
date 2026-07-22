"""Re-shape the Figure-2 data into Prism-friendly tables (one colour-segment per column).

Reads the CSVs produced by export_fig2_data.py and writes:
  fig2a_prism.csv : X=phi; Y columns eta_surge / eta_feasible / eta_choke
                    (each filled only in its band, blank elsewhere; boundary points shared so
                     the coloured segments join). In Prism: XY table, plot the three Y columns
                     as one curve in three colours — no manual shading needed.
  fig2b_prism.csv : X=commanded_alpha; Y columns realized_OFF / realized_ON / identity
                    (OFF = dead band → 1; ON = feasible realized α; identity = dotted reference).

Run from repo root:  python paper/export_fig2_prism.py
"""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "raw_data")


def _read(name):
	with open(os.path.join(RAW, name)) as f:
		return list(csv.DictReader(f))


def main():
	# constants
	const = {r["quantity"]: r["value"] for r in _read("fig2_constants.csv")}
	phi_lo = float(const["phi_lower"]); phi_hi = float(const["phi_upper"])
	alpha_on = float(const["alpha_on"])

	# (a) eta curve split into surge / feasible / choke (<= on both bounds so segments touch)
	with open(os.path.join(RAW, "fig2a_prism.csv"), "w", newline="") as f:
		w = csv.writer(f); w.writerow(["phi", "eta_surge", "eta_feasible", "eta_choke"])
		for r in _read("fig2_eta_curve.csv"):
			ph = float(r["phi"]); eta = float(r["eta"])
			surge = eta if ph <= phi_lo else ""
			feas = eta if phi_lo <= ph <= phi_hi else ""
			choke = eta if ph >= phi_hi else ""
			w.writerow([ph, surge, feas, choke])

	# (b) realized alpha split into OFF (dead band) / ON (feasible) + identity reference
	with open(os.path.join(RAW, "fig2b_prism.csv"), "w", newline="") as f:
		w = csv.writer(f); w.writerow(["commanded_alpha", "realized_OFF", "realized_ON", "identity"])
		for r in _read("fig2_alpha_realization.csv"):
			a = float(r["commanded_alpha"]); rz = float(r["realized_alpha"])
			off = rz if a < alpha_on else ""
			on = rz if a >= alpha_on else ""
			w.writerow([a, off, on, round(a, 4)])

	print(f"phi band = [{phi_lo}, {phi_hi}], alpha_on = {alpha_on}")
	print("Saved fig2a_prism.csv (3 colour segments) and fig2b_prism.csv to", RAW)


if __name__ == "__main__":
	main()
