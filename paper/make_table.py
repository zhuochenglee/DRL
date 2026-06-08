"""Render the paper's results table + headline numbers from the experiment CSVs.

Usage: python paper/make_table.py
Reads models/exports/paper/results_summary.csv (+ raw for latency) and prints a
Markdown table and the derived numbers (DP-optimum gap, speed-ups) to paste into draft.md.
"""

import csv
import collections
from pathlib import Path

EXP = Path("models/exports/paper")
ORDER = ["DP-oracle", "MPC", "GA", "PPO", "TD3", "SAC", "Constrained-SAC", "Rule"]


def load_summary():
	rows = {}
	for r in csv.DictReader(open(EXP / "results_summary.csv")):
		rows[r["method"]] = r
	return rows


def load_latency():
	t = collections.defaultdict(list)
	for r in csv.DictReader(open(EXP / "results_raw.csv")):
		if r["kind"] == "nominal":
			t[r["method"]].append(float(r["time"]))
	return {m: sum(v) / len(v) for m, v in t.items()}


def fmt_latency(s):
	return f"{s*1000:.1f} ms" if s < 1 else f"{s:.1f} s"


def main():
	S = load_summary()
	L = load_latency()
	print("| Method | Nominal cost | Nominal viol-h | Robust cost | Robust viol-h | Mean η | Latency |")
	print("|---|---|---|---|---|---|---|")
	for m in ORDER:
		if m not in S:
			continue
		r = S[m]
		nc = f"{float(r['nominal_cost_mean']):.2f} ± {float(r['nominal_cost_std']):.1f}"
		rc = f"{float(r['robust_cost_mean']):.2f} ± {float(r['robust_cost_std']):.1f}"
		print(f"| {m} | {nc} | {float(r['nominal_viol']):.1f} | {rc} | "
		      f"{float(r['robust_viol_mean']):.1f} | {float(r['mean_eff']):.3f} | {fmt_latency(L.get(m,0))} |")

	# headline derived numbers
	dp = float(S["DP-oracle"]["nominal_cost_mean"])
	print("\n--- headline numbers ---")
	for m in ["MPC", "GA", "SAC", "Constrained-SAC", "PPO", "TD3"]:
		if m in S:
			c = float(S[m]["nominal_cost_mean"])
			gap = 100 * (c - dp) / dp
			print(f"{m:<16} cost={c:6.2f}  gap_vs_DP={gap:+6.1f}%  viol={float(S[m]['nominal_viol']):.1f}")
	if "MPC" in L:
		for m in ["SAC", "Constrained-SAC"]:
			if m in L and L[m] > 0:
				print(f"{m} latency speed-up vs MPC: {L['MPC']/L[m]:.0f}x ; vs GA: {L['GA']/L[m]:.0f}x")


if __name__ == "__main__":
	main()
