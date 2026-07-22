"""Analysis figures from the saved CSVs (no retraining):

  fig_ablation.png        : Constrained-SAC component ablations (Table 1b visualised)
  fig_cost_latency.png    : cost-vs-latency trade-off on the gun-barrel network
  fig_summary_networks.png: cost (× DP optimum) per method across the three networks

Run from repo root:  python paper/make_analysis_figures.py
"""
import csv
import collections
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "raw_data")
BLUE, RED, GREEN, GREY, ORANGE = "#2c6fbb", "#d1495b", "#3aaa64", "#9aa0a6", "#e8923b"


def _summary(net):
	rows = {}
	with open(os.path.join(RAW, net, "results_summary.csv")) as f:
		for r in csv.DictReader(f):
			rows[r["method"]] = r
	return rows


def _latency_ms(net):
	t = collections.defaultdict(list)
	with open(os.path.join(RAW, net, "results_raw.csv")) as f:
		for r in csv.DictReader(f):
			if r["kind"] == "nominal":
				t[r["method"]].append(float(r["time"]))
	return {m: 1000.0 * np.mean(v) for m, v in t.items()}


# ── Fig: ablation (Table 1b) ─────────────────────────────────────────────────
def fig_ablation():
	agg = collections.defaultdict(lambda: collections.defaultdict(list))
	with open(os.path.join(RAW, "ablations.csv")) as f:
		for r in csv.DictReader(f):
			agg[r["variant"]]["cost"].append(float(r["cost"]))
			agg[r["variant"]]["viol"].append(float(r["viol"]))
			agg[r["variant"]]["tlp"].append(float(r["tlp"]))
	order = ["full", "no_feasibility_realization", "no_soft_margin", "no_surge_penalty", "no_terminal_linepack"]
	labels = ["Full", "− feasibility\nrealization", "− soft\nmargin", "− surge\npenalty", "− terminal\npenalty"]
	cost = [np.mean(agg[v]["cost"]) for v in order]
	cost_sd = [np.std(agg[v]["cost"]) for v in order]
	viol = [np.mean(agg[v]["viol"]) for v in order]
	tlp = [np.mean(agg[v]["tlp"]) for v in order]
	colors = [GREY, RED, GREEN, GREY, GREY]

	fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
	b = a1.bar(labels, cost, yerr=cost_sd, capsize=4, color=colors)
	for bar, g in zip(b, tlp):
		a1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4, f"gap={g:.0f}", ha="center", fontsize=7.5)
	a1.set_ylabel("24-h electricity cost"); a1.set_title("(a) Cost (term-gap annotated)", fontsize=10)
	a1.grid(axis="y", alpha=0.3)

	b2 = a2.bar(labels, viol, color=colors)
	for bar, v in zip(b2, viol):
		a2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f"{v:.1f}", ha="center", fontsize=8)
	a2.set_ylabel("constraint-violation hours"); a2.set_title("(b) Violation-hours", fontsize=10)
	a2.grid(axis="y", alpha=0.3)

	fig.suptitle("Ablation (gun-barrel): removing the feasibility realization breaks feasibility; "
	             "removing the soft margin recovers near-optimal cost", fontsize=9.5, y=1.02)
	fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_ablation.png"), dpi=150, bbox_inches="tight")
	plt.close(fig)


# ── Fig: cost vs latency (gun-barrel) ────────────────────────────────────────
def fig_cost_latency():
	S = _summary("gunbarrel"); L = _latency_ms("gunbarrel")
	methods = ["DP-oracle", "MPC", "GA", "PPO", "TD3", "SAC", "Constrained-SAC", "Distilled-MPC", "Rule"]
	fig, ax = plt.subplots(figsize=(8.2, 5.2))
	for m in methods:
		if m not in S:
			continue
		c = float(S[m]["nominal_cost_mean"]); lat = L[m]; viol = float(S[m]["nominal_viol"])
		feasible = viol < 0.5
		col = GREEN if feasible else RED
		mk = "*" if m == "Distilled-MPC" else ("s" if m in ("DP-oracle", "MPC", "GA") else "o")
		ax.scatter(lat, c, s=190 if mk == "*" else 90, c=col, marker=mk, edgecolor="black", zorder=3)
		# per-label offsets to avoid the DP-oracle/MPC overlap (both ~1.1 s, cost 5.38)
		yoff = {"MPC": -0.85, "DP-oracle": 0.45, "Distilled-MPC": 0.5}.get(m, 0.4)
		xfac = {"GA": 0.55}.get(m, 1.12)
		ax.annotate(m, (lat, c), xytext=(lat*xfac, c+yoff), fontsize=8)
	ax.set_xscale("log")
	ax.set_xlabel("deployment latency per 24-h dispatch (ms, log scale)")
	ax.set_ylabel("nominal 24-h cost")
	ax.set_title("Cost vs latency (gun-barrel). Lower-left = cheaper & faster.\n"
	             "green = feasible (≈0 viol), red = infeasible; ★ = Distilled-MPC", fontsize=9.5)
	ax.grid(alpha=0.3, which="both")
	ax.annotate("ideal", (3.2, 5.3), fontsize=9, color=BLUE, weight="bold")
	fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_cost_latency.png"), dpi=150, bbox_inches="tight")
	plt.close(fig)


# ── Fig: cross-network summary (cost / DP optimum) ───────────────────────────
def fig_summary_networks():
	nets = [("gunbarrel", "Gun-barrel"), ("branched", "Branched"), ("gaslib", "GasLib-40")]
	methods = ["GA", "MPC", "Distilled-MPC", "SAC", "Constrained-SAC"]
	data = {}  # net -> {method -> cost/DP}
	for key, _ in nets:
		S = _summary(key); dp = float(S["DP-oracle"]["nominal_cost_mean"])
		data[key] = {m: float(S[m]["nominal_cost_mean"]) / dp for m in methods if m in S}
	x = np.arange(len(methods)); w = 0.26
	fig, ax = plt.subplots(figsize=(9, 4.4))
	colors = [BLUE, ORANGE, GREEN]
	for i, (key, name) in enumerate(nets):
		vals = [data[key].get(m, np.nan) for m in methods]
		ax.bar(x + (i-1)*w, vals, w, label=name, color=colors[i])
	ax.axhline(1.0, color="black", ls="--", lw=1)
	ax.text(len(methods)-0.6, 1.03, "DP optimum (×1)", fontsize=8)
	ax.set_xticks(x); ax.set_xticklabels(methods, rotation=12)
	ax.set_ylabel("nominal cost  (× DP optimum)")
	ax.set_title("Cost relative to the DP optimum across the three networks\n"
	             "(Distilled-MPC ≈ optimum everywhere; from-scratch SAC/C-SAC carry a premium)", fontsize=9.5)
	ax.legend(title="network"); ax.grid(axis="y", alpha=0.3)
	fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_summary_networks.png"), dpi=150, bbox_inches="tight")
	plt.close(fig)


if __name__ == "__main__":
	fig_ablation()
	fig_cost_latency()
	fig_summary_networks()
	print("Saved fig_ablation.png, fig_cost_latency.png, fig_summary_networks.png to", HERE)
