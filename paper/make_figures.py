"""Generate the three schematic figures for the paper (Figs 1-3).

  fig1_networks.png   : the three test-network topologies
  fig2_compressor.png : compressor characteristic map + the disconnected feasible action set
  fig3_method.png     : method overview (CMDP loop -> constrained SAC -> MPC distillation)

Fig 2 uses the real compressor model in envs/compressor.py so the feasible band is exact.
Run from the repo root:  python paper/make_figures.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle
import numpy as np

from envs.pipeline_env import EnvConfig, PaperInspiredDynamicLinepackEnv
from envs import compressor

OUT = os.path.dirname(os.path.abspath(__file__))
BLUE, RED, GREEN, GREY = "#2c6fbb", "#d1495b", "#3aaa64", "#888888"


# ── Fig 1: network topologies ────────────────────────────────────────────────
def fig_networks():
	fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
	for ax in axes:
		ax.set_xlim(-0.6, 4.9); ax.set_ylim(-1.6, 1.6); ax.axis("off")

	def node(ax, x, y, label, kind="node"):
		fc = {"src": "#cfe3f7", "node": "white", "dem": "#fde2e4"}[kind]
		ax.add_patch(Circle((x, y), 0.30, fc=fc, ec="black", lw=1.3, zorder=3))
		ax.text(x, y, label, ha="center", va="center", fontsize=8.5, zorder=4)

	def pipe(ax, p1, p2, lab=None):
		ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12,
		                             lw=1.6, color=GREY, zorder=1,
		                             shrinkA=16, shrinkB=16))
		if lab:
			ax.text((p1[0]+p2[0])/2, (p1[1]+p2[1])/2 + 0.18, lab, ha="center", fontsize=7, color=GREY)

	def compressor_marker(ax, x, y):
		ax.add_patch(Rectangle((x-0.16, y-0.16), 0.32, 0.32, fc=GREEN, ec="black", lw=1.2, zorder=5))
		ax.text(x, y-0.5, "compressor\n(α)", ha="center", fontsize=7, color=GREEN)

	# (a) gun-barrel
	ax = axes[0]; ax.set_title("(a) Gun-barrel element", fontsize=10)
	node(ax, 0, 0, "S", "src"); node(ax, 2, 0, "2"); node(ax, 4, 0, "3", "dem")
	pipe(ax, (0, 0), (2, 0), "p₁"); pipe(ax, (2, 0), (4, 0), "p₂")
	compressor_marker(ax, 0, 0); ax.text(4, -0.55, "demand", ha="center", fontsize=7, color=RED)

	# (b) branched benchmark
	ax = axes[1]; ax.set_title("(b) Branched benchmark", fontsize=10)
	node(ax, 0, 0, "S", "src"); node(ax, 1.8, 0, "1"); node(ax, 3.8, 0.9, "2", "dem"); node(ax, 3.8, -0.9, "3", "dem")
	pipe(ax, (0, 0), (1.8, 0), "p₀"); pipe(ax, (1.8, 0), (3.8, 0.9), "p₁"); pipe(ax, (1.8, 0), (3.8, -0.9), "p₂")
	compressor_marker(ax, 0, 0)

	# (c) GasLib-40 chain
	ax = axes[2]; ax.set_title("(c) GasLib-40-derived chain", fontsize=10)
	node(ax, 0, 0, "i6", "src"); node(ax, 1.6, 0, "s13", "dem"); node(ax, 3.0, 0, "s14", "dem"); node(ax, 4.3, 0, "s10", "dem")
	pipe(ax, (0, 0), (1.6, 0), "22km"); pipe(ax, (1.6, 0), (3.0, 0), "7km"); pipe(ax, (3.0, 0), (4.3, 0), "58km")
	compressor_marker(ax, 0, 0)

	fig.suptitle("Test networks (S = source with compressor; shaded = demand node)", fontsize=10, y=1.02)
	fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_networks.png"), dpi=150, bbox_inches="tight")
	plt.close(fig)


# ── Fig 2: compressor map + disconnected feasible set ────────────────────────
def fig_compressor():
	env = PaperInspiredDynamicLinepackEnv(config=EnvConfig())
	p = env._comp_params
	phi_lo, phi_hi = env.phi_lower, env.phi_upper

	fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

	# (a) characteristic curves vs phi
	phi = np.linspace(0.5, 1.6, 300)
	eta = np.clip(np.polyval(np.array(p.poly_eta[::-1]), phi) / 100.0, 0.5, 0.95)
	axL.plot(phi, eta, color=BLUE, lw=2, label="efficiency η(φ)")
	axL.axvspan(0.5, phi_lo, color=RED, alpha=0.12); axL.axvspan(phi_hi, 1.6, color=RED, alpha=0.12)
	axL.axvspan(phi_lo, phi_hi, color=GREEN, alpha=0.10)
	axL.axvline(phi_lo, color=GREEN, ls="--", lw=1); axL.axvline(phi_hi, color=GREEN, ls="--", lw=1)
	axL.text((phi_lo+phi_hi)/2, 0.55, "feasible\nφ band", ha="center", color=GREEN, fontsize=9)
	axL.text(0.62, 0.6, "surge", color=RED, fontsize=8); axL.text(1.46, 0.6, "choke", color=RED, fontsize=8)
	axL.set_xlabel("normalised inlet flow  φ = Qin/ω"); axL.set_ylabel("efficiency η")
	axL.set_title("(a) Compressor characteristic map (Eqs. 2–3)", fontsize=10); axL.legend(fontsize=8); axL.grid(alpha=0.3)

	# (b) commanded -> realized alpha (the dead-band snap), from the real model at p2=100
	p2 = 100.0
	cmd = np.linspace(1.0, 2.0, 400)
	realized = np.array([compressor.effective_discharge_ratio(a, p2, env.p0_ref, float(env.K[0]), p) for a in cmd])
	# find alpha_on (first commanded alpha that stays on)
	on = cmd[realized > 1.0 + 1e-6]
	alpha_on = on.min() if len(on) else 2.0
    # over-speed region (commanded alpha whose natural speed exceeds omega_max)
	axR.plot([1, 2], [1, 2], color=GREY, ls=":", lw=1, label="identity (no realization)")
	axR.plot(cmd, realized, color=BLUE, lw=2.2, label="realized $\\tilde α$ (Eq. 5)")
	axR.axvspan(1.0, alpha_on, color=RED, alpha=0.12)
	axR.text((1.0+alpha_on)/2, 1.6, "dead band\n→ OFF", ha="center", color=RED, fontsize=8.5)
	axR.axvspan(alpha_on, 2.0, color=GREEN, alpha=0.08)
	axR.text((alpha_on+2.0)/2, 1.15, "feasible ON", ha="center", color=GREEN, fontsize=8.5)
	axR.set_xlabel("commanded discharge ratio  α"); axR.set_ylabel("realized discharge ratio  $\\tilde α$")
	axR.set_title("(b) Disconnected feasible action set", fontsize=10); axR.legend(fontsize=8, loc="upper left"); axR.grid(alpha=0.3)

	fig.suptitle("Compressor feasible set: a command in the sub-minimum-speed dead band is realized as unit-off",
	             fontsize=10, y=1.01)
	fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_compressor.png"), dpi=150, bbox_inches="tight")
	plt.close(fig)
	return alpha_on


# ── Fig 3: method overview ────────────────────────────────────────────────────
def fig_method():
	fig, ax = plt.subplots(figsize=(12, 4.2)); ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")

	def box(x, y, w, h, text, fc):
		ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
		                            fc=fc, ec="black", lw=1.2))
		ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=8.2)

	def arrow(p1, p2):
		ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13, lw=1.5, color="black"))

	# Stage A: CMDP loop
	ax.text(2.0, 5.7, "(a) Dynamic CMDP environment", ha="center", fontsize=9, weight="bold")
	box(0.3, 3.6, 3.4, 0.9, "state $s_t$ (pressures, line-pack,\nprice/demand look-ahead)", "#eef3fb")
	box(0.3, 2.3, 3.4, 0.8, "policy $\\pi$ → discharge ratio $α$", "#dfeafc")
	box(0.3, 1.0, 3.4, 0.9, "feasibility realization (Eq. 5)\n→ Weymouth flow + line-pack (Eq. 6)", "#eef3fb")
	arrow((2.0, 3.6), (2.0, 3.1)); arrow((2.0, 2.3), (2.0, 1.9))
	ax.add_patch(FancyArrowPatch((3.7, 1.45), (3.95, 1.45), arrowstyle="-|>", mutation_scale=12, lw=1.4))
	# reward/cost out
	box(4.0, 1.0, 1.7, 0.9, "$r^{econ}$ (Eq. 7)\n+ cost $c_t$", "#fff3d6")
	arrow((4.85, 1.0), (4.85, 0.4)); arrow((4.85, 0.4), (2.0, 0.4)); arrow((2.0, 0.4), (2.0, 1.0))

	# Stage B: constrained SAC
	ax.text(7.4, 5.7, "(b) Constrained (Lagrangian) SAC", ha="center", fontsize=9, weight="bold")
	box(6.0, 3.4, 2.9, 1.0, "SAC actor–critic\nmaximize $r^{econ}-λ\\,c_t$ (Eq. 9)", "#dfeafc")
	box(6.0, 1.9, 2.9, 0.9, "dual ascent on $λ$ (Eq. 10)\n(drives violations → 0)", "#e7f6ec")
	arrow((7.45, 3.4), (7.45, 2.8));
	ax.add_patch(FancyArrowPatch((7.45, 2.8), (7.45, 3.4), arrowstyle="-|>", mutation_scale=12, lw=1.3, color=GREEN))
	arrow((5.7, 2.7), (6.0, 2.7))  # cost in
	ax.text(5.2, 2.9, "$c_t$", fontsize=9)

	# Stage C: distillation
	ax.text(10.8, 5.7, "(c) MPC distillation", ha="center", fontsize=9, weight="bold")
	box(9.5, 3.6, 2.4, 0.9, "MPC / DP teacher\n(feedback law from\nthe env model)", "#fff3d6")
	box(9.5, 2.3, 2.4, 0.8, "DAgger relabel\nvisited states", "#fde9d6")
	box(9.5, 1.0, 2.4, 0.9, "millisecond MLP\nstudent policy", "#dfeafc")
	arrow((10.7, 3.6), (10.7, 3.1)); arrow((10.7, 2.3), (10.7, 1.9))
	# (b) and (c) are parallel methods on the same env -- no SAC->teacher arrow (the teacher
	# is the DP/MPC feedback law built from the env model, independent of the constrained SAC).
	ax.text(6.0, 0.45, "(a) is the shared environment; (b) and (c) are two independent solution methods on it",
	        fontsize=7.5, style="italic", color="#555")

	fig.suptitle("Method overview: a physics-faithful CMDP solved by constraint-aware SAC, with an MPC controller distilled into a fast policy",
	             fontsize=9.5, y=1.0)
	fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_method.png"), dpi=150, bbox_inches="tight")
	plt.close(fig)


if __name__ == "__main__":
	fig_networks()
	a_on = fig_compressor()
	fig_method()
	print(f"Saved fig1_networks.png, fig2_compressor.png (alpha_on≈{a_on:.2f}), fig3_method.png to {OUT}/")
