"""Figure 3 (Nature-style) — method overview schematic.

A physics-faithful dynamic CMDP (a) solved by two *independent* methods on the same
environment: constraint-aware (Lagrangian) SAC (b) and MPC distillation (c). The teacher in
(c) is the DP/MPC feedback law built from the env model — it is NOT produced by the SAC.
Schematic/diagram archetype; Python/matplotlib backend (exclusive); vector + TIFF export.
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
})

ENV, POLICY, REWARD, DUAL, TEACH, DAGG = "#eef3fb", "#dfeafc", "#fff3d6", "#e7f6ec", "#fff3d6", "#fde9d6"


def main():
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, w, h, text, fc, fs=6.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.10",
                                    fc=fc, ec="black", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.05)

    def arrow(p1, p2, color="black", lw=1.1, double=False):
        style = "<|-|>" if double else "-|>"
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=9, lw=lw, color=color))

    def header(x, t):
        ax.text(x, 5.7, t, ha="center", fontsize=7.5, fontweight="bold")

    # ── (a) shared environment ───────────────────────────────────────────────
    header(2.0, "a  Dynamic CMDP environment")
    box(0.3, 3.7, 3.4, 0.85, "state $s_t$ (pressures, line-pack,\nprice / demand look-ahead)", ENV)
    box(0.3, 2.55, 3.4, 0.7, "policy $\\pi$  →  discharge ratio $α$", POLICY)
    box(0.3, 1.25, 3.4, 0.85, "feasibility realisation (Eq. 5)\n→ Weymouth flow + line-pack (Eq. 6)", ENV)
    box(4.05, 1.25, 1.65, 0.85, "$r^{econ}$ (Eq. 7)\n+ constraint\ncost $c_t$", REWARD)
    arrow((2.0, 3.7), (2.0, 3.25)); arrow((2.0, 2.55), (2.0, 2.1))
    arrow((3.7, 1.67), (4.05, 1.67))
    # feedback loop reward -> back to state
    arrow((4.88, 1.25), (4.88, 0.5)); arrow((4.88, 0.5), (2.0, 0.5)); arrow((2.0, 0.5), (2.0, 1.25))

    # ── (b) constrained SAC ──────────────────────────────────────────────────
    header(7.45, "b  Constrained (Lagrangian) SAC")
    box(6.0, 3.55, 2.9, 0.9, "SAC actor–critic\nmaximise  $r^{econ} - λ\\,c_t$  (Eq. 9)", POLICY)
    box(6.0, 2.05, 2.9, 0.85, "dual ascent on $λ$  (Eq. 10)\n(drives violations → 0)", DUAL)
    arrow((7.45, 3.55), (7.45, 2.9), color="#3aaa64", lw=1.1, double=True)
    ax.text(7.62, 3.22, "$λ$", fontsize=7, color="#3aaa64")
    # c_t from the env reward into the dual update
    arrow((5.7, 1.67), (5.7, 2.45)); arrow((5.7, 2.45), (6.0, 2.45))
    ax.text(5.45, 2.6, "$c_t$", fontsize=7)

    # ── (c) MPC distillation ─────────────────────────────────────────────────
    header(10.7, "c  MPC distillation")
    box(9.5, 3.55, 2.4, 0.9, "MPC / DP teacher\n(feedback law from\nthe env model)", TEACH)
    box(9.5, 2.25, 2.4, 0.8, "DAgger relabel\nvisited states", DAGG)
    box(9.5, 1.0, 2.4, 0.85, "millisecond MLP\nstudent policy", POLICY)
    arrow((10.7, 3.55), (10.7, 3.05)); arrow((10.7, 2.25), (10.7, 1.85))

    ax.text(6.0, 0.25, "(a) is the shared environment; (b) and (c) are two independent solution methods on it",
            ha="center", fontsize=6, style="italic", color="#555")

    fig.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig3_nature.{ext}"), bbox_inches="tight")
    fig.savefig(os.path.join(HERE, "fig3_nature.tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_nature.{svg,pdf,png,tiff} to", HERE)


if __name__ == "__main__":
    main()
