"""Figure 4 (Nature-style) — gun-barrel nominal 24-h cost by method.

Core conclusion: distilled-MPC matches the model-based optimum at zero violations;
from-scratch DRL pays a cost premium; the naive rule is expensive AND infeasible.
Single quantitative panel; SD error bars (n=5 seeds for DRL/GA; deterministic for
DP/MPC/Rule); violation-hours flagged; DP-optimum reference line.
Python/matplotlib backend (exclusive). Source data: raw_data/fig4_prism.csv.
"""
import csv
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",     # editable text in SVG
    "pdf.fonttype": 42,         # editable TrueType text in PDF
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

# restrained palette: one neutral family + two accents + one signal (red = infeasible)
NEUTRAL_DARK, NEUTRAL_MID, NEUTRAL_LIGHT = "#4d4d4d", "#8c8c8c", "#c2c2c2"
ACCENT_DISTILL, ACCENT_CSAC, SIGNAL_BAD = "#2c6fbb", "#e8923b", "#c44e58"
COLOR = {
    "DP-oracle": NEUTRAL_DARK, "MPC": NEUTRAL_DARK, "GA": NEUTRAL_MID,
    "Distilled-MPC": ACCENT_DISTILL, "Constrained-SAC": ACCENT_CSAC,
    "PPO": NEUTRAL_LIGHT, "TD3": NEUTRAL_LIGHT, "SAC": NEUTRAL_LIGHT,
    "Rule": SIGNAL_BAD,
}


def save_pub_py(fig, stem, dpi=600):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=dpi, bbox_inches="tight")


def main():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "raw_data", "fig4_prism.csv"))))
    methods = [r["method"] for r in rows]
    mean = [float(r["cost_mean"]) for r in rows]
    sd = [float(r["cost_SD"]) for r in rows]
    viol = [float(r["violation_hours"]) for r in rows]
    dp_opt = next(float(r["cost_mean"]) for r in rows if r["method"] == "DP-oracle")

    fig, ax = plt.subplots(figsize=(4.72, 2.95))  # ~120 x 75 mm (1.5-column)
    x = range(len(methods))
    colors = [COLOR[m] for m in methods]
    bars = ax.bar(x, mean, width=0.72, color=colors, edgecolor="black", linewidth=0.5, zorder=3)
    # the infeasible rule bar is hatched to flag it
    bars[methods.index("Rule")].set_hatch("////")
    ax.errorbar(x, mean, yerr=sd, fmt="none", ecolor="#222222", elinewidth=0.8,
                capsize=2.2, capthick=0.8, zorder=4)

    # DP optimum reference line (label placed over clear space above the short reference bars)
    ax.axhline(dp_opt, color="#444444", ls=(0, (4, 3)), lw=0.7, zorder=1)
    ax.text(1.6, dp_opt + 0.7, "DP optimum", fontsize=6, color="#444444", ha="center", va="bottom")

    # flag violation-hours where non-zero (otherwise feasible)
    for xi, (m, v, mu, s) in enumerate(zip(methods, viol, mean, sd)):
        if v >= 0.5:
            ax.text(xi, mu + s + 1.0, f"{v:.0f} h\nviol.", ha="center", va="bottom",
                    fontsize=5.6, color=SIGNAL_BAD, linespacing=0.9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(methods, rotation=35, ha="right")
    ax.set_ylabel("24-h electricity cost (a.u.)")
    ax.set_ylim(0, 32)
    ax.tick_params(width=0.8, length=3)
    ax.margins(x=0.02)

    # minimal legend: what the two accents and the red flag mean
    handles = [
        Patch(fc=ACCENT_DISTILL, ec="black", lw=0.5, label="Distilled-MPC (this work)"),
        Patch(fc=ACCENT_CSAC, ec="black", lw=0.5, label="Constrained-SAC (this work)"),
        Patch(fc=SIGNAL_BAD, ec="black", lw=0.5, hatch="////", label="infeasible (violations)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=5.8, handlelength=1.1,
              handleheight=1.0, borderaxespad=0.3)

    fig.tight_layout()
    save_pub_py(fig, os.path.join(HERE, "fig4_nature"))
    plt.close(fig)
    print("Saved fig4_nature.{svg,pdf,png,tiff} to", HERE)


if __name__ == "__main__":
    main()
