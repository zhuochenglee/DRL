"""Nature-style Figure 11: cost-latency trade-off for gun-barrel dispatch.

Python/matplotlib backend. Reads saved experiment CSVs only; no retraining.
Outputs editable SVG/PDF plus PNG/TIFF previews.
"""
from __future__ import annotations

import csv
import collections
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "raw_data", "gunbarrel")

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

BLUE = "#2c6fbb"
ORANGE = "#d9822b"
GREY = "#666666"
LIGHT_GREY = "#cfcfcf"
RED = "#c44e58"
GREEN = "#3aaa64"
GOLD = "#c9a227"
BLACK = "#222222"

METHOD_ORDER = [
    "Distilled-MPC", "PPO", "Constrained-SAC", "SAC", "TD3", "Rule",
    "DP-oracle", "MPC", "GA",
]

STYLE = {
    "Distilled-MPC": dict(color=BLUE, marker="*", size=260, z=6),
    "PPO": dict(color=ORANGE, marker="o", size=52, z=4),
    "Constrained-SAC": dict(color=ORANGE, marker="o", size=52, z=4),
    "SAC": dict(color=ORANGE, marker="o", size=52, z=4),
    "TD3": dict(color=ORANGE, marker="o", size=52, z=4),
    "Rule": dict(color=RED, marker="X", size=80, z=5),
    "DP-oracle": dict(color=GREY, marker="D", size=56, z=4),
    "MPC": dict(color=GREY, marker="D", size=56, z=4),
    "GA": dict(color=GREY, marker="s", size=62, z=4),
}

LABELS = {
    "Distilled-MPC": "Distilled-MPC\n(this work)",
    "PPO": "PPO",
    "Constrained-SAC": "C-SAC",
    "SAC": "SAC",
    "TD3": "TD3",
    "Rule": "Rule\n19 viol-h",
    "DP-oracle": "DP/MPC",
    "GA": "GA",
}

# Manual label positions in data coordinates. MPC is deliberately not labelled;
# it is coincident with DP-oracle and explained by the DP/MPC label.
LABEL_POS = {
    "Distilled-MPC": (3.1, 3.25, "left"),
    "PPO": (5.8, 8.5, "left"),
    "Constrained-SAC": (5.8, 13.2, "left"),
    "SAC": (5.8, 16.5, "left"),
    "TD3": (5.8, 21.5, "left"),
    "Rule": (2.05, 24.6, "left"),
    "DP-oracle": (430, 4.35, "center"),
    "GA": (11500, 7.85, "center"),
}


def read_summary() -> dict[str, dict[str, str]]:
    with open(os.path.join(RAW, "results_summary.csv"), newline="") as f:
        return {r["method"]: r for r in csv.DictReader(f)}


def read_latency_ms() -> dict[str, float]:
    times = collections.defaultdict(list)
    with open(os.path.join(RAW, "results_raw.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r["kind"] == "nominal":
                times[r["method"]].append(1000.0 * float(r["time"]))
    return {m: float(np.mean(v)) for m, v in times.items()}


def save(fig: plt.Figure, stem: str = "fig11_nature", dpi: int = 600) -> None:
    for ext in ("svg", "pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{stem}.{ext}"), bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(HERE, f"{stem}.tiff"), dpi=dpi, bbox_inches="tight", facecolor="white")


def main() -> None:
    summary = read_summary()
    latency = read_latency_ms()

    dp_cost = float(summary["DP-oracle"]["nominal_cost_mean"])
    mpc_latency = 0.5 * (latency["DP-oracle"] + latency["MPC"])
    speedup = mpc_latency / latency["Distilled-MPC"]

    fig, ax = plt.subplots(figsize=(6.25, 3.55))
    ax.set_xscale("log")
    ax.set_xlim(1.7, 22000)
    ax.set_ylim(0, 26)

    # Reading aids: deployment region and near-optimal cost band.
    ax.axvspan(1.7, 10, color=BLUE, alpha=0.055, lw=0, zorder=0)
    ax.axvline(10, color=BLUE, lw=0.75, ls=(0, (3, 3)), alpha=0.75, zorder=1)
    ax.text(9.25, 25.0, "10 ms real-time\nbudget", color=BLUE, fontsize=6,
            ha="right", va="top", fontweight="bold")

    ax.axhspan(dp_cost - 0.35, dp_cost + 0.85, color=GREEN, alpha=0.10, lw=0, zorder=0)
    ax.axhline(dp_cost, color="#555555", lw=0.75, ls=(0, (4, 3)), zorder=1)
    ax.text(4200, dp_cost + 0.45, "DP optimum", color="#555555", fontsize=6,
            ha="left", va="bottom", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))

    # Family labels tell the story without a bulky legend.
    ax.text(6200, 1.05, "seconds-level planning", color=GREY, fontsize=6.2,
            ha="center", va="bottom", fontweight="bold")
    ax.text(2.0, 18.3, "from-scratch DRL:\nfast but costly", color=ORANGE,
            fontsize=6.4, ha="left", va="center", fontweight="bold")

    # Plot points with light uncertainty for stochastic methods.
    for method in METHOD_ORDER:
        if method not in summary or method not in latency:
            continue
        x = latency[method]
        y = float(summary[method]["nominal_cost_mean"])
        sd = float(summary[method]["nominal_cost_std"])
        style = STYLE[method]
        if sd > 0:
            ax.errorbar(x, y, yerr=sd, color=style["color"], alpha=0.35,
                        lw=0.75, capsize=2.0, capthick=0.75, zorder=style["z"] - 1)
        ax.scatter(x, y, s=style["size"], c=style["color"], marker=style["marker"],
                   edgecolor=BLACK, linewidth=0.65 if style["marker"] != "*" else 0.85,
                   zorder=style["z"])

    # Leader-line labels. Skip MPC because it overlaps DP-oracle by design.
    for method, (tx, ty, ha) in LABEL_POS.items():
        x = latency[method]
        y = float(summary[method]["nominal_cost_mean"])
        ax.annotate(
            LABELS[method], xy=(x, y), xytext=(tx, ty), ha=ha, va="center",
            fontsize=6.2 if method != "Distilled-MPC" else 6.6,
            fontweight="bold" if method == "Distilled-MPC" else "normal",
            color=BLACK,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.78),
            arrowprops=dict(arrowstyle="-", color=LIGHT_GREY, lw=0.65,
                            shrinkA=1.0, shrinkB=3.0),
            zorder=8,
        )

    # Explicit speedup comparison: same cost class, much lower latency.
    ax.annotate("", xy=(latency["Distilled-MPC"], 6.75), xytext=(mpc_latency, 6.75),
                arrowprops=dict(arrowstyle="<|-", color=GOLD, lw=1.1, mutation_scale=8),
                zorder=3)
    ax.text(65, 7.25, f"~{speedup:.0f}x lower latency\nthan DP/MPC",
            color="#8a6d15", fontsize=6.4, ha="center", va="bottom", fontweight="bold")

    # Direction-of-improvement cue.
    ax.annotate("better", xy=(2.2, 3.2), xytext=(15, 11.5),
                color="#555555", fontsize=6.2, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.75, color="#777777"))

    ax.set_xlabel("decision latency per dispatch (ms, log scale)")
    ax.set_ylabel("nominal 24-h electricity cost (a.u.)")
    ax.set_title("Fig. 11 | Cost-latency Pareto map",
                 loc="left", fontsize=7.8, fontweight="bold", pad=6)

    # Keep grid subtle; Nature-style figures should not feel like dashboards.
    ax.grid(axis="y", color="#eeeeee", lw=0.55, zorder=0)
    ax.set_axisbelow(True)
    save(fig)


if __name__ == "__main__":
    main()
