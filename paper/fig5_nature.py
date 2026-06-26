"""Figure 5 (Nature-style, v2) — gun-barrel 24-h dispatch, made intuitive.

Reads the saved per-hour trajectory (raw_data/fig5_dispatch.csv) — no retraining.
Design: a hero panel (cumulative cost) over a TOU-price background, plus two driver panels
(compressor action, line-pack) that explain WHY: methods compress in cheap off-peak hours and
coast on stored line-pack through the expensive peak. Distilled-MPC tracks the DP/MPC optimum;
Constrained-SAC over-compresses and spends more.
Python/matplotlib backend (exclusive).
"""
import csv
import collections
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from envs.pipeline_env import EnvConfig, PaperInspiredDynamicLinepackEnv

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8, "legend.frameon": False,
})
GREY, BLUE, ORANGE = "#5a5a5a", "#2c6fbb", "#e8923b"
# DP and MPC coincide on the nominal day -> draw both grey (reads as one "optimum" line)
SERIES = [("DP-oracle", GREY, "-", 1.3), ("MPC", GREY, "-", 1.3),
          ("Distilled-MPC", BLUE, "-", 1.8), ("Constrained-SAC", ORANGE, "-", 1.8)]


def load():
    d = collections.defaultdict(list)
    with open(os.path.join(ROOT, "raw_data", "fig5_dispatch.csv")) as f:
        for r in csv.DictReader(f):
            d[r["method"]].append(r)
    for m in d:
        d[m].sort(key=lambda r: int(r["hour"]))
    return d


def peak_spans(price):
    """contiguous hour intervals where the TOU price is above its minimum (peak)."""
    pk = price > price.min() + 1e-9
    spans, s = [], None
    for h, on in enumerate(pk):
        if on and s is None:
            s = h
        if (not on or h == len(pk) - 1) and s is not None:
            end = h if not on else h + 1
            spans.append((s - 0.5, end - 0.5)); s = None
    return spans


def shade_peak(ax, spans, label=False):
    for i, (a, b) in enumerate(spans):
        ax.axvspan(a, b, color="#c9a227", alpha=0.13, lw=0, zorder=0)
    if label and spans:
        a, b = spans[0]
        ax.text((a + b) / 2, ax.get_ylim()[1] * 0.93, "peak price", ha="center",
                fontsize=5.8, color="#9c7d12")


def main():
    data = load()
    env = PaperInspiredDynamicLinepackEnv(config=EnvConfig())
    price = np.asarray(env.tou_price_series, dtype=float)
    spans = peak_spans(price)
    tgt = env._linepack_target_end

    fig = plt.figure(figsize=(7.0, 4.5))
    gs = GridSpec(2, 2, height_ratios=[1.25, 1.0], hspace=0.5, wspace=0.28,
                  left=0.08, right=0.98, top=0.93, bottom=0.12)
    ax_cost = fig.add_subplot(gs[0, :])
    ax_a = fig.add_subplot(gs[1, 0])
    ax_lp = fig.add_subplot(gs[1, 1])

    # ── hero: cumulative cost ────────────────────────────────────────────────
    for m, col, ls, lw in SERIES:
        rs = data[m]; hrs = [int(r["hour"]) for r in rs]
        ax_cost.plot(hrs, [float(r["cum_cost"]) for r in rs], color=col, ls=ls, lw=lw, zorder=3)
    ax_cost.set_xlim(0, 23); ax_cost.set_ylim(0, None)
    shade_peak(ax_cost, spans, label=True)
    # end-of-day value labels
    for m, col in [("Constrained-SAC", ORANGE), ("Distilled-MPC", BLUE), ("DP-oracle", GREY)]:
        v = float(data[m][-1]["cum_cost"])
        ax_cost.annotate(f"{v:.1f}", (23, v), xytext=(23.3, v), color=col, fontsize=6.5, va="center")
    ax_cost.set_ylabel("cumulative electricity cost (a.u.)")
    ax_cost.set_xlabel("hour")
    ax_cost.set_title("a   Distilled-MPC tracks the DP/MPC optimum;  Constrained-SAC spends ~3× more",
                      loc="left", fontweight="bold", fontsize=8)
    # legend (group DP/MPC)
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], color=GREY, lw=1.3, label="DP / MPC (optimum)"),
           Line2D([0], [0], color=BLUE, lw=1.8, label="Distilled-MPC"),
           Line2D([0], [0], color=ORANGE, lw=1.8, label="Constrained-SAC")]
    ax_cost.legend(handles=leg, loc="upper left", fontsize=6.3, handlelength=1.6)

    # ── driver 1: compressor action ──────────────────────────────────────────
    for m, col, ls, lw in SERIES:
        rs = data[m]
        ax_a.step([int(r["hour"]) for r in rs], [float(r["alpha"]) for r in rs], where="mid",
                  color=col, ls=ls, lw=lw, zorder=3)
    shade_peak(ax_a, spans)
    ax_a.set_xlim(0, 23); ax_a.set_ylim(0.95, 1.7)
    ax_a.set_ylabel("compressor action  α"); ax_a.set_xlabel("hour")
    ax_a.set_title("b   compress off-peak", loc="left", fontweight="bold", fontsize=8)

    # ── driver 2: line-pack ──────────────────────────────────────────────────
    for m, col, ls, lw in SERIES:
        rs = data[m]
        ax_lp.plot([int(r["hour"]) for r in rs], [float(r["linepack"]) for r in rs],
                   color=col, ls=ls, lw=lw, zorder=3)
    ax_lp.axhline(tgt, color="#444", ls=(0, (4, 3)), lw=0.7)
    ax_lp.text(0.4, tgt + 70, "terminal target", fontsize=5.6, color="#444")
    shade_peak(ax_lp, spans)
    ax_lp.set_xlim(0, 23)
    ax_lp.set_ylabel("line-pack (a.u.)"); ax_lp.set_xlabel("hour")
    ax_lp.set_title("c   store, then coast", loc="left", fontweight="bold", fontsize=8)

    for ax in (ax_cost, ax_a, ax_lp):
        ax.tick_params(width=0.8, length=3)

    for ext in ("svg", "pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig5_nature.{ext}"), bbox_inches="tight")
    fig.savefig(os.path.join(HERE, "fig5_nature.tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig5_nature.{svg,pdf,png,tiff} (re-plotted from raw_data/fig5_dispatch.csv)")


if __name__ == "__main__":
    main()
