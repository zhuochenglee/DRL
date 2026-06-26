"""Remaining Nature-style figures (Figs 1, 6, 7, 8, 9, 10, 11, 12) from the saved CSVs.

Python/matplotlib backend (exclusive). Consistent palette with figs 2-5. Vector + TIFF export.
Run from repo root:  python paper/figs_nature_rest.py
"""
import csv
import collections
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrowPatch, Circle, Rectangle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "raw_data")

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8, "legend.frameon": False,
})
ND, NM, NL = "#4d4d4d", "#8c8c8c", "#c2c2c2"
BLUE, ORANGE, BAD, GREEN, GREY = "#2c6fbb", "#e8923b", "#c44e58", "#3aaa64", "#7f7f7f"
COLOR = {"DP-oracle": ND, "MPC": ND, "GA": NM, "Distilled-MPC": BLUE, "Constrained-SAC": ORANGE,
         "PPO": NL, "TD3": NL, "SAC": NL, "Rule": BAD}
ORDER = ["DP-oracle", "MPC", "Distilled-MPC", "GA", "PPO", "TD3", "SAC", "Constrained-SAC", "Rule"]


def summary(net):
    with open(os.path.join(RAW, net, "results_summary.csv")) as f:
        return {r["method"]: r for r in csv.DictReader(f)}


def latency_ms(net):
    t = collections.defaultdict(list)
    with open(os.path.join(RAW, net, "results_raw.csv")) as f:
        for r in csv.DictReader(f):
            if r["kind"] == "nominal":
                t[r["method"]].append(float(r["time"]))
    return {m: 1000 * np.mean(v) for m, v in t.items()}


def save(fig, stem, dpi=600):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{stem}.{ext}"), bbox_inches="tight")
    fig.savefig(os.path.join(HERE, f"{stem}.tiff"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ── cost / robustness bar charts (Figs 6, 8, 9, 10) ───────────────────────────
def cost_bar(net, stem, kind="nominal", ylim=None, title=""):
    S = summary(net)
    methods = [m for m in ORDER if m in S]
    cpre = "nominal" if kind == "nominal" else "robust"
    mean = [float(S[m][f"{cpre}_cost_mean"]) for m in methods]
    sd = [float(S[m][f"{cpre}_cost_std"]) for m in methods]
    viol = [float(S[m]["nominal_viol" if kind == "nominal" else "robust_viol_mean"]) for m in methods]
    dp = float(S["DP-oracle"][f"{cpre}_cost_mean"])
    fig, ax = plt.subplots(figsize=(4.6, 2.95))
    x = range(len(methods)); cols = [COLOR[m] for m in methods]
    bars = ax.bar(x, mean, 0.72, color=cols, edgecolor="black", lw=0.5, zorder=3)
    if "Rule" in methods:
        bars[methods.index("Rule")].set_hatch("////")
    ax.errorbar(x, mean, yerr=sd, fmt="none", ecolor="#222", elinewidth=0.8, capsize=2.2, capthick=0.8, zorder=4)
    ax.axhline(dp, color="#444", ls=(0, (4, 3)), lw=0.7)
    ax.text(1.6, dp + (ylim[1] if ylim else max(mean)) * 0.02, "DP optimum", fontsize=6, color="#444", ha="center", va="bottom")
    for xi, (m, v, mu, s) in enumerate(zip(methods, viol, mean, sd)):
        if v >= 0.5:
            ax.text(xi, mu + s + (ylim[1] if ylim else max(mean)) * 0.03, f"{v:.0f}h" if v >= 1 else f"{v:.1f}h",
                    ha="center", va="bottom", fontsize=5.6, color=BAD)
        if mu < 0.3:  # degenerate: never compresses -> 0 electricity cost but depletes line-pack
            ax.text(xi, 0.6, "never compresses (degenerate)", ha="center", va="bottom",
                    fontsize=5.2, color=GREY, rotation=90)
    ax.set_xticks(list(x)); ax.set_xticklabels(methods, rotation=35, ha="right")
    ax.set_ylabel(("robust " if kind != "nominal" else "") + "24-h electricity cost (a.u.)")
    if ylim:
        ax.set_ylim(*ylim)
    ax.tick_params(width=0.8, length=3)
    if title:
        ax.set_title(title, fontsize=7.5)
    save(fig, stem)


# ── ablation (Fig 7) ──────────────────────────────────────────────────────────
def ablation(stem):
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(os.path.join(RAW, "ablations.csv")) as f:
        for r in csv.DictReader(f):
            for k in ("cost", "viol", "tlp"):
                agg[r["variant"]][k].append(float(r[k]))
    order = ["full", "no_feasibility_realization", "no_soft_margin", "no_surge_penalty", "no_terminal_linepack"]
    labels = ["Full", "− feasibility\nrealisation", "− soft\nmargin", "− surge\npenalty", "− terminal\npenalty"]
    cost = [np.mean(agg[v]["cost"]) for v in order]; sd = [np.std(agg[v]["cost"]) for v in order]
    viol = [np.mean(agg[v]["viol"]) for v in order]; tlp = [np.mean(agg[v]["tlp"]) for v in order]
    cols = [NM, BAD, GREEN, NM, NM]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.9))
    b = a1.bar(labels, cost, yerr=sd, capsize=3, color=cols, edgecolor="black", lw=0.5)
    for bar, g in zip(b, tlp):
        a1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"gap {g:.0f}", ha="center", fontsize=5.6)
    a1.set_ylabel("24-h cost (a.u.)"); a1.set_title("a", loc="left", fontweight="bold", fontsize=9)
    a1.tick_params(width=0.8, length=3)
    b2 = a2.bar(labels, viol, color=cols, edgecolor="black", lw=0.5)
    for bar, v in zip(b2, viol):
        a2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08, f"{v:.1f}", ha="center", fontsize=6)
    a2.set_ylabel("violation-hours"); a2.set_title("b", loc="left", fontweight="bold", fontsize=9)
    a2.tick_params(width=0.8, length=3)
    save(fig, stem)


# ── cost vs latency Pareto (Fig 11) ──────────────────────────────────────────
def pareto(stem):
    S = summary("gunbarrel"); L = latency_ms("gunbarrel")
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    for m in ORDER:
        c = float(S[m]["nominal_cost_mean"]); lat = L[m]; v = float(S[m]["nominal_viol"])
        col = GREEN if v < 0.5 else BAD
        mk = "*" if m == "Distilled-MPC" else ("s" if m in ("DP-oracle", "MPC", "GA") else "o")
        ax.scatter(lat, c, s=200 if mk == "*" else 70, c=col, marker=mk, edgecolor="black", lw=0.5, zorder=3)
        yo = {"MPC": -1.0, "DP-oracle": 0.5, "Distilled-MPC": 0.6}.get(m, 0.5)
        xf = {"GA": 0.5}.get(m, 1.13)
        ax.annotate(m, (lat, c), xytext=(lat * xf, c + yo), fontsize=6)
    ax.set_xscale("log"); ax.set_xlabel("deployment latency per 24-h dispatch (ms, log)")
    ax.set_ylabel("nominal 24-h cost (a.u.)")
    ax.annotate("ideal", (3.0, 5.4), fontsize=7, color=BLUE, fontweight="bold")
    ax.tick_params(width=0.8, length=3)
    handles = [Patch(fc=GREEN, ec="black", lw=0.5, label="feasible (≈0 viol)"),
               Patch(fc=BAD, ec="black", lw=0.5, label="infeasible")]
    ax.legend(handles=handles, loc="upper right", fontsize=6)
    save(fig, stem)


# ── cross-network summary (Fig 12) ───────────────────────────────────────────
def cross_network(stem):
    nets = [("gunbarrel", "Gun-barrel"), ("branched", "Branched"), ("gaslib", "GasLib-40")]
    methods = ["GA", "MPC", "Distilled-MPC", "SAC", "Constrained-SAC"]
    data = {}
    for k, _ in nets:
        S = summary(k); dp = float(S["DP-oracle"]["nominal_cost_mean"])
        data[k] = {m: float(S[m]["nominal_cost_mean"]) / dp for m in methods if m in S}
    x = np.arange(len(methods)); w = 0.26
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    cols = [BLUE, ORANGE, GREEN]
    for i, (k, name) in enumerate(nets):
        ax.bar(x + (i - 1) * w, [data[k].get(m, np.nan) for m in methods], w, label=name, color=cols[i],
               edgecolor="black", lw=0.4)
    ax.axhline(1.0, color="black", ls=(0, (4, 3)), lw=0.8)
    ax.text(len(methods) - 0.55, 1.05, "DP optimum (×1)", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=12, ha="right")
    ax.set_ylabel("nominal cost  (× DP optimum)")
    ax.legend(title="network", fontsize=6, title_fontsize=6.5)
    ax.tick_params(width=0.8, length=3)
    save(fig, stem)


# ── networks schematic (Fig 1) ───────────────────────────────────────────────
def networks(stem):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2))
    for ax in axes:
        ax.set_xlim(-0.6, 4.9); ax.set_ylim(-1.5, 1.5); ax.axis("off")

    def node(ax, x, y, lab, kind):
        fc = {"src": "#cfe3f7", "dem": "#fde2e4"}[kind]
        ax.add_patch(Circle((x, y), 0.28, fc=fc, ec="black", lw=1.0, zorder=3))
        ax.text(x, y, lab, ha="center", va="center", fontsize=6.3, zorder=4)

    def pipe(ax, p1, p2, lab=None):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=8, lw=1.1, color=GREY,
                                     shrinkA=14, shrinkB=14, zorder=1))
        if lab:
            ax.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + 0.18, lab, ha="center", fontsize=5.4, color=GREY)

    def comp(ax, x, y):
        ax.add_patch(Rectangle((x - 0.14, y - 0.14), 0.28, 0.28, fc=GREEN, ec="black", lw=1.0, zorder=5))
        ax.text(x, y - 0.46, "α", ha="center", fontsize=6, color=GREEN)

    a, b, c = axes
    a.set_title("a  Gun-barrel", fontsize=7.5)
    node(a, 0, 0, "S", "src"); node(a, 2, 0, "2", "dem"); node(a, 4, 0, "3", "dem")
    pipe(a, (0, 0), (2, 0), "$p_1$"); pipe(a, (2, 0), (4, 0), "$p_2$"); comp(a, 0, 0)
    b.set_title("b  Branched", fontsize=7.5)
    node(b, 0, 0, "S", "src"); node(b, 1.8, 0, "1", "dem"); node(b, 3.8, 0.8, "2", "dem"); node(b, 3.8, -0.8, "3", "dem")
    pipe(b, (0, 0), (1.8, 0), "$p_0$"); pipe(b, (1.8, 0), (3.8, 0.8), "$p_1$"); pipe(b, (1.8, 0), (3.8, -0.8), "$p_2$"); comp(b, 0, 0)
    c.set_title("c  GasLib-40 chain", fontsize=7.5)
    node(c, 0, 0, "i6", "src"); node(c, 1.6, 0, "s13", "dem"); node(c, 3.0, 0, "s14", "dem"); node(c, 4.3, 0, "s10", "dem")
    pipe(c, (0, 0), (1.6, 0), "22km"); pipe(c, (1.6, 0), (3.0, 0), "7km"); pipe(c, (3.0, 0), (4.3, 0), "58km"); comp(c, 0, 0)
    fig.text(0.5, -0.02, "source = node with green compressor (α);  shaded = demand node", ha="center", fontsize=6, color="#555")
    save(fig, stem)


if __name__ == "__main__":
    networks("fig1_nature")
    cost_bar("gunbarrel", "fig6_nature", kind="robust", ylim=(0, 30), title="Gun-barrel — robustness (perturbed)")
    ablation("fig7_nature")
    cost_bar("branched", "fig8_nature", kind="nominal", ylim=(0, 30), title="Branched — nominal cost")
    cost_bar("branched", "fig9_nature", kind="robust", ylim=(0, 30), title="Branched — robustness (perturbed)")
    cost_bar("gaslib", "fig10_nature", kind="nominal", ylim=(0, 36), title="GasLib-40 — nominal cost")
    pareto("fig11_nature")
    cross_network("fig12_nature")
    print("Saved fig1,6,7,8,9,10,11,12 _nature.{svg,pdf,png,tiff} to", HERE)
