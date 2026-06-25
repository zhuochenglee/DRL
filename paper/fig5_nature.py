"""Figure 5 (Nature-style) — gun-barrel 24-h dispatch behaviour.

Core conclusion: Distilled-MPC reproduces the DP/MPC price-arbitrage dispatch at low cumulative
cost; Constrained-SAC keeps a larger pressure margin and so spends more. Four shared-x panels:
compressor action α, demand-node pressure, line-pack, cumulative cost.

This reproduces the seed-0 models (Constrained-SAC + Distilled-MPC) and the deterministic
DP/MPC planners, records the per-hour trajectory, exports it to raw_data/fig5_dispatch.csv,
and plots it. Python/matplotlib backend (exclusive).
"""
import csv
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
import sys
sys.path.insert(0, ROOT)

from envs.pipeline_env import EnvConfig, PaperInspiredDynamicLinepackEnv
from baselines import evaluate_schedule, evaluate_model, run_mpc
from DP import solve_dp
from constrained_sac import train_constrained_sac
from distill_mpc import distill

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
})

# palette consistent with Figure 4
STYLE = {
    "DP-oracle":       dict(color="#4d4d4d", ls="-",  lw=1.4),
    "MPC":             dict(color="#8c8c8c", ls=(0, (4, 2)), lw=1.4),
    "Distilled-MPC":   dict(color="#2c6fbb", ls="-",  lw=1.7),
    "Constrained-SAC": dict(color="#e8923b", ls="-",  lw=1.7),
}


def collect():
    cfg_eval = EnvConfig(noise_scale=0.0)
    cfg_train = EnvConfig(noise_scale=0.08)
    ev = PaperInspiredDynamicLinepackEnv(config=cfg_eval)
    d, p = ev.demand_series, ev.tou_price_series

    recs = {}
    recs["DP-oracle"] = evaluate_schedule(ev, solve_dp(ev, d, p, n_p=121, n_a=101), d, p)["records"]
    recs["MPC"] = run_mpc(ev, d, p, d, p, n_p=121, n_a=101)["records"]
    csac, _ = train_constrained_sac(cfg_train, total_timesteps=150000, seed=0,
                                    lam_init=50, dual_lr=1.5, lam_max=2000, learning_rate=3e-4)
    recs["Constrained-SAC"] = evaluate_model(ev, csac, d, p)["records"]
    student, _ = distill(cfg_eval, n_train=300, n_dagger=12, epochs=1000, seed=0)
    recs["Distilled-MPC"] = evaluate_model(ev, student, d, p)["records"]

    # export per-hour CSV
    with open(os.path.join(ROOT, "raw_data", "fig5_dispatch.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "hour", "alpha", "p2", "p3", "linepack", "purchase_cost", "cum_cost"])
        for m, rs in recs.items():
            cc = 0.0
            for r in rs:
                cc += r["purchase_cost"]
                w.writerow([m, r["hour"], round(r["alpha"], 4), round(r["p2"], 2), round(r["p3"], 2),
                            round(r["linepack"], 1), round(r["purchase_cost"], 4), round(cc, 4)])
    return ev, recs


def plot(ev, recs):
    order = ["DP-oracle", "MPC", "Distilled-MPC", "Constrained-SAC"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True)
    (a, b), (c, d) = axes

    for m in order:
        rs = recs[m]; hrs = [r["hour"] for r in rs]; st = STYLE[m]
        a.plot(hrs, [r["alpha"] for r in rs], **st)
        b.plot(hrs, [r["p3"] for r in rs], **st)
        c.plot(hrs, [r["linepack"] for r in rs], **st)
        cum = np.cumsum([r["purchase_cost"] for r in rs])
        d.plot(hrs, cum, **st, label=m)

    a.set_ylabel("compressor action  α"); a.set_title("a", loc="left", fontweight="bold", fontsize=9)
    b.axhspan(ev.p_min_safe, ev.p_max_safe, color="#3aaa64", alpha=0.07, lw=0)
    b.axhline(ev.p_min_safe, color="#c44e58", ls=(0, (3, 2)), lw=0.7)
    b.text(0.4, ev.p_min_safe + 1.2, "safe floor", fontsize=5.6, color="#c44e58")
    b.set_ylabel("demand-node pressure (a.u.)"); b.set_title("b", loc="left", fontweight="bold", fontsize=9)
    tgt = ev._linepack_target_end
    c.axhline(tgt, color="#444444", ls=(0, (4, 3)), lw=0.7)
    c.text(0.4, tgt + 60, "terminal target", fontsize=5.6, color="#444444")
    c.set_ylabel("line-pack (a.u.)"); c.set_xlabel("hour"); c.set_title("c", loc="left", fontweight="bold", fontsize=9)
    d.set_ylabel("cumulative cost (a.u.)"); d.set_xlabel("hour"); d.set_title("d", loc="left", fontweight="bold", fontsize=9)
    d.legend(loc="upper left", fontsize=6, handlelength=1.6, borderaxespad=0.3)
    for ax in (a, b, c, d):
        ax.set_xlim(0, 23); ax.tick_params(width=0.8, length=3)

    fig.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig5_nature.{ext}"), bbox_inches="tight")
    fig.savefig(os.path.join(HERE, "fig5_nature.tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ev, recs = collect()
    plot(ev, recs)
    print("Saved fig5_nature.{svg,pdf,png,tiff} + raw_data/fig5_dispatch.csv")
