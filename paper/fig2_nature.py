"""Figure 2 (Nature-style) — compressor characteristic map + disconnected feasible action set.

Core conclusion: the compressor's feasible operating set is *disconnected* (off, or on above a
state-dependent minimum speed); a command in the sub-minimum-speed dead band is realized as
unit-off. (a) efficiency characteristic eta(phi) with the feasible surge/choke band; (b) the
realized discharge ratio vs the command, showing the disconnected feasible set {1} U [alpha_on, 2].
Python/matplotlib backend (exclusive). Source data: raw_data/fig2_eta_curve.csv,
fig2_alpha_realization.csv, fig2_constants.csv (computed from envs/compressor.py at p2=100).
"""
import csv
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "raw_data")

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

LINE, FEAS, BAD, GREY = "#2c6fbb", "#3aaa64", "#c44e58", "#7f7f7f"


def _read(name):
    return list(csv.DictReader(open(os.path.join(RAW, name))))


def save_pub_py(fig, stem, dpi=600):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=dpi, bbox_inches="tight")


def main():
    const = {r["quantity"]: r["value"] for r in _read("fig2_constants.csv")}
    phi_lo, phi_hi = float(const["phi_lower"]), float(const["phi_upper"])
    alpha_on = float(const["alpha_on"])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    # ── (a) characteristic map eta(phi) ──────────────────────────────────────
    eta_rows = _read("fig2_eta_curve.csv")
    phi = [float(r["phi"]) for r in eta_rows]
    eta = [float(r["eta"]) for r in eta_rows]
    axL.axvspan(min(phi), phi_lo, color=BAD, alpha=0.10, lw=0)
    axL.axvspan(phi_hi, max(phi), color=BAD, alpha=0.10, lw=0)
    axL.axvspan(phi_lo, phi_hi, color=FEAS, alpha=0.10, lw=0)
    axL.plot(phi, eta, color=LINE, lw=1.6, zorder=3)
    for b in (phi_lo, phi_hi):
        axL.axvline(b, color=FEAS, ls=(0, (3, 2)), lw=0.8)
    axL.text((phi_lo + phi_hi) / 2, 0.555, "feasible", ha="center", color=FEAS, fontsize=6.5)
    axL.text(0.6, 0.9, "surge", color=BAD, fontsize=6.5)
    axL.text(1.5, 0.9, "choke", color=BAD, fontsize=6.5)
    axL.set_xlabel("normalised inlet flow  φ = Q$_{in}$/ω")
    axL.set_ylabel("compressor efficiency  η")
    axL.set_xlim(min(phi), max(phi)); axL.set_ylim(0.5, 0.95)
    axL.tick_params(width=0.8, length=3)
    axL.set_title("a", loc="left", fontweight="bold", fontsize=9)

    # ── (b) realized vs commanded alpha (the dead-band snap) ──────────────────
    rz_rows = _read("fig2_alpha_realization.csv")
    cmd = [float(r["commanded_alpha"]) for r in rz_rows]
    rz = [float(r["realized_alpha"]) for r in rz_rows]
    off_x = [a for a, _ in zip(cmd, rz) if a < alpha_on]
    off_y = [r for a, r in zip(cmd, rz) if a < alpha_on]
    on_x = [a for a, _ in zip(cmd, rz) if a >= alpha_on]
    on_y = [r for a, r in zip(cmd, rz) if a >= alpha_on]
    axR.axvspan(1.0, alpha_on, color=BAD, alpha=0.10, lw=0)
    axR.axvspan(alpha_on, 2.0, color=FEAS, alpha=0.08, lw=0)
    axR.plot([1, 2], [1, 2], color=GREY, ls=(0, (1, 2)), lw=0.9, zorder=2)
    axR.plot(off_x, off_y, color=LINE, lw=1.8, zorder=3)
    axR.plot(on_x, on_y, color=LINE, lw=1.8, zorder=3)
    axR.axvline(alpha_on, color="#444444", ls=(0, (4, 3)), lw=0.7)
    # direct labels instead of a legend box
    axR.text(1.66, 1.86, "realised  $\\tilde{α}$ (Eq. 5)", color=LINE, fontsize=6.3, rotation=27)
    axR.text(1.42, 1.30, "identity ($\\tilde{α}=α$)", color=GREY, fontsize=6, rotation=27)
    axR.text(alpha_on + 0.02, 1.01, f"α$_{{on}}$≈{alpha_on:.2f}", fontsize=6.3, color="#444444", ha="left")
    axR.text((1.0 + alpha_on) / 2, 1.55, "dead band\n→ OFF", ha="center", color=BAD, fontsize=6.5, linespacing=0.95)
    axR.text((alpha_on + 2.0) / 2, 1.10, "feasible ON", ha="center", color=FEAS, fontsize=6.5)
    axR.set_xlabel("commanded discharge ratio  α")
    axR.set_ylabel("realised discharge ratio  $\\tilde{α}$")
    axR.set_xlim(1.0, 2.0); axR.set_ylim(0.97, 2.0)
    axR.tick_params(width=0.8, length=3)
    axR.set_title("b", loc="left", fontweight="bold", fontsize=9)

    fig.tight_layout()
    save_pub_py(fig, os.path.join(HERE, "fig2_nature"))
    plt.close(fig)
    print(f"Saved fig2_nature.{{svg,pdf,png,tiff}} (alpha_on={alpha_on}) to", HERE)


if __name__ == "__main__":
    main()
