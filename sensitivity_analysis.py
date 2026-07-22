#!/usr/bin/env python3
"""
Linepack Sensitivity Analysis for SAC
======================================
Sweeps one of two parameters while keeping everything else at the default config,
trains a fresh SAC agent for each value, evaluates the final 24-h dispatch, and
records the total electricity cost and violation hours.

Two sweep modes (--mode):
  w_penalty   : vary w_terminal_linepack   (reward shaping weight)
  target      : vary the absolute linepack target (kg equiv.)

Results are written to models/exports/sensitivity_<mode>.csv and a summary
figure is saved to models/exports/sensitivity_<mode>.png.

Usage examples
--------------
  python sensitivity_analysis.py                      # default: w_penalty sweep
  python sensitivity_analysis.py --mode target
  python sensitivity_analysis.py --timesteps 80000    # increase training budget
  python sensitivity_analysis.py --mode w_penalty --values 0 50 100 200 250 400 600
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from envs.pipeline_env import EnvConfig, PaperInspiredDynamicLinepackEnv


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_TIMESTEPS = 60_000

# Default sweep grids
DEFAULT_W_PENALTY_GRID = [0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0]
DEFAULT_TARGET_GRID    = [7000.0, 7500.0, 8000.0, 8500.0, 9000.0, 9500.0, 10000.0, 10500.0, 11000.0]

TB_BASE = "models/tensorboard"
EXPORT_DIR = Path("models/exports")

BASE_CONFIG_KWARGS = dict(
    noise_scale=0.0,
    horizon=24,
    w_pressure=1000.0,
    w_flow=5.0,
    w_terminal_linepack=250.0,   # overridden in w_penalty sweep
    linepack_target=None,        # overridden in target sweep
)


# ── Minimal progress callback ─────────────────────────────────────────────────

class _SilentProgressBar(BaseCallback):
    """Prints a compact one-line progress bar; no per-step noise."""

    def __init__(self, total_timesteps: int):
        super().__init__()
        self._total = total_timesteps
        self._interval = max(1, total_timesteps // 20)
        self._last = -1
        self._t0: float | None = None

    def _on_training_start(self):
        self._t0 = time.time()

    def _on_step(self):
        n = self.num_timesteps
        if n >= self._total or n - self._last >= self._interval:
            self._last = n
            pct = min(n / self._total, 1.0)
            bar = "#" * int(20 * pct) + "-" * (20 - int(20 * pct))
            elapsed = 0.0 if self._t0 is None else time.time() - self._t0
            print(f"\r    [{bar}] {pct:5.1%}  {elapsed:5.1f}s", end="", flush=True)
        return True

    def _on_training_end(self):
        print()   # newline after bar


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate(env: PaperInspiredDynamicLinepackEnv, model: SAC) -> dict:
    """Run one deterministic 24-h episode; return result dict."""
    obs, _ = env.reset()
    total_cost = 0.0
    total_power = 0.0
    violation_hours = 0
    total_linepack_gap = 0.0
    alphas = []

    for _ in range(env.horizon):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        total_cost  += info["purchase_cost"]
        total_power += info["power_kwh"]
        alphas.append(info["alpha"])
        if info["violation"]:
            violation_hours += 1
        if terminated or truncated:
            total_linepack_gap = info.get("terminal_linepack_gap", 0.0)
            break

    return {
        "total_cost":         round(total_cost, 4),
        "total_power_kwh":    round(total_power, 4),
        "violation_hours":    violation_hours,
        "mean_alpha":         round(float(np.mean(alphas)), 4),
        "terminal_linepack_gap": round(total_linepack_gap, 4),
    }


# ── Single training + eval run ────────────────────────────────────────────────

def train_and_eval(config: EnvConfig, total_timesteps: int, tb_tag: str) -> dict:
    """Train a fresh SAC model and return evaluation metrics."""
    env      = PaperInspiredDynamicLinepackEnv(config=config)
    eval_env = PaperInspiredDynamicLinepackEnv(config=config)

    model = SAC(
        "MlpPolicy", env,
        verbose=0,
        learning_rate=1e-3,
        gamma=0.995,
        tensorboard_log=TB_BASE,
    )
    cb = _SilentProgressBar(total_timesteps)
    model.learn(
        total_timesteps=total_timesteps,
        callback=cb,
        tb_log_name=f"sensitivity_{tb_tag}",
    )
    return evaluate(eval_env, model)


# ── Sweep drivers ─────────────────────────────────────────────────────────────

def sweep_w_penalty(values: list[float], total_timesteps: int) -> list[dict]:
    rows = []
    for w in values:
        cfg = EnvConfig(**{**BASE_CONFIG_KWARGS, "w_terminal_linepack": w})
        print(f"\n  w_terminal_linepack = {w:>7.1f}  ({total_timesteps} steps)")
        t0 = time.time()
        result = train_and_eval(cfg, total_timesteps, tb_tag=f"w{int(w)}")
        result["w_terminal_linepack"] = w
        result["train_time_s"] = round(time.time() - t0, 1)
        rows.append(result)
        _print_row(result, param_name="w_terminal_linepack", param_val=w)
    return rows


def sweep_target(values: list[float], total_timesteps: int) -> list[dict]:
    rows = []
    for tgt in values:
        cfg = EnvConfig(**{**BASE_CONFIG_KWARGS, "linepack_target": tgt})
        print(f"\n  linepack_target = {tgt:>8.0f}  ({total_timesteps} steps)")
        t0 = time.time()
        result = train_and_eval(cfg, total_timesteps, tb_tag=f"lp{int(tgt)}")
        result["linepack_target"] = tgt
        result["train_time_s"] = round(time.time() - t0, 1)
        rows.append(result)
        _print_row(result, param_name="linepack_target", param_val=tgt)
    return rows


def _print_row(row: dict, param_name: str, param_val: float):
    print(
        f"    → cost={row['total_cost']:>8.3f}  violations={row['violation_hours']:>2d}"
        f"  lp_gap={row['terminal_linepack_gap']:>8.2f}  "
        f"time={row['train_time_s']:>6.1f}s"
    )


# ── CSV writer ────────────────────────────────────────────────────────────────

def save_csv(rows: list[dict], out_path: Path):
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV → {out_path}")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_sensitivity(rows: list[dict], param_name: str, out_path: Path):
    xs = [r[param_name] for r in rows]
    costs  = [r["total_cost"]      for r in rows]
    viols  = [r["violation_hours"] for r in rows]
    gaps   = [r["terminal_linepack_gap"] for r in rows]
    alphas = [r["mean_alpha"]      for r in rows]

    fig = plt.figure(figsize=(12, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle(
        f"SAC Sensitivity Analysis — {param_name}",
        fontsize=13, fontweight="bold",
    )

    MARKER = "o-"
    COLOR  = "#1976D2"

    def _subplot(pos, ys, ylabel, title, color=COLOR):
        ax = fig.add_subplot(pos)
        ax.plot(xs, ys, MARKER, color=color, linewidth=1.8, markersize=6)
        ax.set_xlabel(param_name, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.35)
        ax.tick_params(labelsize=8)
        # Annotate min cost / min violations
        best_idx = int(np.argmin(ys))
        ax.annotate(
            f"best: {ys[best_idx]:.3g}",
            xy=(xs[best_idx], ys[best_idx]),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            arrowprops=dict(arrowstyle="->", lw=0.8),
        )

    _subplot(gs[0, 0], costs,  "yuan",      "Total electricity cost",   "#1976D2")
    _subplot(gs[0, 1], viols,  "hours",     "Pressure violation hours", "#D32F2F")
    _subplot(gs[1, 0], gaps,   "kg equiv.", "Terminal linepack gap",    "#7B1FA2")
    _subplot(gs[1, 1], alphas, "α",          "Mean compressor α",       "#388E3C")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Linepack sensitivity analysis for SAC.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["w_penalty", "target"], default="w_penalty",
        help="Which parameter to sweep (default: w_penalty)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=DEFAULT_TIMESTEPS,
        help=f"SAC training steps per configuration (default: {DEFAULT_TIMESTEPS})",
    )
    parser.add_argument(
        "--values", type=float, nargs="+",
        help="Override the default sweep grid with explicit values",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Sensitivity mode : {args.mode}")
    print(f"  Training steps   : {args.timesteps}")

    if args.mode == "w_penalty":
        grid = args.values if args.values else DEFAULT_W_PENALTY_GRID
        print(f"  Sweep grid       : {grid}")
        print("=" * 60)
        rows = sweep_w_penalty(grid, args.timesteps)
        param_name = "w_terminal_linepack"
    else:
        grid = args.values if args.values else DEFAULT_TARGET_GRID
        print(f"  Sweep grid       : {grid}")
        print("=" * 60)
        rows = sweep_target(grid, args.timesteps)
        param_name = "linepack_target"

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"  {'Param':>12}  {'Cost':>10}  {'Violations':>10}  {'LP gap':>10}  {'α':>8}")
    print("  " + "-" * 56)
    for r in rows:
        print(
            f"  {r[param_name]:>12.1f}  {r['total_cost']:>10.3f}  "
            f"{r['violation_hours']:>10d}  {r['terminal_linepack_gap']:>10.2f}  "
            f"{r['mean_alpha']:>8.4f}"
        )
    print("=" * 60)

    tag = args.mode
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    save_csv(rows, EXPORT_DIR / f"sensitivity_{tag}.csv")
    plot_sensitivity(rows, param_name, EXPORT_DIR / f"sensitivity_{tag}.png")


if __name__ == "__main__":
    main()
