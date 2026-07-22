#!/usr/bin/env python3
"""
Read scalar data from models/tensorboard/ and plot all algorithms
on a single unified comparison Figure.

Usage:
    python plot_comparison.py                   # scan default logdir
    python plot_comparison.py --logdir path/to/tb
    python plot_comparison.py --out comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_LOGDIR = "models/tensorboard"
DEFAULT_OUT = "models/tensorboard/comparison_figure.png"

# Map directory-name substring → (display label, tag prefix)
ALGO_PATTERNS: list[tuple[str, str, str]] = [
    ("dp",  "DP",  "dp/episode"),
    ("ga",  "GA",  "ga/generation"),
    ("ppo", "PPO", "ppo/episode"),
    ("sac", "SAC", "sac/episode"),
    ("td3", "TD3", "td3/episode"),
]

# Ordered metric keys and their y-axis labels
METRICS: list[tuple[str, str]] = [
    ("total_cost",      "Total cost (yuan)"),
    ("violation_hours", "Violation hours"),
    ("mean_alpha",      "Mean α"),
    ("mean_p2_bar",     "Mean p₂ (bar)"),
    ("mean_p3_bar",     "Mean p₃ (bar)"),
    ("mean_linepack",   "Mean linepack"),
]

# One colour per algorithm
ALGO_COLORS: dict[str, str] = {
    "DP":  "#2196F3",
    "GA":  "#FF9800",
    "PPO": "#4CAF50",
    "SAC": "#9C27B0",
    "TD3": "#F44336",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _match_algo(dirname: str) -> tuple[str, str] | None:
    """Return (label, tag_prefix) if dirname matches a known algorithm."""
    lower = dirname.lower()
    for substr, label, prefix in ALGO_PATTERNS:
        if substr in lower:
            return label, prefix
    return None


def _load_scalars(run_dir: Path, tag: str) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Return (steps, values) for *tag* inside *run_dir*, or None if not found.
    Loads only scalar summaries for speed.
    """
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    available = ea.Tags().get("scalars", [])
    if tag not in available:
        return None
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events], dtype=np.float64)
    values = np.array([e.value for e in events], dtype=np.float64)
    return steps, values


def _discover_runs(logdir: Path) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """
    Scan *logdir* (one level deep) and return:
        { algo_label: { metric_key: (steps, values) } }
    """
    data: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    for run_dir in sorted(logdir.iterdir()):
        if not run_dir.is_dir():
            continue
        match = _match_algo(run_dir.name)
        if match is None:
            continue
        label, prefix = match
        print(f"  [{label}] reading {run_dir.name} ...", end=" ")

        run_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for key, _ in METRICS:
            tag = f"{prefix}/{key}"
            result = _load_scalars(run_dir, tag)
            if result is not None:
                run_data[key] = result

        if run_data:
            # If multiple runs have the same label, keep the one with more data
            if label not in data or sum(len(v[0]) for v in run_data.values()) > sum(
                len(v[0]) for v in data[label].values()
            ):
                data[label] = run_data
            print(f"({len(run_data)} metrics loaded)")
        else:
            print("(no matching tags found)")

    return data


def _normalize_steps(steps: np.ndarray) -> np.ndarray:
    """Scale steps to [0, 1] for cross-algorithm overlay."""
    if steps.size == 0:
        return steps
    rng = steps[-1] - steps[0]
    return (steps - steps[0]) / rng if rng > 0 else np.zeros_like(steps)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _add_curve(ax: plt.Axes, label: str, steps: np.ndarray, values: np.ndarray, *, normalize_x: bool) -> None:
    color = ALGO_COLORS.get(label, "gray")
    x = _normalize_steps(steps) if normalize_x else steps
    ax.plot(x, values, color=color, label=label, linewidth=1.6, alpha=0.85)
    # for single-point algos (DP), also draw a horizontal reference line
    if len(x) == 1:
        ax.axhline(values[0], color=color, linestyle="--", linewidth=0.9, alpha=0.5)


def build_figure(
    data: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    normalize_x: bool = True,
) -> plt.Figure:
    n_metrics = len(METRICS)
    n_cols = 2
    n_rows = (n_metrics + 1) // n_cols   # 3 rows for 6 metrics

    fig = plt.figure(figsize=(14, 4 * n_rows + 1.5))
    gs = gridspec.GridSpec(n_rows + 1, n_cols, figure=fig, hspace=0.55, wspace=0.35,
                           height_ratios=[1] * n_rows + [0.30])
    fig.suptitle("Algorithm Comparison — TensorBoard Scalars", fontsize=13, fontweight="bold")

    axes: list[plt.Axes] = []
    for idx, (key, ylabel) in enumerate(METRICS):
        row, col = divmod(idx, n_cols)
        ax = fig.add_subplot(gs[row, col])
        ax.set_title(ylabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        if normalize_x:
            ax.set_xlabel("Training progress (0 → 1)", fontsize=7)
        else:
            ax.set_xlabel("Step / Generation", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

        has_data = False
        for label, run_data in sorted(data.items()):
            if key not in run_data:
                continue
            steps, values = run_data[key]
            _add_curve(ax, label, steps, values, normalize_x=normalize_x)
            has_data = True

        if not has_data:
            ax.text(0.5, 0.5, f"no data\n({key})", ha="center", va="center",
                    transform=ax.transAxes, color="gray", fontsize=8)
        axes.append(ax)

    # Shared legend in the bottom row spanning both columns
    legend_ax = fig.add_subplot(gs[n_rows, :])
    legend_ax.axis("off")
    handles, labels_used = [], set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels_used:
                handles.append(h)
                labels_used.add(l)
    if handles:
        legend_ax.legend(
            handles,
            list(labels_used),
            loc="center",
            ncol=min(len(handles), 5),
            fontsize=9,
            frameon=True,
        )

    return fig


# ── Summary table ─────────────────────────────────────────────────────────────

def _print_summary(data: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]) -> None:
    col_w = 14
    header = f"{'Algo':<8}" + "".join(f"{ylabel[:col_w]:>{col_w}}" for _, ylabel in METRICS)
    print("\n" + "=" * len(header))
    print("  Final-value summary (last recorded point per metric)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for label in sorted(data.keys()):
        row = f"{label:<8}"
        for key, _ in METRICS:
            if key in data[label]:
                v = data[label][key][1][-1]
                row += f"{v:>{col_w}.3f}"
            else:
                row += f"{'N/A':>{col_w}}"
        print(row)
    print("=" * len(header))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot TensorBoard runs for all algorithms.")
    parser.add_argument("--logdir", default=DEFAULT_LOGDIR,
                        help=f"TensorBoard log directory (default: {DEFAULT_LOGDIR})")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"Output PNG path (default: {DEFAULT_OUT})")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Keep original step values on x-axis (not normalized to 0-1)")
    args = parser.parse_args()

    logdir = Path(args.logdir)
    if not logdir.exists():
        print(f"Error: logdir not found: {logdir}")
        raise SystemExit(1)

    print(f"Scanning {logdir} ...")
    data = _discover_runs(logdir)

    if not data:
        print("No algorithm runs found. Check that the directory contains"
              " subfolders matching: dp, ga, ppo, sac, td3.")
        raise SystemExit(1)

    _print_summary(data)

    fig = build_figure(data, normalize_x=not args.no_normalize)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved → {out_path}")


if __name__ == "__main__":
    main()
