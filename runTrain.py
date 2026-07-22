#!/usr/bin/env python3
"""Unified runner for GA, DP, TD3, SAC, PPO training scripts."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_MAP = {
	"ga": "GA.py",
	"dp": "DP.py",
	"td3": "TD3.py",
	"sac": "papercode.py",
	"ppo": "PPO.py",
}
DEFAULT_ORDER = ["ga", "dp", "td3", "sac", "ppo"]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run one or more training scripts (GA, DP, TD3, SAC, PPO).",
	)
	parser.add_argument(
		"algorithms",
		nargs="*",
		help="Algorithms to run: ga dp td3 sac ppo. If omitted, run all.",
	)
	parser.add_argument(
		"--continue-on-error",
		action="store_true",
		help="Continue remaining algorithms even if one fails.",
	)
	parser.add_argument(
		"--list",
		action="store_true",
		help="List supported algorithms and exit.",
	)
	return parser.parse_args()


def resolve_algorithms(requested: list[str]) -> list[str]:
	if not requested:
		return DEFAULT_ORDER

	resolved = []
	unknown = []
	for name in requested:
		key = name.strip().lower()
		if key in SCRIPT_MAP:
			resolved.append(key)
		else:
			unknown.append(name)

	if unknown:
		valid = ", ".join(DEFAULT_ORDER)
		raise ValueError(f"Unknown algorithm(s): {', '.join(unknown)}. Valid options: {valid}")

	return resolved


def run_script(project_dir: Path, algo: str) -> tuple[int, float]:
	script_name = SCRIPT_MAP[algo]
	script_path = project_dir / script_name
	if not script_path.exists():
		raise FileNotFoundError(f"Script not found: {script_path}")

	print(f"\n=== Running {algo.upper()} via {script_name} ===")
	start = time.time()
	result = subprocess.run(
		[sys.executable, str(script_path)],
		cwd=str(project_dir),
	)
	elapsed = time.time() - start
	return result.returncode, elapsed


def main() -> int:
	args = parse_args()
	if args.list:
		print("Supported algorithms:")
		for algo in DEFAULT_ORDER:
			print(f"- {algo} -> {SCRIPT_MAP[algo]}")
		return 0

	try:
		algorithms = resolve_algorithms(args.algorithms)
	except ValueError as exc:
		print(f"Error: {exc}")
		return 2

	project_dir = Path(__file__).resolve().parent
	print("Selected algorithms:", ", ".join(algorithms))

	summary: list[tuple[str, int, float]] = []
	for algo in algorithms:
		try:
			code, elapsed = run_script(project_dir, algo)
		except Exception as exc:
			print(f"{algo.upper()} failed to start: {exc}")
			summary.append((algo, 1, 0.0))
			if not args.continue_on_error:
				break
			continue

		summary.append((algo, code, elapsed))
		if code != 0 and not args.continue_on_error:
			break

	print("\n=== Summary ===")
	failed = False
	for algo, code, elapsed in summary:
		status = "OK" if code == 0 else f"FAILED({code})"
		print(f"- {algo.upper():>3}: {status:>10} | {elapsed:7.1f}s")
		if code != 0:
			failed = True

	return 1 if failed else 0


if __name__ == "__main__":
	raise SystemExit(main())
