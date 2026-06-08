#!/usr/bin/env python3
"""
Run ACDentNet backbone comparison — trains all backbones sequentially (5-fold CV each).

Backbones:
  resnet18 | resnet101 | convnext_tiny | efficientnet_b3 | hrnet_w32

Usage:
  python run_backbone_comparison.py                          # all backbones
  python run_backbone_comparison.py --backbones resnet18 hrnet_w32
  python run_backbone_comparison.py --epochs 200 --batch-size 4
  python run_backbone_comparison.py --skip resnet101        # exclude one
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

DSAA_ROOT = Path(__file__).resolve().parent

BACKBONES = [
    "resnet18",
    "resnet101",
    "convnext_tiny",
    "efficientnet_b3",
    "hrnet_w32",
]

SCRIPT_MAP = {
    "resnet18":       DSAA_ROOT / "train_resnet18.py",
    "resnet101":      DSAA_ROOT / "train_resnet101.py",
    "convnext_tiny":  DSAA_ROOT / "train_convnext_tiny.py",
    "efficientnet_b3": DSAA_ROOT / "train_efficientnet_b3.py",
    "hrnet_w32":      DSAA_ROOT / "train_hrnet_w32.py",
}


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def header(text: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {text}")
    print(f"  {ts()}")
    print(f"{'='*65}\n", flush=True)


def run_backbone(backbone: str, extra_args: list[str]) -> tuple[str, float]:
    script = SCRIPT_MAP[backbone]
    cmd    = [sys.executable, str(script)] + extra_args
    header(f"Training — {backbone}")
    print(f"  Command: {' '.join(cmd)}\n", flush=True)

    t0     = time.time()
    result = subprocess.run(cmd, cwd=str(DSAA_ROOT))
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    return status, elapsed


def print_summary(results: list[tuple]) -> None:
    print(f"\n{'='*65}")
    print("  BACKBONE COMPARISON — RUN SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Backbone':<20} {'Status':<10} {'Time'}")
    print(f"  {'-'*50}")
    total_ok = 0
    for backbone, status, elapsed in results:
        icon = "OK" if status == "OK" else "!!"
        mins = elapsed / 60
        print(f"  [{icon}] {backbone:<20} {status:<10} {mins:.1f} min")
        if status == "OK":
            total_ok += 1
    print(f"{'='*65}")
    print(f"  {total_ok}/{len(results)} completed successfully.")
    print(f"  Results saved to: {DSAA_ROOT / 'results'}/train_<backbone>/\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ACDentNet backbone comparison (5-fold CV).")
    parser.add_argument("--backbones", nargs="+", default=BACKBONES,
                        choices=BACKBONES, metavar="B",
                        help=f"Backbones to run (default: all). Choices: {BACKBONES}")
    parser.add_argument("--skip", nargs="+", default=[], choices=BACKBONES, metavar="B",
                        help="Backbones to skip")
    parser.add_argument("--epochs",     type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--k-folds",    type=int, default=5)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    backbones = [b for b in args.backbones if b not in args.skip]

    extra_args = [f"--k-folds={args.k_folds}", f"--seed={args.seed}"]
    if args.epochs:
        extra_args.append(f"--epochs={args.epochs}")
    if args.batch_size:
        extra_args.append(f"--batch-size={args.batch_size}")

    print(f"\n{'='*65}")
    print(f"  ACDentNet — Backbone Comparison")
    print(f"  Backbones : {backbones}")
    print(f"  Extra args: {extra_args}")
    print(f"  Started   : {ts()}")
    print(f"{'='*65}")

    total_t0 = time.time()
    results  = []

    for backbone in backbones:
        status, elapsed = run_backbone(backbone, extra_args)
        results.append((backbone, status, elapsed))

    print_summary(results)
    total_min = (time.time() - total_t0) / 60
    print(f"  Total wall time: {total_min:.1f} min  |  Finished: {ts()}\n")
