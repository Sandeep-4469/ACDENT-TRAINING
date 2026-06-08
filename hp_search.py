#!/usr/bin/env python3
"""
Hyperparameter search for train_resnet50.py.
Runs each config on fold_1 only (fast feedback), logs all results,
and prints the best config to run on all 5 folds.

Strategy: random search over the most impactful parameters.
Based on ablation: coord_loss and length_loss have the biggest single effects.
LR and sigma also meaningfully affect convergence and precision.

Usage:
  python hp_search.py                  # 15 random trials, 150 epochs each
  python hp_search.py --trials 20 --epochs 120
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

DSAA_ROOT  = Path(__file__).resolve().parent
SCRIPT     = DSAA_ROOT / "train_resnet50.py"
RESULTS    = DSAA_ROOT / "results" / "hp_search"

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
SEARCH_SPACE = {
    "lr":             [5e-5, 1e-4, 2e-4],
    "coord_loss_w":   [0.01, 0.02, 0.05],
    "length_loss_w":  [0.01, 0.02, 0.05],
    "arc_contrast_w": [0.004, 0.008, 0.016],
    "sigma":          [2, 3, 4],
    # arc_side and arc_center kept fixed — ablation shows less sensitivity
}

# Fixed for all trials
FIXED = {
    "arc_side_w":   0.008,
    "arc_center_w": 0.005,
    "exclude_vlm":  True,   # use clean real-only data
}


def sample_config(rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}


def config_to_args(cfg: dict, run_name: str, epochs: int) -> list[str]:
    cmd = [
        sys.executable, str(SCRIPT),
        "--k-folds",       "1",
        "--epochs",        str(epochs),
        "--run-name",      run_name,
        "--save-root",     str(RESULTS),
        "--lr",            str(cfg["lr"]),
        "--coord-loss-w",  str(cfg["coord_loss_w"]),
        "--length-loss-w", str(cfg["length_loss_w"]),
        "--arc-side-w",    str(FIXED["arc_side_w"]),
        "--arc-center-w",  str(FIXED["arc_center_w"]),
        "--arc-contrast-w",str(cfg["arc_contrast_w"]),
        "--sigma",         str(cfg["sigma"]),
    ]
    if FIXED["exclude_vlm"]:
        cmd.append("--exclude-vlm")
    return cmd


def read_result(run_name: str) -> dict | None:
    summary = RESULTS / run_name / "kfold_summary.csv"
    if not summary.exists():
        return None
    with summary.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    r = rows[0]
    return {
        "incisor_mm":  float(r.get("incisor_sum_mm") or 0),
        "left_arc_mm": float(r.get("left_arc_mm")    or 0),
        "right_arc_mm":float(r.get("right_arc_mm")   or 0),
        "sum3_mm":     float(r.get("sum_incisor_left_right_mm") or 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials",  type=int, default=15)
    ap.add_argument("--epochs",  type=int, default=150)
    ap.add_argument("--seed",    type=int, default=7)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    log_path = RESULTS / f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_fields = ["trial", "run_name", "sum3_mm", "incisor_mm", "left_arc_mm", "right_arc_mm",
                  "lr", "coord_loss_w", "length_loss_w", "arc_contrast_w", "sigma"]

    all_results = []

    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

        for trial in range(1, args.trials + 1):
            cfg      = sample_config(rng)
            run_name = f"trial_{trial:03d}"
            print(f"\n{'='*60}")
            print(f"Trial {trial}/{args.trials}: {cfg}")
            print(f"{'='*60}")

            cmd = config_to_args(cfg, run_name, args.epochs)
            ret = subprocess.run(cmd, cwd=str(DSAA_ROOT))

            result = read_result(run_name)
            if result is None:
                print(f"  [Trial {trial}] Failed — no result found")
                continue

            row = {"trial": trial, "run_name": run_name, **result, **cfg}
            writer.writerow(row)
            f.flush()
            all_results.append(row)

            print(f"  sum3={result['sum3_mm']:.3f}mm | "
                  f"incisor={result['incisor_mm']:.3f} | "
                  f"l_arc={result['left_arc_mm']:.3f} | "
                  f"r_arc={result['right_arc_mm']:.3f}")

    if not all_results:
        print("No results collected.")
        return

    # Sort and report
    all_results.sort(key=lambda r: r["sum3_mm"])
    best = all_results[0]

    print(f"\n{'='*60}")
    print("HYPERPARAMETER SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Results log: {log_path}\n")
    print("Top 5 configs:")
    for i, r in enumerate(all_results[:5], 1):
        print(f"  #{i} sum3={r['sum3_mm']:.3f}mm | "
              f"incisor={r['incisor_mm']:.3f} | "
              f"l_arc={r['left_arc_mm']:.3f} | "
              f"r_arc={r['right_arc_mm']:.3f} | "
              f"lr={r['lr']} coord={r['coord_loss_w']} len={r['length_loss_w']} "
              f"contrast={r['arc_contrast_w']} sigma={r['sigma']}")

    print(f"\nBest config (run on all 5 folds with):")
    print(f"  python train_resnet50.py --k-folds 5 --exclude-vlm \\")
    print(f"    --lr {best['lr']} \\")
    print(f"    --coord-loss-w {best['coord_loss_w']} \\")
    print(f"    --length-loss-w {best['length_loss_w']} \\")
    print(f"    --arc-contrast-w {best['arc_contrast_w']} \\")
    print(f"    --sigma {best['sigma']}")


if __name__ == "__main__":
    main()
