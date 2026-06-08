#!/usr/bin/env python3
"""
Compute Moyers and Tanaka-Johnston tooth space discrepancy:
  - GT (ground truth) using original GT scale px→mm
  - Predicted using model keypoints + GT scale (original scale only)

For each patient with BOTH mandible + maxilla in the test set:
  - Incisors come from the mandible measurement (as in app3.py)
  - Arc available space from mandible (left/right arc) for mandible discrepancy
  - Arc available space from maxilla (left/right arc) for maxilla discrepancy

Tanaka-Johnston:  predicted_per_side = (sum_incisors / 2) + constant
                  constant = 10.5 (mandible), 11.0 (maxilla)
Moyers (75%ile):  lookup table from app3.py

Discrepancy = (left_avail - pred_per_side) + (right_avail - pred_per_side)

Usage:
  python tooth_discrepancy.py            # all backbones, all folds
  python tooth_discrepancy.py --backbone resnet101
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

DSAA = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Moyers / Tanaka-Johnston formulas (from app3.py)
# ---------------------------------------------------------------------------
MOYERS_TABLE = {
    "male":   {19.5:20.4,20.0:20.8,20.5:21.1,21.0:21.4,21.5:21.8,22.0:22.1,
               22.5:22.5,23.0:22.8,23.5:23.1,24.0:23.5,24.5:23.8,25.0:24.1,25.5:24.5},
    "female": {19.5:20.4,20.0:20.7,20.5:21.0,21.0:21.4,21.5:21.7,22.0:22.0,
               22.5:22.4,23.0:22.7,23.5:23.0,24.0:23.3,24.5:23.7,25.0:24.0,25.5:24.3},
}

def tanaka_johnston(incisors_mm: List[float], arch: str) -> float:
    """Predicted canine+premolar space per side (mm)."""
    s = sum(incisors_mm)
    return (s / 2.0) + (11.0 if arch == "maxilla" else 10.5)

def moyers(incisors_mm: List[float], arch: str, gender: str = "male") -> float:
    s = sum(incisors_mm)
    key = round(10.5 + (s / 2.0), 1)
    tbl = MOYERS_TABLE.get(gender.lower(), MOYERS_TABLE["male"])
    ck  = min(tbl.keys(), key=lambda k: abs(key - k))
    val = tbl[ck]
    if arch == "maxilla":
        val += 0.5
    return round(val, 2)

def discrepancy(left_mm: float, right_mm: float, pred_per_side: float) -> float:
    return round((left_mm - pred_per_side) + (right_mm - pred_per_side), 3)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def line_len(p1, p2) -> float:
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def patient_id(key: str) -> str:
    m = re.match(r'^(.+?)_(?:mandible|maxilla)', key, re.IGNORECASE)
    return m.group(1).strip() if m else key

def arch_of(key: str) -> str:
    return "maxilla" if "maxilla" in key.lower() else "mandible"

def gt_measurements(ann: dict, px2mm: float) -> dict:
    """Extract GT measurements from annotation dict using GT scale."""
    lines = ann.get("lines", [])
    if len(lines) == 7:  # mandible
        incisors = [line_len(lines[i][0], lines[i][1]) * px2mm for i in range(1, 5)]
        left_arc  = line_len(lines[5][0], lines[5][1]) * px2mm
        right_arc = line_len(lines[6][0], lines[6][1]) * px2mm
        return {"arch":"mandible","incisors":incisors,"left_arc":left_arc,"right_arc":right_arc}
    elif len(lines) == 3:  # maxilla
        left_arc  = line_len(lines[1][0], lines[1][1]) * px2mm
        right_arc = line_len(lines[2][0], lines[2][1]) * px2mm
        return {"arch":"maxilla","left_arc":left_arc,"right_arc":right_arc}
    return {}

def pred_measurements(pred_kps: List, arch: str, px2mm: float) -> dict:
    """Extract predicted measurements from keypoints using GT scale (original scale)."""
    pts = pred_kps
    if arch == "mandible":
        incisors = [line_len(pts[2*i], pts[2*i+1]) * px2mm for i in range(1, 5)]
        left_arc  = line_len(pts[10], pts[11]) * px2mm
        right_arc = line_len(pts[12], pts[13]) * px2mm
        return {"arch":"mandible","incisors":incisors,"left_arc":left_arc,"right_arc":right_arc}
    else:
        left_arc  = line_len(pts[2], pts[3]) * px2mm
        right_arc = line_len(pts[4], pts[5]) * px2mm
        return {"arch":"maxilla","left_arc":left_arc,"right_arc":right_arc}

def compute_discrepancies(man: dict, max_: dict, formula: str) -> dict:
    """Compute discrepancies for both arches given mandible+maxilla measurements."""
    inc = man["incisors"]
    fn  = tanaka_johnston if formula == "tanaka" else moyers

    pred_man = fn(inc, "mandible")
    pred_max = fn(inc, "maxilla")

    disc_man = discrepancy(man["left_arc"], man["right_arc"], pred_man)
    disc_max = discrepancy(max_["left_arc"], max_["right_arc"], pred_max)

    return {
        "incisor_sum":   sum(inc),
        "pred_per_side_man": pred_man,
        "pred_per_side_max": pred_max,
        "man_left_arc":  man["left_arc"],
        "man_right_arc": man["right_arc"],
        "max_left_arc":  max_["left_arc"],
        "max_right_arc": max_["right_arc"],
        "disc_mandible": disc_man,
        "disc_maxilla":  disc_max,
        "disc_total":    round(disc_man + disc_max, 3),
    }

# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def analyse_backbone(backbone: str, formula: str = "tanaka") -> List[dict]:
    rows = []
    compare_dir = DSAA / "results" / "backbone_compare" / backbone

    for fold_i in range(1, 5):
        fold     = f"fold_{fold_i}"
        pred_json = compare_dir / fold / "test_predictions.json"
        gt_json   = DSAA / "dataset_v2" / fold / "test_annotations.json"

        if not pred_json.exists() or not gt_json.exists():
            continue

        preds = json.load(open(pred_json))
        gt_all = json.load(open(gt_json))

        # Group by patient
        patients: Dict[str, Dict[str, str]] = {}
        for key in gt_all:
            pid   = patient_id(key)
            arch  = arch_of(key)
            patients.setdefault(pid, {})[arch] = key

        # Only process patients with BOTH arches in this test fold
        for pid, arches in patients.items():
            if "mandible" not in arches or "maxilla" not in arches:
                continue

            man_key = arches["mandible"]
            max_key = arches["maxilla"]

            if man_key not in preds or max_key not in preds:
                continue

            man_ann = gt_all[man_key]
            max_ann = gt_all[max_key]

            # GT scale from mandible (has scale line)
            man_lines = man_ann.get("lines", [])
            if len(man_lines) < 1:
                continue
            scale_px = line_len(man_lines[0][0], man_lines[0][1])
            if scale_px < 1:
                continue
            px2mm = 5.0 / scale_px   # GT scale → mm

            # GT measurements
            gt_man = gt_measurements(man_ann, px2mm)
            gt_max = gt_measurements(max_ann, px2mm)
            if not gt_man or not gt_max:
                continue

            # Predicted measurements (using GT scale)
            pr_man = pred_measurements(preds[man_key]["predicted_keypoints"], "mandible", px2mm)
            pr_max = pred_measurements(preds[max_key]["predicted_keypoints"], "maxilla",  px2mm)

            # Compute discrepancies
            gt_disc   = compute_discrepancies(gt_man,  gt_max,  formula)
            pred_disc = compute_discrepancies(pr_man,  pr_max,  formula)

            rows.append({
                "patient":   pid,
                "fold":      fold,
                "backbone":  backbone,
                "formula":   formula,
                # GT
                "gt_incisor_sum":       round(gt_disc["incisor_sum"],    3),
                "gt_man_left_arc":      round(gt_disc["man_left_arc"],   3),
                "gt_man_right_arc":     round(gt_disc["man_right_arc"],  3),
                "gt_max_left_arc":      round(gt_disc["max_left_arc"],   3),
                "gt_max_right_arc":     round(gt_disc["max_right_arc"],  3),
                "gt_pred_per_side_man": round(gt_disc["pred_per_side_man"], 3),
                "gt_pred_per_side_max": round(gt_disc["pred_per_side_max"], 3),
                "gt_disc_mandible":     gt_disc["disc_mandible"],
                "gt_disc_maxilla":      gt_disc["disc_maxilla"],
                "gt_disc_total":        gt_disc["disc_total"],
                # Predicted
                "pr_incisor_sum":       round(pred_disc["incisor_sum"],    3),
                "pr_man_left_arc":      round(pred_disc["man_left_arc"],   3),
                "pr_man_right_arc":     round(pred_disc["man_right_arc"],  3),
                "pr_max_left_arc":      round(pred_disc["max_left_arc"],   3),
                "pr_max_right_arc":     round(pred_disc["max_right_arc"],  3),
                "pr_pred_per_side_man": round(pred_disc["pred_per_side_man"], 3),
                "pr_pred_per_side_max": round(pred_disc["pred_per_side_max"], 3),
                "pr_disc_mandible":     pred_disc["disc_mandible"],
                "pr_disc_maxilla":      pred_disc["disc_maxilla"],
                "pr_disc_total":        pred_disc["disc_total"],
                # Error (|GT - Pred|)
                "err_disc_mandible":    round(abs(gt_disc["disc_mandible"] - pred_disc["disc_mandible"]), 3),
                "err_disc_maxilla":     round(abs(gt_disc["disc_maxilla"]  - pred_disc["disc_maxilla"]),  3),
                "err_disc_total":       round(abs(gt_disc["disc_total"]    - pred_disc["disc_total"]),    3),
            })

    return rows


def print_summary(all_rows: List[dict], formula: str) -> None:
    from collections import defaultdict
    by_backbone: Dict[str, List[dict]] = defaultdict(list)
    for r in all_rows:
        by_backbone[r["backbone"]].append(r)

    print(f"\n{'='*80}")
    print(f"TOOTH DISCREPANCY — {formula.upper()}  (GT scale only, paired patients)")
    print(f"{'='*80}")

    print(f"\n{'Backbone':<20} {'Patients':>9} {'|ΔDisc Man|':>12} {'|ΔDisc Max|':>12} {'|ΔDisc Tot|':>12}")
    print("-"*68)
    for bb, rows in sorted(by_backbone.items(), key=lambda x: np.mean([r["err_disc_total"] for r in x[1]])):
        n    = len(rows)
        eman = np.mean([r["err_disc_mandible"] for r in rows])
        emax = np.mean([r["err_disc_maxilla"]  for r in rows])
        etot = np.mean([r["err_disc_total"]    for r in rows])
        print(f"{bb:<20} {n:>9} {eman:>12.3f} {emax:>12.3f} {etot:>12.3f}")

    # Per-patient detail for best backbone
    if not all_rows:
        return
    best_bb = min(by_backbone.keys(),
                  key=lambda bb: np.mean([r["err_disc_total"] for r in by_backbone[bb]]))
    rows = by_backbone[best_bb]
    print(f"\n--- Per-patient detail: {best_bb} ({formula}) ---")
    print(f"{'Patient':<30} {'GT Man':>8} {'Pr Man':>8} {'GT Max':>8} {'Pr Max':>8} {'GT Tot':>8} {'Pr Tot':>8} {'|Err|':>7}")
    print("-"*85)
    for r in sorted(rows, key=lambda x: x["patient"]):
        print(f"{r['patient']:<30} "
              f"{r['gt_disc_mandible']:>8.2f} {r['pr_disc_mandible']:>8.2f} "
              f"{r['gt_disc_maxilla']:>8.2f}  {r['pr_disc_maxilla']:>8.2f} "
              f"{r['gt_disc_total']:>8.2f} {r['pr_disc_total']:>8.2f} "
              f"{r['err_disc_total']:>7.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", type=str, default=None,
                    help="Specific backbone (default: all available)")
    ap.add_argument("--formula",  type=str, default="both",
                    choices=["tanaka","moyers","both"])
    ap.add_argument("--out-csv",  type=str, default=None)
    args = ap.parse_args()

    compare_dir = DSAA / "results" / "backbone_compare"
    backbones = [args.backbone] if args.backbone else \
                [d.name for d in compare_dir.iterdir() if d.is_dir()]

    formulas = ["tanaka","moyers"] if args.formula == "both" else [args.formula]

    all_rows = []
    for formula in formulas:
        rows = []
        for bb in backbones:
            bb_rows = analyse_backbone(bb, formula)
            if not bb_rows:
                print(f"  {bb}: no paired patients found")
            rows.extend(bb_rows)
        all_rows.extend(rows)
        print_summary(rows, formula)

    if args.out_csv:
        import csv
        if all_rows:
            with open(args.out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
                w.writeheader()
                w.writerows(all_rows)
            print(f"\nSaved: {args.out_csv}")

    # Quick count of paired patients
    print(f"\nTotal paired patient-fold observations: {len([r for r in all_rows if r['formula']=='tanaka'])}")


if __name__ == "__main__":
    main()
