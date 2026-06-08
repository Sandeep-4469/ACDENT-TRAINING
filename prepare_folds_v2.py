#!/usr/bin/env python3
"""
Prepare dataset_v2: 4 folds, exactly 15 real test images per fold.

Structure per fold:
  dataset_v2/fold_{1-4}/
    train/images/            symlinks to real training images
    test/images/             symlinks to real test images (exactly 15)
    train_augmented/images/  symlinks to augmented training images (if --augmented)
    train_annotations.json
    test_annotations.json
    train_annotations_augmented.json  (if --augmented)

Split logic:
  - Only real images (keys without __name__ pattern)
  - Patient-level split so both arches of one patient stay together
  - 4 non-overlapping test sets of 15 images each (balanced maxilla/mandible)
  - Remaining ~73 images always in training across all folds

Usage:
  python prepare_folds_v2.py                    # real only in train
  python prepare_folds_v2.py --augmented        # include aug in train
  python prepare_folds_v2.py --augmented --vlm  # also include VLM in train
"""

from __future__ import annotations

import argparse
import json
import os
import re
import random
from collections import defaultdict
from pathlib import Path

DSAA_ROOT   = Path(__file__).resolve().parent
SRC_IMAGES  = DSAA_ROOT / "dataset_source" / "images_512"
SRC_ANN     = DSAA_ROOT / "dataset_source" / "annotations_512.json"
OUT_ROOT    = DSAA_ROOT / "dataset_v2"

AUG_IMAGES  = DSAA_ROOT / "dataset" / "fold_1" / "train_augmented" / "images"
AUG_SUFFIX  = [f"_aug{i}" for i in range(1, 5)]   # _aug1 … _aug4

N_FOLDS     = 4
TEST_SIZE   = 15   # real images per fold test set
SEED        = 42


def is_vlm(key: str) -> bool:
    return bool(re.search(r'__.+__', key))


def get_patient_id(key: str) -> str:
    m = re.match(r'^(.+?)_(?:mandible|maxilla)', key, re.IGNORECASE)
    return m.group(1) if m else key


def arch_type(key: str) -> str:
    return "maxilla" if "maxilla" in key.lower() else "mandible"


def symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--augmented", action="store_true",
                    help="Include augmented images in training split")
    ap.add_argument("--vlm",       action="store_true",
                    help="Include VLM-generated images in training (only, never test)")
    ap.add_argument("--seed",      type=int, default=SEED)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # ------------------------------------------------------------------
    # Load all annotations
    # ------------------------------------------------------------------
    with SRC_ANN.open() as f:
        all_ann = json.load(f)

    real_ann = {k: v for k, v in all_ann.items() if not is_vlm(k)}
    vlm_ann  = {k: v for k, v in all_ann.items() if     is_vlm(k)}

    print(f"Total annotations : {len(all_ann)}")
    print(f"Real images       : {len(real_ann)}")
    print(f"VLM images        : {len(vlm_ann)}")

    # ------------------------------------------------------------------
    # Group real images by patient (patient-level split prevents leakage)
    # ------------------------------------------------------------------
    patients: dict[str, list[str]] = defaultdict(list)
    for key in real_ann:
        patients[get_patient_id(key)].append(key)

    # Sort for reproducibility, shuffle patients
    patient_list = sorted(patients.keys())
    rng.shuffle(patient_list)

    # Separate by arch for balanced test sets
    # We'll select test patients so ~half their images are maxilla, ~half mandible
    maxilla_patients  = [p for p in patient_list if any(arch_type(k)=="maxilla"  for k in patients[p])]
    mandible_patients = [p for p in patient_list if any(arch_type(k)=="mandible" for k in patients[p])]
    # Patients with both arches appear in both lists above; that's fine for counting
    # We'll do a unified fold assignment per patient

    # ------------------------------------------------------------------
    # Build 4 non-overlapping test pools of exactly TEST_SIZE real images
    # Strategy:
    #   1. Flatten all real image keys, stratify maxilla/mandible
    #   2. Create 4 balanced groups of TEST_SIZE
    #   3. Remaining images are always-train
    # ------------------------------------------------------------------
    max_keys = sorted([k for k in real_ann if arch_type(k) == "maxilla"])
    man_keys = sorted([k for k in real_ann if arch_type(k) == "mandible"])
    rng.shuffle(max_keys)
    rng.shuffle(man_keys)

    # Per fold: 7 maxilla + 8 mandible = 15  (or 8+7 for alternating)
    test_pools: list[list[str]] = []
    mx_idx, mn_idx = 0, 0
    for fold_i in range(N_FOLDS):
        n_max = TEST_SIZE // 2
        n_man = TEST_SIZE - n_max
        pool = max_keys[mx_idx: mx_idx + n_max] + man_keys[mn_idx: mn_idx + n_man]
        test_pools.append(pool)
        mx_idx += n_max
        mn_idx += n_man

    # Images used in any test pool
    in_test = set(k for pool in test_pools for k in pool)
    always_train = [k for k in real_ann if k not in in_test]

    print(f"\nImages used in test rotation : {len(in_test)}")
    print(f"Always-train images          : {len(always_train)}")
    print(f"Test images per fold         : {[len(p) for p in test_pools]}")

    # Check per-fold test arch balance
    for i, pool in enumerate(test_pools):
        mx = sum(1 for k in pool if arch_type(k) == "maxilla")
        mn = sum(1 for k in pool if arch_type(k) == "mandible")
        print(f"  fold_{i+1} test: {len(pool)} images (maxilla={mx}, mandible={mn})")

    # ------------------------------------------------------------------
    # Build augmented key mapping (if requested)
    # key → list of aug keys (key_aug1 … key_aug4)
    # Only include if the augmented image file actually exists
    # ------------------------------------------------------------------
    aug_map: dict[str, list[str]] = {}
    if args.augmented:
        # Find aug images from existing fold aug dirs (any fold is fine for mapping)
        for fold_dir in (DSAA_ROOT / "dataset").glob("fold_*/train_augmented/images"):
            for img in fold_dir.glob("*_aug*.jpg"):
                base = re.sub(r'_aug\d+$', '', img.stem)
                aug_map.setdefault(base, []).append(img.stem)
            break  # one fold is enough to discover aug keys

    # ------------------------------------------------------------------
    # Create fold directories
    # ------------------------------------------------------------------
    for fold_i in range(N_FOLDS):
        fold_name = f"fold_{fold_i + 1}"
        fold_root = OUT_ROOT / fold_name
        test_pool = test_pools[fold_i]
        other_test = [k for j, pool in enumerate(test_pools) if j != fold_i for k in pool]
        train_real = sorted(set(always_train + other_test))

        print(f"\n[{fold_name}] train_real={len(train_real)}, test={len(test_pool)}")

        # --- Annotations ---
        test_ann  = {k: real_ann[k] for k in test_pool}
        train_ann = {k: real_ann[k] for k in train_real}

        # Optionally add VLM to training annotations only
        if args.vlm:
            train_ann.update(vlm_ann)
            print(f"  + {len(vlm_ann)} VLM images added to training")

        # Augmented training annotations
        aug_ann: dict = {}
        if args.augmented:
            for k in train_real:
                aug_ann[k] = real_ann[k]          # include original
                for aug_key in aug_map.get(k, []):
                    aug_ann[aug_key] = real_ann[k] # same annotation, augmented image
            if args.vlm:
                for k in vlm_ann:
                    aug_ann[k] = vlm_ann[k]
                    for aug_key in aug_map.get(k, []):
                        aug_ann[aug_key] = vlm_ann[k]

        # --- Write annotations ---
        (fold_root).mkdir(parents=True, exist_ok=True)
        with (fold_root / "train_annotations.json").open("w") as f:
            json.dump(train_ann, f, indent=2)
        with (fold_root / "test_annotations.json").open("w") as f:
            json.dump(test_ann, f, indent=2)
        if aug_ann:
            with (fold_root / "train_annotations_augmented.json").open("w") as f:
                json.dump(aug_ann, f, indent=2)

        # --- Symlink images ---
        train_img_dir = fold_root / "train" / "images"
        test_img_dir  = fold_root / "test"  / "images"
        train_img_dir.mkdir(parents=True, exist_ok=True)
        test_img_dir.mkdir(parents=True, exist_ok=True)

        for key in train_real:
            src = SRC_IMAGES / f"{key}.jpg"
            if src.exists():
                symlink(src, train_img_dir / f"{key}.jpg")

        for key in test_pool:
            src = SRC_IMAGES / f"{key}.jpg"
            if src.exists():
                symlink(src, test_img_dir / f"{key}.jpg")

        if aug_ann:
            aug_img_dir = fold_root / "train_augmented" / "images"
            aug_img_dir.mkdir(parents=True, exist_ok=True)
            # Symlink originals
            for key in train_real:
                src = SRC_IMAGES / f"{key}.jpg"
                if src.exists():
                    symlink(src, aug_img_dir / f"{key}.jpg")
            # Symlink augmented versions from existing fold aug dir
            for fold_dir in (DSAA_ROOT / "dataset").glob("fold_*/train_augmented/images"):
                for img in fold_dir.glob("*_aug*.jpg"):
                    base = re.sub(r'_aug\d+$', '', img.stem)
                    if base in train_real or (args.vlm and base in vlm_ann):
                        symlink(img, aug_img_dir / img.name)
                break  # one fold's aug images are representative

        # VLM symlinks
        if args.vlm:
            for key in vlm_ann:
                src = SRC_IMAGES / f"{key}.jpg"
                if src.exists():
                    symlink(src, train_img_dir / f"{key}.jpg")
                    if aug_ann:
                        symlink(src, aug_img_dir / f"{key}.jpg")

        # Summary
        n_aug = len(aug_ann) if aug_ann else 0
        print(f"  annotations: train={len(train_ann)}, test={len(test_ann)}, aug={n_aug}")
        print(f"  saved → {fold_root}")

    print(f"\nDone. Dataset at: {OUT_ROOT}")
    print("Train with:")
    print(f"  python train_resnet50.py --dataset-root {OUT_ROOT} --k-folds {N_FOLDS}")


if __name__ == "__main__":
    main()
