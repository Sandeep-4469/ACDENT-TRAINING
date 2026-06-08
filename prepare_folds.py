#!/usr/bin/env python3
"""
Prepare 5-fold cross-validation dataset with offline augmentation for DSAA_Dental.

Steps:
  1. Read annotations_512.json from dataset_source/
  2. Exclude boston images (controlled by EXCLUDE_BOSTON flag)
  3. Group keys by patient, shuffle, split into K patient-level folds
  4. For each fold create:
       dataset/fold_X/train/images/              <- symlinks to original 512px images
       dataset/fold_X/train_annotations.json
       dataset/fold_X/train_augmented/images/    <- 4 augmentations per original + original copy
       dataset/fold_X/train_annotations_augmented.json
       dataset/fold_X/test/images/               <- symlinks (no augmentation)
       dataset/fold_X/test_annotations.json
  5. Write dataset/split_metadata.json with fold info

Run ONCE before any training:
  python prepare_folds.py

Optional flags:
  --k-folds       Number of folds (default 5)
  --seed          Random seed for fold splitting and augmentation (default 42)
  --aug-per-image Augmented copies per original image (default 4)
  --no-boston     Exclude boston images (default: True, set flag to keep them)
  --dry-run       Skip writing images/JSON (for testing the split logic)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List

from augment import augment_fold

DSAA_ROOT    = Path(__file__).resolve().parent
DATASET_ROOT = DSAA_ROOT / "dataset"

SOURCE_ROOT        = DSAA_ROOT / "dataset_source"
SOURCE_ANNOTATIONS = SOURCE_ROOT / "annotations_512.json"
SOURCE_IMAGES      = SOURCE_ROOT / "images_512"

EXCLUDE_BOSTON = True


def patient_id_from_key(key: str) -> str:
    for tok in ("__mandible__", "__maxilla__", "_mandible__", "_maxilla__", "_mandible", "_maxilla"):
        if tok in key:
            return key.split(tok, 1)[0].strip()
    return key.strip()


def link_or_copy(src: Path, dst: Path) -> None:
    src = src.resolve()
    if dst.exists():
        return
    if dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DSAA_Dental 5-fold CV dataset.")
    parser.add_argument("--source-annotations", type=Path, default=SOURCE_ANNOTATIONS)
    parser.add_argument("--source-images",       type=Path, default=SOURCE_IMAGES)
    parser.add_argument("--dataset-root",        type=Path, default=DATASET_ROOT)
    parser.add_argument("--k-folds",             type=int,  default=5)
    parser.add_argument("--seed",                type=int,  default=42)
    parser.add_argument("--aug-per-image",        type=int,  default=4)
    parser.add_argument("--jpg-quality",         type=int,  default=95)
    parser.add_argument("--keep-boston",         action="store_true",
                        help="Include boston images (excluded by default)")
    parser.add_argument("--dry-run",             action="store_true",
                        help="Print split info without writing images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exclude_boston = not args.keep_boston

    if not args.source_annotations.exists():
        raise FileNotFoundError(f"Source annotations not found: {args.source_annotations}")
    if not args.source_images.exists():
        raise FileNotFoundError(f"Source images not found: {args.source_images}")

    with args.source_annotations.open("r", encoding="utf-8") as f:
        raw_data: Dict[str, dict] = json.load(f)

    if exclude_boston:
        data = {k: v for k, v in raw_data.items() if "boston" not in k.lower()}
    else:
        data = dict(raw_data)

    print(f"Source records:   {len(raw_data)}")
    print(f"After filtering:  {len(data)} (excluded {len(raw_data) - len(data)} boston)")

    # Group by patient (patient-level split prevents data leakage across folds)
    patient_to_keys: Dict[str, List[str]] = {}
    for key in data:
        pid = patient_id_from_key(key)
        patient_to_keys.setdefault(pid, []).append(key)

    import random
    patients = list(patient_to_keys)
    rng = random.Random(args.seed)
    rng.shuffle(patients)

    k = args.k_folds
    folds: List[List[str]] = [[] for _ in range(k)]
    for idx, patient in enumerate(patients):
        folds[idx % k].extend(patient_to_keys[patient])

    print(f"\nPatient groups:   {len(patient_to_keys)}")
    print(f"K-folds:          {k}")
    for i, fold_keys in enumerate(folds):
        print(f"  fold_{i+1}: {len(fold_keys)} samples "
              f"({len([patient_to_keys[p] for p in patients if any(k_ in fold_keys for k_ in patient_to_keys[p])])} patients)")

    if not args.dry_run:
        args.dataset_root.mkdir(parents=True, exist_ok=True)

    fold_meta = []
    for fold_idx in range(k):
        fold_name = f"fold_{fold_idx + 1}"
        fold_root = args.dataset_root / fold_name

        test_keys  = set(folds[fold_idx])
        train_data = {key: data[key] for key in data if key not in test_keys}
        test_data  = {key: data[key] for key in data if key in test_keys}

        train_img_dir = fold_root / "train" / "images"
        test_img_dir  = fold_root / "test"  / "images"
        aug_img_dir   = fold_root / "train_augmented" / "images"
        train_ann     = fold_root / "train_annotations.json"
        test_ann      = fold_root / "test_annotations.json"
        aug_ann       = fold_root / "train_annotations_augmented.json"

        print(f"\n{'='*55}")
        print(f"Creating {fold_name}: {len(train_data)} train | {len(test_data)} test")

        if not args.dry_run:
            train_img_dir.mkdir(parents=True, exist_ok=True)
            test_img_dir.mkdir(parents=True, exist_ok=True)

            with train_ann.open("w", encoding="utf-8") as f:
                json.dump(train_data, f, indent=2, ensure_ascii=False)
            with test_ann.open("w", encoding="utf-8") as f:
                json.dump(test_data, f, indent=2, ensure_ascii=False)

            for key in train_data:
                src = args.source_images / f"{key}.jpg"
                if src.exists():
                    link_or_copy(src, train_img_dir / f"{key}.jpg")
            for key in test_data:
                src = args.source_images / f"{key}.jpg"
                if src.exists():
                    link_or_copy(src, test_img_dir / f"{key}.jpg")

            # Offline augmentation on train fold
            print(f"  Augmenting train fold ({args.aug_per_image} aug/image)...")
            augment_fold(
                train_images=train_img_dir,
                train_json=train_ann,
                out_images=aug_img_dir,
                out_json=aug_ann,
                aug_per_image=args.aug_per_image,
                seed=args.seed + fold_idx,
                jpg_quality=args.jpg_quality,
                dry_run=False,
            )

        fold_meta.append({
            "fold":          fold_name,
            "fold_root":     str(fold_root),
            "train_samples": len(train_data),
            "test_samples":  len(test_data),
            "aug_samples":   len(train_data) * (args.aug_per_image + 1),
            "train_ann":     str(train_ann),
            "test_ann":      str(test_ann),
            "aug_ann":       str(aug_ann),
        })

    # Write metadata
    meta = {
        "source_annotations":    str(args.source_annotations),
        "source_images":         str(args.source_images),
        "dataset_root":          str(args.dataset_root),
        "exclude_boston":        exclude_boston,
        "total_source_records":  len(raw_data),
        "total_filtered_records": len(data),
        "total_patients":        len(patient_to_keys),
        "k_folds":               k,
        "seed":                  args.seed,
        "aug_per_image":         args.aug_per_image,
        "fold_sizes":            [len(fold) for fold in folds],
        "folds":                 fold_meta,
    }
    if not args.dry_run:
        meta_path = args.dataset_root / "split_metadata.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"\nDataset prepared at: {args.dataset_root}")
        print(f"Split metadata:      {meta_path}")

    print("\nDone. You can now run:")
    print("  python train_resnet50.py")
    print("  python train_resnet18.py")
    print("  python train_convnext_tiny.py")


if __name__ == "__main__":
    main()
