#!/usr/bin/env python3
"""
Augment dental train-fold images and annotations without mirror flips.

Input:
  dataset/fold_X/train/images/
  dataset/fold_X/train_annotations.json

Output:
  dataset/fold_X/train_augmented/images/
  dataset/fold_X/train_annotations_augmented.json

This script is called automatically by prepare_folds.py for each fold.
It can also be run standalone per fold.

Example (standalone, fold 1):
  python augment.py \
    --train-images dataset/fold_1/train/images \
    --train-json   dataset/fold_1/train_annotations.json \
    --out-images   dataset/fold_1/train_augmented/images \
    --out-json     dataset/fold_1/train_annotations_augmented.json
"""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

DSAA_ROOT = Path(__file__).resolve().parent

DEFAULT_TRAIN_IMAGES = DSAA_ROOT / "dataset" / "fold_1" / "train" / "images"
DEFAULT_TRAIN_JSON   = DSAA_ROOT / "dataset" / "fold_1" / "train_annotations.json"
DEFAULT_OUT_IMAGES   = DSAA_ROOT / "dataset" / "fold_1" / "train_augmented" / "images"
DEFAULT_OUT_JSON     = DSAA_ROOT / "dataset" / "fold_1" / "train_annotations_augmented.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment train split without mirror flips.")
    parser.add_argument("--train-images", type=Path, default=DEFAULT_TRAIN_IMAGES)
    parser.add_argument("--train-json",   type=Path, default=DEFAULT_TRAIN_JSON)
    parser.add_argument("--out-images",   type=Path, default=DEFAULT_OUT_IMAGES)
    parser.add_argument("--out-json",     type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--aug-per-image", type=int, default=4)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--jpg-quality",   type=int, default=95)
    parser.add_argument("--dry-run",       action="store_true")
    return parser.parse_args()


def clamp_point(x: float, y: float, w: int, h: int) -> List[float]:
    return [float(min(max(x, 0.0), w - 1.0)), float(min(max(y, 0.0), h - 1.0))]


def transform_lines(lines: List[list], M: np.ndarray, w: int, h: int) -> List[list]:
    out = []
    for line in lines:
        if not (isinstance(line, list) and len(line) == 2):
            continue
        new_line = []
        for p in line:
            if not (isinstance(p, list) and len(p) == 2):
                continue
            px, py = float(p[0]), float(p[1])
            nx, ny = np.dot(M, np.array([px, py, 1.0], dtype=np.float32))
            new_line.append(clamp_point(float(nx), float(ny), w, h))
        if len(new_line) == 2:
            out.append(new_line)
    return out


def random_affine(
    image: np.ndarray, lines: List[list], rng: random.Random
) -> Tuple[np.ndarray, List[list]]:
    h, w = image.shape[:2]
    angle = rng.uniform(-8.0, 8.0)
    scale = rng.uniform(0.94, 1.06)
    tx = rng.uniform(-0.04 * w, 0.04 * w)
    ty = rng.uniform(-0.04 * h, 0.04 * h)
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    aug_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    aug_lines = transform_lines(lines, M, w, h)
    return aug_img, aug_lines


def random_color(image: np.ndarray, rng: random.Random) -> np.ndarray:
    img = image.astype(np.float32)
    alpha = rng.uniform(0.85, 1.20)
    beta = rng.uniform(-18.0, 18.0)
    img = img * alpha + beta
    gamma = rng.uniform(0.85, 1.20)
    img = 255.0 * np.power(np.clip(img, 0.0, 255.0) / 255.0, gamma)
    hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.85, 1.20), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.90, 1.15), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def random_blur_or_noise(image: np.ndarray, rng: random.Random) -> np.ndarray:
    img = image.copy()
    p = rng.random()
    if p < 0.35:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0.0)
    elif p < 0.70:
        noise_std = rng.uniform(3.0, 10.0)
        n = np.random.normal(0.0, noise_std, size=img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
    else:
        q = rng.randint(70, 92)
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img


def read_annotations(path: Path) -> Dict[str, dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict JSON at {path}, got {type(data).__name__}")
    return data


def load_image_from_dir(
    train_images: Path, key: str, entry: dict
) -> Tuple[np.ndarray | None, Path | None]:
    if entry.get("split_image_path"):
        p = Path(entry["split_image_path"])
        if p.exists():
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            return (img, p) if img is not None else (None, None)
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
        p = train_images / f"{key}{ext}"
        if p.exists():
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            return (img, p) if img is not None else (None, None)
    return None, None


def augment_fold(
    train_images: Path,
    train_json: Path,
    out_images: Path,
    out_json: Path,
    aug_per_image: int = 4,
    seed: int = 42,
    jpg_quality: int = 95,
    dry_run: bool = False,
) -> None:
    """Augment one fold's training set. Called by prepare_folds.py."""
    if not train_images.exists():
        raise FileNotFoundError(f"Train images folder not found: {train_images}")
    if not train_json.exists():
        raise FileNotFoundError(f"Train JSON not found: {train_json}")

    rng = random.Random(seed)
    annotations = read_annotations(train_json)

    if not dry_run:
        out_images.mkdir(parents=True, exist_ok=True)

    out_annotations: Dict[str, dict] = {}
    saved_original = 0
    saved_augmented = 0
    skipped = 0

    for key, entry in annotations.items():
        image, image_path = load_image_from_dir(train_images, key, entry)
        if image is None:
            skipped += 1
            print(f"  [SKIP] {key}")
            continue

        base_out_path = out_images / f"{key}.jpg"
        out_entry = deepcopy(entry)
        out_entry["split_image_path"] = str(base_out_path)
        out_entry["source_train_image_path"] = str(image_path)
        out_annotations[key] = out_entry
        if not dry_run:
            cv2.imwrite(str(base_out_path), image, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
        saved_original += 1

        lines = entry.get("lines", [])
        for i in range(aug_per_image):
            aug_img, aug_lines = random_affine(image, lines, rng)
            aug_img = random_color(aug_img, rng)
            aug_img = random_blur_or_noise(aug_img, rng)

            aug_key = f"{key}_aug{i + 1}"
            aug_out_path = out_images / f"{aug_key}.jpg"

            aug_entry = deepcopy(entry)
            aug_entry["lines"] = aug_lines
            aug_entry["split_image_path"] = str(aug_out_path)
            aug_entry["source_train_image_path"] = str(image_path)
            aug_entry["augmented_from"] = key
            aug_entry["augmentation"] = "affine+color+blur_or_noise_no_mirror"
            out_annotations[aug_key] = aug_entry

            if not dry_run:
                cv2.imwrite(str(aug_out_path), aug_img, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
            saved_augmented += 1

    if not dry_run:
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(out_annotations, f, indent=2, ensure_ascii=False)

    print(
        f"  Originals: {saved_original}  |  Augmented: {saved_augmented}  |  "
        f"Total: {len(out_annotations)}  |  Skipped: {skipped}"
    )


def main() -> None:
    args = parse_args()
    print(f"Augmenting: {args.train_images}")
    augment_fold(
        train_images=args.train_images,
        train_json=args.train_json,
        out_images=args.out_images,
        out_json=args.out_json,
        aug_per_image=args.aug_per_image,
        seed=args.seed,
        jpg_quality=args.jpg_quality,
        dry_run=args.dry_run,
    )
    print(f"Output images: {args.out_images}")
    print(f"Output JSON:   {args.out_json}")


if __name__ == "__main__":
    main()
