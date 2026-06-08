#!/usr/bin/env python3
"""
Evaluate a trained HeatmapNet model on a test set.

Usage:
  python test.py \
    --model   results/train_resnet50/run_XXXX/best_model_overall.pth \
    --backbone resnet50 \
    --test-images dataset/fold_1/test/images \
    --test-json   dataset/fold_1/test_annotations.json

  # Auto-detect backbone from run_config.json (if model is inside a fold dir):
  python test.py \
    --model results/train_resnet50/run_XXXX/fold_1/heatmap_best.pth \
    --test-images dataset/fold_1/test/images \
    --test-json   dataset/fold_1/test_annotations.json

Outputs (saved next to the model by default):
  test_results/
    test_predictions.json   <- per-image predicted keypoints
    test_metrics.json       <- all numeric metrics
    test_metrics.csv        <- same metrics in CSV
    test_summary.txt        <- human-readable report
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants (must match training)
# ---------------------------------------------------------------------------
IMG_SIZE     = 512
HEATMAP_SIZE = 128
NUM_KPS      = 14
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TestDataset(Dataset):
    def __init__(self, keys: List[str], data_dict: Dict[str, dict], img_dir: Path):
        self.keys    = keys
        self.data    = data_dict
        self.img_dir = img_dir
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.keys)

    def _generate_heatmap(self, pts: List[List[float]]) -> np.ndarray:
        heatmaps = np.zeros((NUM_KPS, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
        sigma = 3
        xx, yy = np.meshgrid(np.arange(HEATMAP_SIZE), np.arange(HEATMAP_SIZE))
        for i, (x, y) in enumerate(pts):
            if x < 0 or y < 0:
                continue
            hx = int(x * HEATMAP_SIZE / IMG_SIZE)
            hy = int(y * HEATMAP_SIZE / IMG_SIZE)
            heatmaps[i] = np.exp(-((xx - hx) ** 2 + (yy - hy) ** 2) / (2 * sigma ** 2))
        return heatmaps

    def __getitem__(self, idx: int):
        key  = self.keys[idx]
        item = self.data[key]

        img_path = self.img_dir / f"{key}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        h, w = img.shape[:2]
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        lines = _sort_arc_lines(item["lines"])
        pts   = []
        for line in lines:
            pts.append(line[0])
            pts.append(line[1])
        while len(pts) < NUM_KPS:
            pts.append([-1, -1])
        pts = pts[:NUM_KPS]

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        sx, sy = IMG_SIZE / w, IMG_SIZE / h
        scaled_pts: List[List[float]] = []
        for x, y in pts:
            if x < 0:
                scaled_pts.append([-1.0, -1.0])
            else:
                scaled_pts.append([float(x) * sx, float(y) * sy])

        heatmaps = self._generate_heatmap(scaled_pts)
        return self.normalize(img), torch.tensor(heatmaps), torch.tensor(scaled_pts), key


# ---------------------------------------------------------------------------
# Model definitions — all three backbones
# ---------------------------------------------------------------------------
class HeatmapNetResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=None)
        self.backbone    = nn.Sequential(*list(backbone.children())[:-2])
        self.deconv      = _deconv_head(2048)
        self.final_layer = nn.Conv2d(256, NUM_KPS, 1)

    def forward(self, x):
        return self.final_layer(self.deconv(self.backbone(x)))


class HeatmapNetResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.backbone    = nn.Sequential(*list(backbone.children())[:-2])
        self.deconv      = _deconv_head(512)
        self.final_layer = nn.Conv2d(256, NUM_KPS, 1)

    def forward(self, x):
        return self.final_layer(self.deconv(self.backbone(x)))


class HeatmapNetConvNeXtTiny(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.convnext_tiny(weights=None)
        self.backbone    = backbone.features
        self.deconv      = _deconv_head(768)
        self.final_layer = nn.Conv2d(256, NUM_KPS, 1)

    def forward(self, x):
        return self.final_layer(self.deconv(self.backbone(x)))


def _deconv_head(in_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, 256, 4, 2, 1),
        nn.BatchNorm2d(256), nn.ReLU(),
        nn.ConvTranspose2d(256, 256, 4, 2, 1),
        nn.BatchNorm2d(256), nn.ReLU(),
        nn.ConvTranspose2d(256, 256, 4, 2, 1),
        nn.BatchNorm2d(256), nn.ReLU(),
    )


class HeatmapNetResNet101(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet101(weights=None)
        self.backbone    = nn.Sequential(*list(backbone.children())[:-2])
        self.deconv      = _deconv_head(2048)
        self.final_layer = nn.Conv2d(256, NUM_KPS, 1)

    def forward(self, x):
        return self.final_layer(self.deconv(self.backbone(x)))


class HeatmapNetEfficientNetB3(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.efficientnet_b3(weights=None)
        self.backbone    = backbone.features   # [B, 1536, 16, 16] for 512×512
        self.deconv      = _deconv_head(1536)
        self.final_layer = nn.Conv2d(256, NUM_KPS, 1)

    def forward(self, x):
        return self.final_layer(self.deconv(self.backbone(x)))


BACKBONE_MAP = {
    "resnet18":        HeatmapNetResNet18,
    "resnet50":        HeatmapNetResNet50,
    "resnet101":       HeatmapNetResNet101,
    "convnext_tiny":   HeatmapNetConvNeXtTiny,
    "efficientnet_b3": HeatmapNetEfficientNetB3,
}


# ---------------------------------------------------------------------------
# Arc ordering normalisation (must match training)
# ---------------------------------------------------------------------------
def _arc_cx(line) -> float:
    return 0.5 * (line[0][0] + line[1][0])


def _sort_arc_lines(lines: list) -> list:
    lines = list(lines)
    if len(lines) == 3:
        lines[1:3] = sorted(lines[1:3], key=_arc_cx)
    elif len(lines) == 7:
        lines[5:7] = sorted(lines[5:7], key=_arc_cx)
    return lines


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def decode_heatmap(hm: torch.Tensor) -> List[List[float]]:
    hm_np = hm.cpu().numpy()
    return [[float(x), float(y)]
            for y, x in (np.unravel_index(np.argmax(hm_np[i]), hm_np[i].shape)
                         for i in range(NUM_KPS))]


def decode_heatmap_anatomical(hm: torch.Tensor, valid_kps: int) -> List[List[float]]:
    """Decode heatmap; for arc channels mask image border before argmax.

    Arc endpoints are never at the image edge, so border activations are always
    spurious. Masking 10% border eliminates corner false-peaks without assuming
    any left/right orientation.
    """
    hm_np  = hm.cpu().numpy()
    border = max(1, HEATMAP_SIZE // 10)   # ~13 px for 128-grid

    if valid_kps == 6:
        arc_chs = {2, 3, 4, 5}
    else:
        arc_chs = {10, 11, 12, 13}

    pts = []
    for i in range(NUM_KPS):
        ch = hm_np[i]
        if i in arc_chs:
            masked = ch.copy()
            masked[:border, :]  = 0.0
            masked[-border:, :] = 0.0
            masked[:, :border]  = 0.0
            masked[:, -border:] = 0.0
            y, x = np.unravel_index(np.argmax(masked), masked.shape)
        else:
            y, x = np.unravel_index(np.argmax(ch), ch.shape)
        pts.append([float(x), float(y)])
    return pts


def line_len(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)))


def detect_backbone(model_path: Path) -> str | None:
    """Try to read backbone from run_config.json sitting next to the model."""
    cfg = model_path.parent / "run_config.json"
    if cfg.exists():
        try:
            with cfg.open() as f:
                return json.load(f).get("backbone")
        except Exception:
            pass
    # also check one level up (best_model_overall.json)
    best = model_path.parent / "best_model_overall.json"
    if best.exists():
        try:
            with best.open() as f:
                return json.load(f).get("backbone")
        except Exception:
            pass
    return None


def load_model(model_path: Path, backbone: str) -> nn.Module:
    if backbone not in BACKBONE_MAP:
        raise ValueError(f"Unknown backbone '{backbone}'. Choose from: {list(BACKBONE_MAP)}")
    model = BACKBONE_MAP[backbone]()
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
# Colors: GT=green, Predicted=red, lines connect point pairs
LINE_COLORS = [
    (255, 200,   0),   # scale line  — yellow
    (  0, 200, 255),   # line 1      — cyan
    (255, 100, 200),   # line 2      — pink
    (100, 255, 100),   # line 3      — light green
    (200, 100, 255),   # line 4      — purple
    ( 50, 180, 255),   # line 5      — sky blue
    (255, 160,  50),   # line 6      — orange
]

GT_PT_COLOR   = (0,   220,   0)   # green
PRED_PT_COLOR = (0,    50, 255)   # red-ish blue → red
PRED_LINE_ALPHA = 0.6


def draw_keypoints(
    img_dir: Path,
    key: str,
    gt_pts: List[List[float]],
    pred_pts: List[List[float]],
    out_path: Path,
) -> None:
    img_path = img_dir / f"{key}.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        return
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    overlay = img.copy()

    n_valid_gt = sum(1 for p in gt_pts if p[0] >= 0)

    # Draw GT lines + points
    for li in range(NUM_KPS // 2):
        p1, p2 = gt_pts[2 * li], gt_pts[2 * li + 1]
        if p1[0] < 0 or p2[0] < 0:
            continue
        color = LINE_COLORS[li % len(LINE_COLORS)]
        cv2.line(overlay,
                 (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                 color, 2, cv2.LINE_AA)
        cv2.circle(overlay, (int(p1[0]), int(p1[1])), 5, GT_PT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(overlay, (int(p2[0]), int(p2[1])), 5, GT_PT_COLOR, -1, cv2.LINE_AA)

    # Draw predicted lines + points (thinner, different marker)
    for li in range(NUM_KPS // 2):
        p1, p2 = pred_pts[2 * li], pred_pts[2 * li + 1]
        if gt_pts[2 * li][0] < 0:  # skip if GT says this keypoint doesn't exist
            continue
        color = LINE_COLORS[li % len(LINE_COLORS)]
        cv2.line(overlay,
                 (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                 color, 1, cv2.LINE_AA)
        # predicted points as hollow circles
        cv2.circle(overlay, (int(p1[0]), int(p1[1])), 6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(overlay, (int(p2[0]), int(p2[1])), 6, (0, 0, 255), 2, cv2.LINE_AA)

    # Blend overlay onto original for semi-transparency on lines
    img = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)

    # Legend (top-left)
    cv2.putText(img, "GT",   (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GT_PT_COLOR,   2, cv2.LINE_AA)
    cv2.putText(img, "Pred", (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255),   2, cv2.LINE_AA)
    arch = "Maxilla" if n_valid_gt == 6 else "Mandible"
    cv2.putText(img, arch,   (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    out_dir: Path,
    test_images_dir: Path,
) -> Dict[str, float]:

    predictions = {}
    errors_px   = {"incisor_sum": [], "left_arc": [], "right_arc": []}
    errors_mm   = {"incisor_sum": [], "left_arc": [], "right_arc": []}
    per_image   = []

    with torch.no_grad():
        for imgs, _, gt_pts, keys in tqdm(loader, desc="Evaluating"):
            imgs  = imgs.to(DEVICE)
            preds = model(imgs)

            for i in range(len(imgs)):
                gt          = gt_pts[i].numpy()
                valid_kps   = int(np.sum(gt[:, 0] >= 0))
                pred_coords = decode_heatmap_anatomical(preds[i], valid_kps)
                pred_pts    = [[x * IMG_SIZE / HEATMAP_SIZE, y * IMG_SIZE / HEATMAP_SIZE]
                               for x, y in pred_coords]
                key         = keys[i]

                # GT scale used only for px→mm conversion (not reported as error)
                scale_gt = line_len(gt[0], gt[1])
                px_to_mm = 5.0 / scale_gt if scale_gt > 1e-6 else None

                img_row = {"key": key, "arch": "maxilla" if valid_kps == 6 else "mandible"}

                # After _sort_arc_lines (annotation sort), kp pairs are consistent:
                # Maxilla: kp 2-3 = LEFT arc, kp 4-5 = RIGHT arc
                # Mandible: kp 10-11 = LEFT arc, kp 12-13 = RIGHT arc
                # Use direct kp pairing — x-sort was mispairing when chord endpoints overlap in x
                if valid_kps == 6:
                    l_err = abs(line_len(pred_pts[2], pred_pts[3]) - line_len(gt[2], gt[3]))
                    r_err = abs(line_len(pred_pts[4], pred_pts[5]) - line_len(gt[4], gt[5]))
                    errors_px["left_arc"].append(l_err)
                    errors_px["right_arc"].append(r_err)
                    img_row.update({"left_arc_px": l_err, "right_arc_px": r_err})
                    if px_to_mm:
                        errors_mm["left_arc"].append(l_err * px_to_mm)
                        errors_mm["right_arc"].append(r_err * px_to_mm)
                        img_row.update({"left_arc_mm": l_err*px_to_mm, "right_arc_mm": r_err*px_to_mm})

                else:
                    pred_inc = sum(line_len(pred_pts[2*li], pred_pts[2*li+1]) for li in [1,2,3,4])
                    gt_inc   = sum(line_len(gt[2*li],       gt[2*li+1])       for li in [1,2,3,4])
                    inc_err  = abs(pred_inc - gt_inc)
                    errors_px["incisor_sum"].append(inc_err)
                    img_row["incisor_sum_px"] = inc_err
                    if px_to_mm:
                        errors_mm["incisor_sum"].append(inc_err * px_to_mm)
                        img_row["incisor_sum_mm"] = inc_err * px_to_mm

                    l_err = abs(line_len(pred_pts[10], pred_pts[11]) - line_len(gt[10], gt[11]))
                    r_err = abs(line_len(pred_pts[12], pred_pts[13]) - line_len(gt[12], gt[13]))
                    errors_px["left_arc"].append(l_err)
                    errors_px["right_arc"].append(r_err)
                    img_row.update({"left_arc_px": l_err, "right_arc_px": r_err})
                    if px_to_mm:
                        errors_mm["left_arc"].append(l_err * px_to_mm)
                        errors_mm["right_arc"].append(r_err * px_to_mm)
                        img_row.update({"left_arc_mm": l_err*px_to_mm, "right_arc_mm": r_err*px_to_mm})

                predictions[key] = {"predicted_keypoints": pred_pts, "valid_kps": valid_kps}
                per_image.append(img_row)

                # Visualize GT vs predicted on image
                draw_keypoints(
                    img_dir=test_images_dir,
                    key=key,
                    gt_pts=gt.tolist(),
                    pred_pts=pred_pts,
                    out_path=out_dir / "visualizations" / f"{key}.jpg",
                )

    # Aggregate metrics
    metrics: Dict[str, float] = {}
    for k, vals in errors_px.items():
        if vals:
            metrics[f"{k}_px_mean"] = float(np.mean(vals))
            metrics[f"{k}_px_std"]  = float(np.std(vals))
            metrics[f"{k}_px_med"]  = float(np.median(vals))
    for k, vals in errors_mm.items():
        if vals:
            metrics[f"{k}_mm_mean"] = float(np.mean(vals))
            metrics[f"{k}_mm_std"]  = float(np.std(vals))
            metrics[f"{k}_mm_med"]  = float(np.median(vals))

    # Composite sum3 (incisor + left_arc + right_arc, no scale)
    for unit in ["px", "mm"]:
        keys_ = [f"incisor_sum_{unit}_mean", f"left_arc_{unit}_mean", f"right_arc_{unit}_mean"]
        if all(k in metrics for k in keys_):
            metrics[f"sum_incisor_left_right_{unit}_mean"] = sum(metrics[k] for k in keys_)

    n_max = sum(1 for r in per_image if r["arch"] == "maxilla")
    n_man = sum(1 for r in per_image if r["arch"] == "mandible")
    metrics["n_total"]    = len(per_image)
    metrics["n_maxilla"]  = n_max
    metrics["n_mandible"] = n_man

    # Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "test_predictions.json").open("w") as f:
        json.dump(predictions, f, indent=2)

    with (out_dir / "test_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    # Per-image CSV — sorted by key for consistent ordering
    per_image_sorted = sorted(per_image, key=lambda r: r["key"])
    all_fields = ["key", "arch",
                  "incisor_sum_px", "incisor_sum_mm",
                  "left_arc_px", "left_arc_mm",
                  "right_arc_px", "right_arc_mm"]
    with (out_dir / "test_per_image.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        w.writeheader()
        for row in per_image_sorted:
            w.writerow({k: row.get(k, "") for k in all_fields})

    # Summary text
    summary_lines = [
        "=" * 55,
        "TEST RESULTS",
        f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Images    : {len(per_image)}  (maxilla={n_max}, mandible={n_man})",
        "=" * 55,
        "",
        "--- Pixel errors (mean ± std) ---",
        f"  Incisor sum  : {metrics.get('incisor_sum_px_mean', float('nan')):.2f} ± {metrics.get('incisor_sum_px_std', float('nan')):.2f} px",
        f"  Left arc     : {metrics.get('left_arc_px_mean', float('nan')):.2f} ± {metrics.get('left_arc_px_std', float('nan')):.2f} px",
        f"  Right arc    : {metrics.get('right_arc_px_mean', float('nan')):.2f} ± {metrics.get('right_arc_px_std', float('nan')):.2f} px",
        f"  Sum (inc+L+R): {metrics.get('sum_incisor_left_right_px_mean', float('nan')):.2f} px",
        "",
        "--- mm errors (mean ± std) ---",
        f"  Incisor sum  : {metrics.get('incisor_sum_mm_mean', float('nan')):.3f} ± {metrics.get('incisor_sum_mm_std', float('nan')):.3f} mm",
        f"  Left arc     : {metrics.get('left_arc_mm_mean', float('nan')):.3f} ± {metrics.get('left_arc_mm_std', float('nan')):.3f} mm",
        f"  Right arc    : {metrics.get('right_arc_mm_mean', float('nan')):.3f} ± {metrics.get('right_arc_mm_std', float('nan')):.3f} mm",
        f"  Sum (inc+L+R): {metrics.get('sum_incisor_left_right_mm_mean', float('nan')):.3f} mm",
        "=" * 55,
    ]
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    with (out_dir / "test_summary.txt").open("w") as f:
        f.write(summary_text + "\n")

    return metrics


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained HeatmapNet on a test set.")
    parser.add_argument("--model",       type=Path, required=True,
                        help="Path to .pth weights file")
    parser.add_argument("--backbone",    type=str,  default=None,
                        choices=list(BACKBONE_MAP),
                        help="Backbone name. Auto-detected from run_config.json if omitted.")
    parser.add_argument("--test-images", type=Path, required=True,
                        help="Directory containing test .jpg images")
    parser.add_argument("--test-json",   type=Path, required=True,
                        help="JSON file with ground-truth annotations")
    parser.add_argument("--output-dir",  type=Path, default=None,
                        help="Where to save outputs (default: model_dir/test_results/)")
    parser.add_argument("--batch-size",  type=int,  default=8)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.test_images.exists():
        raise FileNotFoundError(f"Test images dir not found: {args.test_images}")
    if not args.test_json.exists():
        raise FileNotFoundError(f"Test JSON not found: {args.test_json}")

    # Resolve backbone
    backbone = args.backbone or detect_backbone(args.model)
    if backbone is None:
        raise ValueError("Could not auto-detect backbone. Pass --backbone explicitly.")
    print(f"Backbone  : {backbone}")
    print(f"Model     : {args.model}")
    print(f"Images    : {args.test_images}")
    print(f"JSON      : {args.test_json}")
    print(f"Device    : {DEVICE}")

    # Load annotations
    with args.test_json.open() as f:
        data = json.load(f)
    keys = list(data.keys())
    print(f"Samples   : {len(keys)}")

    # Dataset + loader
    dataset = TestDataset(keys, data, args.test_images)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)

    # Load model
    model = load_model(args.model, backbone)

    # Output directory
    out_dir = args.output_dir or (args.model.parent / "test_results")

    # Run evaluation
    evaluate(model, loader, out_dir, test_images_dir=args.test_images)

    print(f"\nOutputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
