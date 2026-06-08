#!/usr/bin/env python3
"""
Train HeatmapNet with EfficientNet-B3 backbone on DSAA Dental dataset (5-fold CV).

Prerequisites:
  Run prepare_folds.py first to create the 5-fold dataset with augmentation.

Usage:
  python train_efficientnet_b3.py
  python train_efficientnet_b3.py --k-folds 5 --epochs 300 --batch-size 8

Outputs (per fold + overall):
  results/train_efficientnet_b3/<run_name>/
    fold_1/  fold_2/  ...  fold_5/
      training_log.csv
      run_config.json
      heatmap_best.pth
      heatmap_last.pth
      test_predictions.json
    kfold_summary.csv
    all_folds_training_log.csv
    loss_curves.png
    best_model_overall.pth
    best_model_overall.json
    run_artifacts.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
IMG_SIZE       = 512
HEATMAP_SIZE   = 128
NUM_KPS        = 14

BATCH_SIZE     = 8
EPOCHS         = 300
LR             = 1e-4
WEIGHT_DECAY   = 1e-4
PATIENCE       = 20
SOFTARGMAX_TEMP = 12.0

COORD_LOSS_W        = 0.02
LENGTH_LOSS_W       = 0.01
ARC_SIDE_LOSS_W     = 0.008
ARC_CENTER_ALIGN_W  = 0.005
ARC_CONTRAST_W      = 0.004
ARC_MARGIN_PX_DEFAULT = 12.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DSAA_ROOT         = Path(__file__).resolve().parent
DEFAULT_DATASET   = DSAA_ROOT / "dataset"
DEFAULT_SAVE_ROOT = DSAA_ROOT / "results" / "train_efficientnet_b3"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class HeatmapDentalDataset(Dataset):
    def __init__(self, keys: List[str], data_dict: Dict[str, dict], img_dir: Path):
        self.keys     = keys
        self.data     = data_dict
        self.img_dir  = img_dir
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

        lines = item["lines"]
        pts   = []
        for line in lines:
            pts.append(line[0])
            pts.append(line[1])
        while len(pts) < NUM_KPS:
            pts.append([-1, -1])
        pts = pts[:NUM_KPS]

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        sx  = IMG_SIZE / w
        sy  = IMG_SIZE / h
        scaled_pts: List[List[float]] = []
        for x, y in pts:
            if x < 0:
                scaled_pts.append([-1.0, -1.0])
            else:
                scaled_pts.append([float(x) * sx, float(y) * sy])

        heatmaps = self._generate_heatmap(scaled_pts)
        return self.normalize(img), torch.tensor(heatmaps), torch.tensor(scaled_pts), key


# ---------------------------------------------------------------------------
# Model — EfficientNet-B3 backbone
# ---------------------------------------------------------------------------
class HeatmapNet(nn.Module):
    def __init__(self):
        super().__init__()
        enet = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        self.backbone = enet.features  # output: (B, 1536, 16, 16) for 512x512 input
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(1536, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.final_layer = nn.Conv2d(256, NUM_KPS, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.deconv(x)
        return self.final_layer(x)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def decode_heatmap(hm: torch.Tensor) -> List[List[float]]:
    hm_np  = hm.cpu().numpy()
    coords = []
    for i in range(NUM_KPS):
        y, x = np.unravel_index(np.argmax(hm_np[i]), hm_np[i].shape)
        coords.append([float(x), float(y)])
    return coords


def softargmax_coords(
    logits: torch.Tensor, grid_x: torch.Tensor, grid_y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs   = torch.softmax(logits.view(logits.size(0), logits.size(1), -1) * SOFTARGMAX_TEMP, dim=2).view_as(logits)
    coord_x = (probs * grid_x).sum(dim=(2, 3)) * IMG_SIZE / HEATMAP_SIZE
    coord_y = (probs * grid_y).sum(dim=(2, 3)) * IMG_SIZE / HEATMAP_SIZE
    return coord_x, coord_y


def line_len(p1: List[float], p2: List[float]) -> float:
    return float(np.linalg.norm(np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)))


def find_train_assets(fold_root: Path) -> Tuple[Path, Path]:
    aug_img_dir = fold_root / "train_augmented" / "images"
    aug_ann     = fold_root / "train_annotations_augmented.json"
    if aug_img_dir.exists() and aug_ann.exists():
        return aug_img_dir, aug_ann
    return fold_root / "train" / "images", fold_root / "train_annotations.json"


def infer_arch(entry: dict) -> str:
    arch = str(entry.get("arch", "")).strip().lower()
    if arch in {"maxilla", "mandible"}:
        return arch
    return "maxilla" if len(entry.get("lines", [])) <= 3 else "mandible"


def estimate_arc_margin_px(
    train_data: Dict[str, dict], default_margin: float = ARC_MARGIN_PX_DEFAULT
) -> float:
    gaps_max, gaps_man = [], []
    for item in train_data.values():
        lines = item.get("lines", [])
        if len(lines) == 3:
            a, b   = lines[1], lines[2]
            bucket = gaps_max
        elif len(lines) == 7:
            a, b   = lines[5], lines[6]
            bucket = gaps_man
        else:
            continue
        ax = 0.5 * (a[0][0] + a[1][0])
        bx = 0.5 * (b[0][0] + b[1][0])
        bucket.append(abs(ax - bx))

    pools = []
    if gaps_max:
        pools.append(float(np.percentile(np.array(gaps_max, dtype=np.float32), 10)))
    if gaps_man:
        pools.append(float(np.percentile(np.array(gaps_man, dtype=np.float32), 10)))
    if not pools:
        return default_margin

    margin = 0.12 * min(pools)
    return float(np.clip(margin, 6.0, 14.0))


def arc_losses(
    coord_x: torch.Tensor, gt_pts: torch.Tensor, arc_margin_px: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    side_order_loss   = torch.tensor(0.0, device=coord_x.device)
    center_align_loss = torch.tensor(0.0, device=coord_x.device)
    contrast_loss     = torch.tensor(0.0, device=coord_x.device)

    # maxilla arcs — indices 2..5
    max_mask = (gt_pts[:, :, 0] >= 0).sum(dim=1) == 6
    if max_mask.any():
        pred_arcs = torch.stack([0.5 * (coord_x[:, 2] + coord_x[:, 3]),
                                 0.5 * (coord_x[:, 4] + coord_x[:, 5])], dim=1)
        gt_arcs   = torch.stack([0.5 * (gt_pts[:, 2, 0] + gt_pts[:, 3, 0]),
                                 0.5 * (gt_pts[:, 4, 0] + gt_pts[:, 5, 0])], dim=1)
        pred_sorted, _ = torch.sort(pred_arcs, dim=1)
        gt_sorted,   _ = torch.sort(gt_arcs,   dim=1)
        lpx, rpx = pred_sorted[:, 0], pred_sorted[:, 1]
        lgx, rgx = gt_sorted[:, 0],  gt_sorted[:, 1]

        side    = F.relu(lpx - rpx + arc_margin_px)
        align   = F.smooth_l1_loss(lpx, lgx, reduction="none") + F.smooth_l1_loss(rpx, rgx, reduction="none")
        d_l_own  = torch.abs(lpx - lgx);  d_l_other = torch.abs(lpx - rgx)
        d_r_own  = torch.abs(rpx - rgx);  d_r_other = torch.abs(rpx - lgx)
        contrast = F.relu(d_l_own + arc_margin_px - d_l_other) + F.relu(d_r_own + arc_margin_px - d_r_other)

        n = max_mask.sum() + 1e-6
        side_order_loss   = side_order_loss   + (side    * max_mask).sum() / n
        center_align_loss = center_align_loss + (align   * max_mask).sum() / n
        contrast_loss     = contrast_loss     + (contrast * max_mask).sum() / n

    # mandible arcs — indices 10..13
    man_mask = (gt_pts[:, :, 0] >= 0).sum(dim=1) == 14
    if man_mask.any():
        pred_arcs = torch.stack([0.5 * (coord_x[:, 10] + coord_x[:, 11]),
                                 0.5 * (coord_x[:, 12] + coord_x[:, 13])], dim=1)
        gt_arcs   = torch.stack([0.5 * (gt_pts[:, 10, 0] + gt_pts[:, 11, 0]),
                                 0.5 * (gt_pts[:, 12, 0] + gt_pts[:, 13, 0])], dim=1)
        pred_sorted, _ = torch.sort(pred_arcs, dim=1)
        gt_sorted,   _ = torch.sort(gt_arcs,   dim=1)
        lpx, rpx = pred_sorted[:, 0], pred_sorted[:, 1]
        lgx, rgx = gt_sorted[:, 0],  gt_sorted[:, 1]

        side    = F.relu(lpx - rpx + arc_margin_px)
        align   = F.smooth_l1_loss(lpx, lgx, reduction="none") + F.smooth_l1_loss(rpx, rgx, reduction="none")
        d_l_own  = torch.abs(lpx - lgx);  d_l_other = torch.abs(lpx - rgx)
        d_r_own  = torch.abs(rpx - rgx);  d_r_other = torch.abs(rpx - lgx)
        contrast = F.relu(d_l_own + arc_margin_px - d_l_other) + F.relu(d_r_own + arc_margin_px - d_r_other)

        n = man_mask.sum() + 1e-6
        side_order_loss   = side_order_loss   + (side    * man_mask).sum() / n
        center_align_loss = center_align_loss + (align   * man_mask).sum() / n
        contrast_loss     = contrast_loss     + (contrast * man_mask).sum() / n

    return side_order_loss / IMG_SIZE, center_align_loss / IMG_SIZE, contrast_loss / IMG_SIZE


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model: nn.Module, loader: DataLoader, pred_json_path: Path) -> Dict[str, float]:
    model.eval()
    predictions  = {}
    errors_px    = {"scale": [], "incisor_sum": [], "left_arc": [], "right_arc": []}
    errors_mm    = {"scale": [], "incisor_sum": [], "left_arc": [], "right_arc": []}

    with torch.no_grad():
        for imgs, _, gt_pts, keys in tqdm(loader, leave=False):
            imgs  = imgs.to(DEVICE)
            preds = model(imgs)
            for i in range(len(imgs)):
                pred_coords = decode_heatmap(preds[i])
                pred_pts    = [[x * IMG_SIZE / HEATMAP_SIZE, y * IMG_SIZE / HEATMAP_SIZE]
                               for x, y in pred_coords]
                gt          = gt_pts[i].numpy()
                valid_kps   = int(np.sum(gt[:, 0] >= 0))

                scale_pred    = line_len(pred_pts[0], pred_pts[1])
                scale_gt      = line_len(gt[0],       gt[1])
                scale_err_px  = abs(scale_pred - scale_gt)
                errors_px["scale"].append(scale_err_px)
                px_to_mm = 5.0 / scale_gt if scale_gt > 1e-6 else None
                if px_to_mm:
                    errors_mm["scale"].append(scale_err_px * px_to_mm)

                if valid_kps == 6:
                    pred_arcs, gt_arcs = [], []
                    for li in [1, 2]:
                        p1, p2 = pred_pts[2 * li], pred_pts[2 * li + 1]
                        g1, g2 = gt[2 * li],       gt[2 * li + 1]
                        pred_arcs.append(((p1[0] + p2[0]) / 2.0, line_len(p1, p2)))
                        gt_arcs.append(  ((g1[0] + g2[0]) / 2.0, line_len(g1, g2)))
                    pred_arcs.sort(key=lambda a: a[0]);  gt_arcs.sort(key=lambda a: a[0])
                    left_err_px  = abs(pred_arcs[0][1] - gt_arcs[0][1])
                    right_err_px = abs(pred_arcs[1][1] - gt_arcs[1][1])
                    errors_px["left_arc"].append(left_err_px)
                    errors_px["right_arc"].append(right_err_px)
                    if px_to_mm:
                        errors_mm["left_arc"].append(left_err_px  * px_to_mm)
                        errors_mm["right_arc"].append(right_err_px * px_to_mm)
                else:
                    pred_inc = sum(line_len(pred_pts[2*li], pred_pts[2*li+1]) for li in [1, 2, 3, 4])
                    gt_inc   = sum(line_len(gt[2*li],       gt[2*li+1])       for li in [1, 2, 3, 4])
                    inc_err_px = abs(pred_inc - gt_inc)
                    errors_px["incisor_sum"].append(inc_err_px)
                    if px_to_mm:
                        errors_mm["incisor_sum"].append(inc_err_px * px_to_mm)

                    pred_arcs, gt_arcs = [], []
                    for li in [5, 6]:
                        p1, p2 = pred_pts[2 * li], pred_pts[2 * li + 1]
                        g1, g2 = gt[2 * li],       gt[2 * li + 1]
                        pred_arcs.append(((p1[0] + p2[0]) / 2.0, line_len(p1, p2)))
                        gt_arcs.append(  ((g1[0] + g2[0]) / 2.0, line_len(g1, g2)))
                    pred_arcs.sort(key=lambda a: a[0]);  gt_arcs.sort(key=lambda a: a[0])
                    left_err_px  = abs(pred_arcs[0][1] - gt_arcs[0][1])
                    right_err_px = abs(pred_arcs[1][1] - gt_arcs[1][1])
                    errors_px["left_arc"].append(left_err_px)
                    errors_px["right_arc"].append(right_err_px)
                    if px_to_mm:
                        errors_mm["left_arc"].append(left_err_px  * px_to_mm)
                        errors_mm["right_arc"].append(right_err_px * px_to_mm)

                predictions[keys[i]] = {"predicted_keypoints": pred_pts}

    with pred_json_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    means: Dict[str, float] = {}
    for k, vals in errors_px.items():
        if vals:
            means[f"{k}_px"] = float(np.mean(vals))
    for k, vals in errors_mm.items():
        if vals:
            means[f"{k}_mm"] = float(np.mean(vals))
    if all(x in means for x in ["incisor_sum_px", "left_arc_px", "right_arc_px"]):
        means["sum_incisor_left_right_px"] = means["incisor_sum_px"] + means["left_arc_px"] + means["right_arc_px"]
    if all(x in means for x in ["incisor_sum_mm", "left_arc_mm", "right_arc_mm"]):
        means["sum_incisor_left_right_mm"] = means["incisor_sum_mm"] + means["left_arc_mm"] + means["right_arc_mm"]
    return means


# ---------------------------------------------------------------------------
# Training — one fold
# ---------------------------------------------------------------------------
def train_one_fold(
    fold_name: str,
    fold_root: Path,
    save_root: Path,
    seed: int,
) -> Dict[str, float]:
    train_img_dir, train_ann_file = find_train_assets(fold_root)
    test_img_dir  = fold_root / "test"  / "images"
    test_ann_file = fold_root / "test_annotations.json"
    for p in [train_img_dir, train_ann_file, test_img_dir, test_ann_file]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing asset: {p}. Run prepare_folds.py first.")

    with train_ann_file.open("r", encoding="utf-8") as f:
        train_data = json.load(f)
    with test_ann_file.open("r", encoding="utf-8") as f:
        test_data  = json.load(f)

    arc_margin_px = estimate_arc_margin_px(train_data)
    print(f"[{fold_name}] ARC_MARGIN_PX={arc_margin_px:.2f}")

    train_keys = list(train_data.keys())
    test_keys  = list(test_data.keys())
    train_ds   = HeatmapDentalDataset(train_keys, train_data, train_img_dir)
    test_ds    = HeatmapDentalDataset(test_keys,  test_data,  test_img_dir)

    arch_labels  = [infer_arch(train_data[k]) for k in train_keys]
    count_max    = max(1, sum(1 for a in arch_labels if a == "maxilla"))
    count_man    = max(1, sum(1 for a in arch_labels if a == "mandible"))
    weights      = [1.0 / (count_max if a == "maxilla" else count_man) for a in arch_labels]
    sampler      = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )
    print(f"[{fold_name}] Train: {len(train_ds)} samples (maxilla={count_max}, mandible={count_man})")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    fold_save      = save_root / fold_name
    fold_save.mkdir(parents=True, exist_ok=True)
    log_csv        = fold_save / "training_log.csv"
    best_model_path = fold_save / "heatmap_best.pth"
    last_model_path = fold_save / "heatmap_last.pth"
    pred_json_path  = fold_save / "test_predictions.json"
    config_json     = fold_save / "run_config.json"

    with config_json.open("w", encoding="utf-8") as f:
        json.dump({
            "fold":             fold_name,
            "backbone":         "efficientnet_b3",
            "seed":             seed,
            "train_img_dir":    str(train_img_dir),
            "train_ann_file":   str(train_ann_file),
            "test_img_dir":     str(test_img_dir),
            "test_ann_file":    str(test_ann_file),
            "save_dir":         str(fold_save),
            "batch_size":       BATCH_SIZE,
            "epochs":           EPOCHS,
            "lr":               LR,
            "weight_decay":     WEIGHT_DECAY,
            "patience":         PATIENCE,
            "softargmax_temp":  SOFTARGMAX_TEMP,
            "coord_loss_w":     COORD_LOSS_W,
            "length_loss_w":    LENGTH_LOSS_W,
            "arc_side_loss_w":  ARC_SIDE_LOSS_W,
            "arc_center_align_w": ARC_CENTER_ALIGN_W,
            "arc_contrast_w":   ARC_CONTRAST_W,
            "arc_margin_px":    arc_margin_px,
            "img_size":         IMG_SIZE,
            "heatmap_size":     HEATMAP_SIZE,
            "num_kps":          NUM_KPS,
            "device":           str(DEVICE),
            "train_samples":    len(train_ds),
            "test_samples":     len(test_ds),
        }, f, indent=2)

    with log_csv.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "learning_rate"])

    model     = HeatmapNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val        = float("inf")
    patience_counter = 0

    grid_y, grid_x = torch.meshgrid(
        torch.arange(HEATMAP_SIZE, device=DEVICE),
        torch.arange(HEATMAP_SIZE, device=DEVICE),
        indexing="ij",
    )
    grid_x = grid_x.float()
    grid_y = grid_y.float()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for imgs, heatmaps, gt_pts, _ in train_loader:
            imgs, heatmaps, gt_pts = imgs.to(DEVICE), heatmaps.to(DEVICE), gt_pts.to(DEVICE)
            optimizer.zero_grad()

            preds      = model(imgs)
            diff       = (preds - heatmaps) ** 2
            valid_mask = (heatmaps.sum(dim=(2, 3)) > 0).float().unsqueeze(2).unsqueeze(3).expand_as(diff)
            heatmap_loss = (diff * valid_mask).sum() / (valid_mask.sum() + 1e-6)

            coord_x, coord_y = softargmax_coords(preds, grid_x, grid_y)
            pred_xy   = torch.stack([coord_x, coord_y], dim=2)
            valid_kp  = (gt_pts[:, :, 0] >= 0)
            coord_res = F.smooth_l1_loss(pred_xy, gt_pts, reduction="none").sum(dim=2) / IMG_SIZE
            coord_loss = (coord_res * valid_kp.float()).sum() / (valid_kp.sum() + 1e-6)

            length_loss = torch.tensor(0.0, device=DEVICE)
            line_count  = torch.tensor(0.0, device=DEVICE)
            for j in range(0, NUM_KPS, 2):
                valid_line = gt_pts[:, j, 0] >= 0
                if not valid_line.any():
                    continue
                pred_len = torch.sqrt((coord_x[:, j] - coord_x[:, j+1])**2 + (coord_y[:, j] - coord_y[:, j+1])**2 + 1e-6)
                gt_len   = torch.sqrt((gt_pts[:, j, 0] - gt_pts[:, j+1, 0])**2 + (gt_pts[:, j, 1] - gt_pts[:, j+1, 1])**2 + 1e-6)
                length_loss = length_loss + ((((pred_len - gt_len) / IMG_SIZE) ** 2) * valid_line.float()).sum()
                line_count  = line_count  + valid_line.float().sum()
            length_loss = length_loss / (line_count + 1e-6)

            side_loss, align_loss, ctrst_loss = arc_losses(coord_x, gt_pts, arc_margin_px)

            loss = (heatmap_loss
                    + COORD_LOSS_W       * coord_loss
                    + LENGTH_LOSS_W      * length_loss
                    + ARC_SIDE_LOSS_W    * side_loss
                    + ARC_CENTER_ALIGN_W * align_loss
                    + ARC_CONTRAST_W     * ctrst_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, heatmaps, _, _ in test_loader:
                imgs, heatmaps = imgs.to(DEVICE), heatmaps.to(DEVICE)
                preds      = model(imgs)
                diff       = (preds - heatmaps) ** 2
                valid_mask = (heatmaps.sum(dim=(2, 3)) > 0).float().unsqueeze(2).unsqueeze(3).expand_as(diff)
                val_loss  += ((diff * valid_mask).sum() / (valid_mask.sum() + 1e-6)).item()
        val_loss /= max(1, len(test_loader))

        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[{fold_name}] Epoch {epoch:03d} | Train {train_loss:.6f} | Val {val_loss:.6f} | LR {lr_now:.2e}")
        with log_csv.open("a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, lr_now])

        if val_loss < best_val:
            best_val         = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"[{fold_name}] Early stopping at epoch {epoch}")
            break

    torch.save(model.state_dict(), last_model_path)
    best_model = HeatmapNet().to(DEVICE)
    best_model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    metrics = evaluate(best_model, test_loader, pred_json_path)
    metrics["best_val_loss"] = float(best_val)
    metrics["fold"]          = fold_name
    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_loss_curves(all_epochs_csv: Path, out_png: Path) -> None:
    rows = []
    with all_epochs_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    "fold":       row.get("fold", ""),
                    "epoch":      int(row.get("epoch", 0)),
                    "train_loss": float(row.get("train_loss", "nan")),
                    "val_loss":   float(row.get("val_loss",   "nan")),
                })
            except (TypeError, ValueError):
                continue
    if not rows:
        return

    import matplotlib.pyplot as plt
    fold_names = sorted({r["fold"] for r in rows if r["fold"]})
    fig, axes  = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for fold in fold_names:
        fr = sorted((r for r in rows if r["fold"] == fold), key=lambda r: r["epoch"])
        axes[0].plot([r["epoch"] for r in fr], [r["train_loss"] for r in fr], linewidth=2, label=fold)
        axes[1].plot([r["epoch"] for r in fr], [r["val_loss"]   for r in fr], linewidth=2, label=fold)
    axes[0].set_title("Train Loss"); axes[1].set_title("Validation Loss")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.grid(True, alpha=0.3)
    if len(fold_names) <= 8:
        axes[0].legend(loc="best"); axes[1].legend(loc="best")
    fig.suptitle("EfficientNet-B3 — DSAA Dental", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HeatmapNet (EfficientNet-B3) on DSAA Dental — 5-fold CV.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET,
                        help="Path to prepared fold dataset (created by prepare_folds.py)")
    parser.add_argument("--save-root",    type=Path, default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--k-folds",      type=int,  default=5)
    parser.add_argument("--seed",         type=int,  default=42)
    parser.add_argument("--run-name",     type=str,  default=None,
                        help="Output subdirectory name (default: run_YYYYmmdd_HHMMSS)")
    parser.add_argument("--epochs",       type=int,  default=EPOCHS)
    parser.add_argument("--batch-size",   type=int,  default=BATCH_SIZE)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    global EPOCHS, BATCH_SIZE
    EPOCHS     = args.epochs
    BATCH_SIZE = args.batch_size

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_name = args.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = args.save_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for i in range(1, args.k_folds + 1):
        fold_name = f"fold_{i}"
        fold_root = args.dataset_root / fold_name
        if not fold_root.exists():
            raise FileNotFoundError(f"{fold_root} not found. Run prepare_folds.py first.")
        print(f"\n{'='*60}")
        print(f"Training {fold_name} — EfficientNet-B3")
        print(f"{'='*60}")
        metrics = train_one_fold(fold_name, fold_root, run_root, seed=args.seed + i)
        summary_rows.append(metrics)
        print(f"[{fold_name}] Metrics: {metrics}")

    fields = ["fold", "best_val_loss", "scale_px", "incisor_sum_px", "left_arc_px",
              "right_arc_px", "sum_incisor_left_right_px", "scale_mm", "incisor_sum_mm",
              "left_arc_mm", "right_arc_mm", "sum_incisor_left_right_mm"]
    summary_csv = run_root / "kfold_summary.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: row.get(k, "") for k in fields})

    all_epochs_csv = run_root / "all_folds_training_log.csv"
    with all_epochs_csv.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["fold", "epoch", "train_loss", "val_loss", "learning_rate"])
        for i in range(1, args.k_folds + 1):
            fold_log = run_root / f"fold_{i}" / "training_log.csv"
            if not fold_log.exists():
                continue
            with fold_log.open("r", encoding="utf-8") as fin:
                for row in csv.DictReader(fin):
                    writer.writerow([f"fold_{i}", row.get("epoch"), row.get("train_loss"),
                                     row.get("val_loss"), row.get("learning_rate")])

    print(f"\n{'='*60}")
    print("PER-FOLD RESULTS — EfficientNet-B3")
    print(f"{'='*60}")
    for row in summary_rows:
        print(f"{row.get('fold')}: "
              f"best_val={row.get('best_val_loss', float('nan')):.4f} | "
              f"scale_px={row.get('scale_px', float('nan')):.3f} | "
              f"incisor_sum_px={row.get('incisor_sum_px', float('nan')):.3f} | "
              f"left_arc_px={row.get('left_arc_px', float('nan')):.3f} | "
              f"right_arc_px={row.get('right_arc_px', float('nan')):.3f} | "
              f"sum3_mm={row.get('sum_incisor_left_right_mm', float('nan')):.3f}")

    loss_curve_png = run_root / "loss_curves.png"
    plot_loss_curves(all_epochs_csv, loss_curve_png)

    if summary_rows:
        has_sum3_mm = all("sum_incisor_left_right_mm" in r and r["sum_incisor_left_right_mm"] != ""
                          for r in summary_rows)
        if has_sum3_mm:
            best_row      = min(summary_rows, key=lambda r: float(r["sum_incisor_left_right_mm"]))
            criterion      = "sum_incisor_left_right_mm (lower is better)"
            criterion_val  = float(best_row["sum_incisor_left_right_mm"])
        else:
            best_row      = min(summary_rows, key=lambda r: float(r["best_val_loss"]))
            criterion      = "best_val_loss (lower is better)"
            criterion_val  = float(best_row["best_val_loss"])

        best_fold  = str(best_row["fold"])
        src_model  = run_root / best_fold / "heatmap_best.pth"
        dst_model  = run_root / "best_model_overall.pth"
        if src_model.exists():
            shutil.copy2(src_model, dst_model)

        best_meta = {
            "best_fold":          best_fold,
            "backbone":           "efficientnet_b3",
            "selection_criterion": criterion,
            "criterion_value":    criterion_val,
            "source_model":       str(src_model),
            "saved_model":        str(dst_model),
        }
        with (run_root / "best_model_overall.json").open("w", encoding="utf-8") as f:
            json.dump(best_meta, f, indent=2)
        print(f"\nBest model: {dst_model} (from {best_fold}, {criterion}={criterion_val:.6f})")

    fold_artifacts = [
        {
            "fold":                 f"fold_{i}",
            "fold_dir":             str(run_root / f"fold_{i}"),
            "training_log_csv":     str(run_root / f"fold_{i}" / "training_log.csv"),
            "config_json":          str(run_root / f"fold_{i}" / "run_config.json"),
            "model_best":           str(run_root / f"fold_{i}" / "heatmap_best.pth"),
            "model_last":           str(run_root / f"fold_{i}" / "heatmap_last.pth"),
            "test_predictions_json": str(run_root / f"fold_{i}" / "test_predictions.json"),
        }
        for i in range(1, args.k_folds + 1)
    ]
    artifacts = {
        "run_root":                  str(run_root),
        "backbone":                  "efficientnet_b3",
        "kfold_summary_csv":         str(summary_csv),
        "all_folds_training_log_csv": str(all_epochs_csv),
        "loss_curves_png":            str(loss_curve_png),
        "best_model_overall_json":    str(run_root / "best_model_overall.json"),
        "best_model_overall_pth":     str(run_root / "best_model_overall.pth"),
        "fold_artifacts":             fold_artifacts,
    }
    with (run_root / "run_artifacts.json").open("w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2)

    print(f"\nOutput: {run_root}")


if __name__ == "__main__":
    main()
