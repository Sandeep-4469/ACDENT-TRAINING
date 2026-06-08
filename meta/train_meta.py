#!/usr/bin/env python3
"""
Meta-learning (Reptile) for dental landmark detection.

Algorithm:
  Phase 1 — Warmup (WARMUP_EPOCHS): standard mini-batch training,
             heatmap loss only. Establishes stable spatial features.
  Phase 2 — Reptile meta-learning: each "epoch" runs META_TASKS_PER_EPOCH
             patient-group tasks. For each task:
               (a) save meta-params θ₀
               (b) run K_INNER SGD steps on task images → θ_K
               (c) Reptile update: θ ← θ + EPSILON·(θ_K − θ₀)
             Then one standard pass over all data to maintain convergence.

Patient grouping: images with the same name prefix before _mandible/_maxilla
are treated as one patient. Each task = META_TASK_PATIENTS random patients.

Usage:
  python meta/train_meta.py --k-folds 5
  python meta/train_meta.py --k-folds 1
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
IMG_SIZE        = 512
HEATMAP_SIZE    = 128
NUM_KPS         = 14

BATCH_SIZE      = 8
EPOCHS          = 300
LR              = 1e-4
WEIGHT_DECAY    = 1e-4
PATIENCE        = 20
SOFTARGMAX_TEMP = 12.0

COORD_LOSS_W       = 0.02
LENGTH_LOSS_W      = 0.01
ARC_SIDE_LOSS_W    = 0.008
ARC_CENTER_ALIGN_W = 0.005
ARC_CONTRAST_W     = 0.004
ARC_MARGIN_PX_DEFAULT = 12.0

# Meta-learning (Reptile)
WARMUP_EPOCHS        = 25    # standard heatmap-only warmup before meta phase
META_TASKS_PER_EPOCH = 8     # Reptile tasks per epoch
META_TASK_PATIENTS   = 6     # patients sampled per task (~12 images)
K_INNER              = 5     # inner SGD steps per task
LR_INNER             = 2e-4  # inner loop learning rate
EPSILON              = 0.4   # Reptile step size

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DSAA_ROOT         = Path(__file__).resolve().parent.parent
DEFAULT_DATASET   = DSAA_ROOT / "dataset"
DEFAULT_SAVE_ROOT = DSAA_ROOT / "results" / "meta"


# ---------------------------------------------------------------------------
# Arc ordering normalisation (same as train_resnet50.py)
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
# Patient grouping for Reptile tasks
# ---------------------------------------------------------------------------
def get_patient_id(key: str) -> str:
    m = re.match(r'^(.+?)_(?:mandible|maxilla)', key, re.IGNORECASE)
    return m.group(1) if m else key


def build_patient_groups(keys: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for k in keys:
        pid = get_patient_id(k)
        groups.setdefault(pid, []).append(k)
    return groups


def sample_task_keys(patient_groups: Dict[str, List[str]], n_patients: int) -> List[str]:
    pids = random.sample(list(patient_groups.keys()), min(n_patients, len(patient_groups)))
    keys: List[str] = []
    for pid in pids:
        keys.extend(patient_groups[pid])
    return keys


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class HeatmapDentalDataset(Dataset):
    def __init__(self, keys: List[str], data_dict: Dict[str, dict], img_dir: Path):
        self.keys      = keys
        self.data      = data_dict
        self.img_dir   = img_dir
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
# Model
# ---------------------------------------------------------------------------
class HeatmapNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(2048, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d( 256, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d( 256, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.final_layer = nn.Conv2d(256, NUM_KPS, 1)

    def forward(self, x):
        return self.final_layer(self.deconv(self.backbone(x)))


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


def line_len(p1, p2) -> float:
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


def estimate_arc_margin_px(train_data: Dict[str, dict], default: float = ARC_MARGIN_PX_DEFAULT) -> float:
    gaps = []
    for item in train_data.values():
        lines = item.get("lines", [])
        if len(lines) == 3:
            a, b = lines[1], lines[2]
        elif len(lines) == 7:
            a, b = lines[5], lines[6]
        else:
            continue
        gaps.append(abs(0.5*(a[0][0]+a[1][0]) - 0.5*(b[0][0]+b[1][0])))
    if not gaps:
        return default
    return float(np.clip(0.12 * np.percentile(gaps, 10), 6.0, 14.0))


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def compute_loss(
    preds: torch.Tensor,
    heatmaps: torch.Tensor,
    gt_pts: torch.Tensor,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    arc_margin_px: float,
    heatmap_only: bool = False,
) -> torch.Tensor:
    diff         = (preds - heatmaps) ** 2
    valid_mask   = (heatmaps.sum(dim=(2, 3)) > 0).float().unsqueeze(2).unsqueeze(3).expand_as(diff)
    heatmap_loss = (diff * valid_mask).sum() / (valid_mask.sum() + 1e-6)

    if heatmap_only:
        return heatmap_loss

    coord_x, coord_y = softargmax_coords(preds, grid_x, grid_y)
    pred_xy   = torch.stack([coord_x, coord_y], dim=2)
    valid_kp  = (gt_pts[:, :, 0] >= 0)
    coord_res = F.smooth_l1_loss(pred_xy, gt_pts, reduction="none").sum(dim=2) / IMG_SIZE
    coord_loss = (coord_res * valid_kp.float()).sum() / (valid_kp.sum() + 1e-6)

    length_loss = torch.tensor(0.0, device=preds.device)
    line_count  = torch.tensor(0.0, device=preds.device)
    for j in range(0, NUM_KPS, 2):
        vl = gt_pts[:, j, 0] >= 0
        if not vl.any():
            continue
        pl = torch.sqrt((coord_x[:,j]-coord_x[:,j+1])**2 + (coord_y[:,j]-coord_y[:,j+1])**2 + 1e-6)
        gl = torch.sqrt((gt_pts[:,j,0]-gt_pts[:,j+1,0])**2 + (gt_pts[:,j,1]-gt_pts[:,j+1,1])**2 + 1e-6)
        length_loss = length_loss + (((pl-gl)/IMG_SIZE)**2 * vl.float()).sum()
        line_count  = line_count  + vl.float().sum()
    length_loss = length_loss / (line_count + 1e-6)

    side_loss = center_loss = contrast_loss = torch.tensor(0.0, device=preds.device)
    for mask, li, ri in [
        ((gt_pts[:,:,0]>=0).sum(1)==6,  2, 4),
        ((gt_pts[:,:,0]>=0).sum(1)==14, 10, 12),
    ]:
        if not mask.any():
            continue
        pa = torch.stack([0.5*(coord_x[:,li]+coord_x[:,li+1]), 0.5*(coord_x[:,ri]+coord_x[:,ri+1])], 1)
        ga = torch.stack([0.5*(gt_pts[:,li,0]+gt_pts[:,li+1,0]), 0.5*(gt_pts[:,ri,0]+gt_pts[:,ri+1,0])], 1)
        ps, _ = torch.sort(pa, 1); gs, _ = torch.sort(ga, 1)
        lpx, rpx = ps[:,0], ps[:,1]; lgx, rgx = gs[:,0], gs[:,1]
        n = mask.sum() + 1e-6
        side_loss    = side_loss    + (F.relu(lpx-rpx+arc_margin_px)*mask).sum()/n/IMG_SIZE
        center_loss  = center_loss  + ((F.smooth_l1_loss(lpx,lgx,"none")+F.smooth_l1_loss(rpx,rgx,"none"))*mask).sum()/n/IMG_SIZE
        dl_own,dl_oth = torch.abs(lpx-lgx), torch.abs(lpx-rgx)
        dr_own,dr_oth = torch.abs(rpx-rgx), torch.abs(rpx-lgx)
        contrast_loss = contrast_loss + ((F.relu(dl_own+arc_margin_px-dl_oth)+F.relu(dr_own+arc_margin_px-dr_oth))*mask).sum()/n/IMG_SIZE

    return (heatmap_loss
            + COORD_LOSS_W       * coord_loss
            + LENGTH_LOSS_W      * length_loss
            + ARC_SIDE_LOSS_W    * side_loss
            + ARC_CENTER_ALIGN_W * center_loss
            + ARC_CONTRAST_W     * contrast_loss)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model: nn.Module, loader: DataLoader, pred_json_path: Path) -> Dict[str, float]:
    model.eval()
    predictions = {}
    errors_px   = {"scale": [], "incisor_sum": [], "left_arc": [], "right_arc": []}
    errors_mm   = {"scale": [], "incisor_sum": [], "left_arc": [], "right_arc": []}

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

                scale_gt     = line_len(gt[0], gt[1])
                scale_err_px = abs(line_len(pred_pts[0], pred_pts[1]) - scale_gt)
                errors_px["scale"].append(scale_err_px)
                px_to_mm = 5.0 / scale_gt if scale_gt > 1e-6 else None
                if px_to_mm:
                    errors_mm["scale"].append(scale_err_px * px_to_mm)

                if valid_kps == 6:
                    left_err_px  = abs(line_len(pred_pts[2], pred_pts[3]) - line_len(gt[2], gt[3]))
                    right_err_px = abs(line_len(pred_pts[4], pred_pts[5]) - line_len(gt[4], gt[5]))
                    errors_px["left_arc"].append(left_err_px)
                    errors_px["right_arc"].append(right_err_px)
                    if px_to_mm:
                        errors_mm["left_arc"].append(left_err_px  * px_to_mm)
                        errors_mm["right_arc"].append(right_err_px * px_to_mm)
                else:
                    pred_inc   = sum(line_len(pred_pts[2*li], pred_pts[2*li+1]) for li in [1, 2, 3, 4])
                    gt_inc     = sum(line_len(gt[2*li], gt[2*li+1]) for li in [1, 2, 3, 4])
                    inc_err_px = abs(pred_inc - gt_inc)
                    errors_px["incisor_sum"].append(inc_err_px)
                    if px_to_mm:
                        errors_mm["incisor_sum"].append(inc_err_px * px_to_mm)
                    left_err_px  = abs(line_len(pred_pts[10], pred_pts[11]) - line_len(gt[10], gt[11]))
                    right_err_px = abs(line_len(pred_pts[12], pred_pts[13]) - line_len(gt[12], gt[13]))
                    errors_px["left_arc"].append(left_err_px)
                    errors_px["right_arc"].append(right_err_px)
                    if px_to_mm:
                        errors_mm["left_arc"].append(left_err_px  * px_to_mm)
                        errors_mm["right_arc"].append(right_err_px * px_to_mm)

                predictions[keys[i]] = {"predicted_keypoints": pred_pts}

    with pred_json_path.open("w") as f:
        json.dump(predictions, f, indent=2)

    means: Dict[str, float] = {}
    for k, v in errors_px.items():
        if v: means[f"{k}_px"] = float(np.mean(v))
    for k, v in errors_mm.items():
        if v: means[f"{k}_mm"] = float(np.mean(v))
    if all(x in means for x in ["incisor_sum_mm", "left_arc_mm", "right_arc_mm"]):
        means["sum_incisor_left_right_mm"] = means["incisor_sum_mm"] + means["left_arc_mm"] + means["right_arc_mm"]
    return means


# ---------------------------------------------------------------------------
# Reptile meta-update
# ---------------------------------------------------------------------------
def reptile_update(model: nn.Module, theta_0: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad and name in theta_0:
                param.data.copy_(theta_0[name] + EPSILON * (param.data - theta_0[name]))


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
    test_img_dir  = fold_root / "test" / "images"
    test_ann_file = fold_root / "test_annotations.json"
    for p in [train_img_dir, train_ann_file, test_img_dir, test_ann_file]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing: {p}")

    with train_ann_file.open() as f:
        train_data = json.load(f)
    with test_ann_file.open() as f:
        test_data  = json.load(f)

    arc_margin_px  = estimate_arc_margin_px(train_data)
    patient_groups = build_patient_groups(list(train_data.keys()))
    print(f"[{fold_name}] {len(train_data)} train images, {len(patient_groups)} patients, "
          f"ARC_MARGIN={arc_margin_px:.1f}px")

    train_keys = list(train_data.keys())
    test_keys  = list(test_data.keys())
    train_ds   = HeatmapDentalDataset(train_keys, train_data, train_img_dir)
    test_ds    = HeatmapDentalDataset(test_keys,  test_data,  test_img_dir)

    arch_labels = [infer_arch(train_data[k]) for k in train_keys]
    count_max   = max(1, sum(1 for a in arch_labels if a == "maxilla"))
    count_man   = max(1, sum(1 for a in arch_labels if a == "mandible"))
    weights     = [1.0 / (count_max if a == "maxilla" else count_man) for a in arch_labels]
    sampler     = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,     num_workers=4, pin_memory=True)

    fold_save       = save_root / fold_name
    fold_save.mkdir(parents=True, exist_ok=True)
    log_csv         = fold_save / "training_log.csv"
    best_model_path = fold_save / "heatmap_best.pth"
    pred_json_path  = fold_save / "test_predictions.json"

    with (fold_save / "run_config.json").open("w") as f:
        json.dump({
            "fold": fold_name, "backbone": "resnet50_reptile", "seed": seed,
            "warmup_epochs": WARMUP_EPOCHS, "meta_tasks_per_epoch": META_TASKS_PER_EPOCH,
            "meta_task_patients": META_TASK_PATIENTS, "k_inner": K_INNER,
            "lr_inner": LR_INNER, "epsilon": EPSILON,
            "train_patients": len(patient_groups), "train_samples": len(train_ds),
            "test_samples": len(test_ds),
        }, f, indent=2)

    with log_csv.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "phase", "sum3_mm"])

    model     = HeatmapNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(HEATMAP_SIZE, device=DEVICE),
        torch.arange(HEATMAP_SIZE, device=DEVICE),
        indexing="ij",
    )
    grid_x = grid_x.float()
    grid_y = grid_y.float()

    best_sum3_mm     = float("inf")
    patience_counter = 0
    best_metrics: Dict[str, float] = {}

    for epoch in range(EPOCHS):
        model.train()
        phase       = "warmup" if epoch < WARMUP_EPOCHS else "reptile"
        train_loss  = 0.0
        n_batches   = 0

        if phase == "warmup":
            # Standard heatmap-only training
            for imgs, heatmaps, gt_pts, _ in train_loader:
                imgs, heatmaps, gt_pts = imgs.to(DEVICE), heatmaps.to(DEVICE), gt_pts.to(DEVICE)
                optimizer.zero_grad()
                preds = model(imgs)
                loss  = compute_loss(preds, heatmaps, gt_pts, grid_x, grid_y, arc_margin_px, heatmap_only=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
                n_batches  += 1

        else:
            # Phase A: Reptile meta-update
            for _ in range(META_TASKS_PER_EPOCH):
                task_keys = sample_task_keys(patient_groups, META_TASK_PATIENTS)
                task_ds   = HeatmapDentalDataset(task_keys, train_data, train_img_dir)
                task_loader = DataLoader(task_ds, batch_size=min(BATCH_SIZE, len(task_ds)),
                                         shuffle=True, num_workers=2)

                # Save meta-params
                theta_0 = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

                # Inner loop with fresh SGD
                inner_opt = torch.optim.SGD(model.parameters(), lr=LR_INNER, momentum=0.9)
                task_iter = iter(task_loader)
                for _ in range(K_INNER):
                    try:
                        batch = next(task_iter)
                    except StopIteration:
                        task_iter = iter(task_loader)
                        batch = next(task_iter)
                    imgs, heatmaps, gt_pts, _ = batch
                    imgs, heatmaps, gt_pts = imgs.to(DEVICE), heatmaps.to(DEVICE), gt_pts.to(DEVICE)
                    inner_opt.zero_grad()
                    preds = model(imgs)
                    loss  = compute_loss(preds, heatmaps, gt_pts, grid_x, grid_y, arc_margin_px)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    inner_opt.step()
                    train_loss += loss.item()
                    n_batches  += 1

                # Reptile step: move meta-params toward adapted params
                reptile_update(model, theta_0)

            # Phase B: one standard pass to maintain convergence
            for imgs, heatmaps, gt_pts, _ in train_loader:
                imgs, heatmaps, gt_pts = imgs.to(DEVICE), heatmaps.to(DEVICE), gt_pts.to(DEVICE)
                optimizer.zero_grad()
                preds = model(imgs)
                loss  = compute_loss(preds, heatmaps, gt_pts, grid_x, grid_y, arc_margin_px)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
                n_batches  += 1

        train_loss /= max(1, n_batches)

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, heatmaps, _, _ in test_loader:
                imgs, heatmaps = imgs.to(DEVICE), heatmaps.to(DEVICE)
                preds      = model(imgs)
                diff       = (preds - heatmaps) ** 2
                valid_mask = (heatmaps.sum(dim=(2,3))>0).float().unsqueeze(2).unsqueeze(3).expand_as(diff)
                val_loss  += ((diff * valid_mask).sum() / (valid_mask.sum()+1e-6)).item()
        val_loss /= max(1, len(test_loader))
        scheduler.step(val_loss)

        # Evaluate every 10 epochs or at end of warmup
        sum3_mm = float("inf")
        if epoch % 10 == 0 or epoch == WARMUP_EPOCHS - 1:
            metrics  = evaluate(model, test_loader, pred_json_path)
            sum3_mm  = metrics.get("sum_incisor_left_right_mm", float("inf"))
            if sum3_mm < best_sum3_mm:
                best_sum3_mm = sum3_mm
                best_metrics = metrics.copy()
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
            print(f"[{fold_name}] Epoch {epoch:3d} [{phase}] | "
                  f"Train {train_loss:.6f} | Val {val_loss:.6f} | sum3={sum3_mm:.3f}mm")

        with log_csv.open("a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, phase,
                                    sum3_mm if sum3_mm < float("inf") else ""])

        if phase == "reptile" and patience_counter >= PATIENCE:
            print(f"[{fold_name}] Early stopping at epoch {epoch}")
            break

    best_metrics["best_sum3_mm"] = best_sum3_mm
    best_metrics["fold"]         = fold_name
    print(f"[{fold_name}] Metrics: {best_metrics}")
    return best_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--save-root",    type=Path, default=DEFAULT_SAVE_ROOT)
    p.add_argument("--k-folds",      type=int,  default=5)
    p.add_argument("--seed",         type=int,  default=42)
    p.add_argument("--run-name",     type=str,  default=None)
    p.add_argument("--epochs",       type=int,  default=EPOCHS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global EPOCHS
    EPOCHS = args.epochs

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
            raise FileNotFoundError(f"{fold_root} not found.")
        print(f"\n{'='*60}\nTraining {fold_name} — ResNet50 + Reptile\n{'='*60}")
        metrics = train_one_fold(fold_name, fold_root, run_root, seed=args.seed + i)
        summary_rows.append(metrics)

    fields = ["fold", "best_sum3_mm", "scale_px", "incisor_sum_px", "left_arc_px",
              "right_arc_px", "sum_incisor_left_right_px", "scale_mm", "incisor_sum_mm",
              "left_arc_mm", "right_arc_mm", "sum_incisor_left_right_mm"]
    with (run_root / "kfold_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: row.get(k, "") for k in fields})

    print(f"\n{'='*60}\nRESULTS — ResNet50 + Reptile Meta-Learning\n{'='*60}")
    for row in summary_rows:
        print(f"{row.get('fold')}: sum3={row.get('sum_incisor_left_right_mm','?'):.3f}mm | "
              f"incisor={row.get('incisor_sum_mm','?'):.3f} | "
              f"l_arc={row.get('left_arc_mm','?'):.3f} | "
              f"r_arc={row.get('right_arc_mm','?'):.3f}")
    print(f"\nOutput: {run_root}")


if __name__ == "__main__":
    main()
