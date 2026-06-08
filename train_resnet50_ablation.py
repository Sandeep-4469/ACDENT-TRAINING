#!/usr/bin/env python3
"""
Additive loss ablation study — ResNet50, fold_1 only.

Trains the same architecture 6 times on fold_1, each time activating one
more loss term. ALL loss components are ALWAYS COMPUTED and LOGGED every
epoch — even when not used for backprop — so you can see what each loss
was tracking throughout training.

Ablation sequence (additive):
  01_heatmap_only            → heatmap
  02_+coord                  → heatmap + coord
  03_+coord_+length          → heatmap + coord + length
  04_+coord_+length_+arcside → heatmap + coord + length + arc_side
  05_+coord_+length_+arcside_+arcalign
                             → heatmap + coord + length + arc_side + arc_align
  06_full                    → heatmap + coord + length + arc_side + arc_align + arc_contrast

Outputs:
  results/ablation/<run_name>/
    01_heatmap_only/
      training_log.csv          ← per-epoch: all 6 individual losses + val_loss + lr
      run_config.json
      heatmap_best.pth
      heatmap_last.pth
      test_predictions.json
    02_+coord/ ...
    ...
    ablation_training_log.csv   ← ALL configs combined, for visualisation
    ablation_summary.csv        ← one row per config, final test metrics
    ablation_report.txt         ← human-readable comparison table

Usage:
  python train_resnet50_ablation.py
  python train_resnet50_ablation.py --fold fold_1 --epochs 150
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
# Hyperparameters  (same as main training)
# ---------------------------------------------------------------------------
IMG_SIZE        = 512
HEATMAP_SIZE    = 128
NUM_KPS         = 14

BATCH_SIZE      = 8
EPOCHS          = 150        # shorter per config — 6 runs total
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DSAA_ROOT         = Path(__file__).resolve().parent
DEFAULT_DATASET   = DSAA_ROOT / "dataset"
DEFAULT_SAVE_ROOT = DSAA_ROOT / "results" / "ablation"


# ---------------------------------------------------------------------------
# Ablation configs  (additive — each row adds one loss over the previous)
# ---------------------------------------------------------------------------
ABLATION_CONFIGS: Dict[str, dict] = {
    "01_heatmap_only": {
        "description":    "Baseline — heatmap loss only",
        "use_coord":      False,
        "use_length":     False,
        "use_arc_side":   False,
        "use_arc_align":  False,
        "use_arc_contrast": False,
    },
    "02_+coord": {
        "description":    "+ coordinate regression loss",
        "use_coord":      True,
        "use_length":     False,
        "use_arc_side":   False,
        "use_arc_align":  False,
        "use_arc_contrast": False,
    },
    "03_+coord_+length": {
        "description":    "+ line length loss",
        "use_coord":      True,
        "use_length":     True,
        "use_arc_side":   False,
        "use_arc_align":  False,
        "use_arc_contrast": False,
    },
    "04_+coord_+length_+arcside": {
        "description":    "+ arc side-order loss",
        "use_coord":      True,
        "use_length":     True,
        "use_arc_side":   True,
        "use_arc_align":  False,
        "use_arc_contrast": False,
    },
    "05_+coord_+length_+arcside_+arcalign": {
        "description":    "+ arc centre-alignment loss",
        "use_coord":      True,
        "use_length":     True,
        "use_arc_side":   True,
        "use_arc_align":  True,
        "use_arc_contrast": False,
    },
    "06_full": {
        "description":    "+ arc contrast loss  (full combination)",
        "use_coord":      True,
        "use_length":     True,
        "use_arc_side":   True,
        "use_arc_align":  True,
        "use_arc_contrast": True,
    },
}

# column order for training log CSV
LOG_FIELDS = [
    "config", "epoch",
    "train_total",
    "train_heatmap",    # always logged (always in loss)
    "train_coord",      # logged even when inactive
    "train_length",
    "train_arc_side",
    "train_arc_align",
    "train_arc_contrast",
    "val_loss",
    "learning_rate",
    "active_losses",    # which terms were added to backprop
]


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
# Model — ResNet50
# ---------------------------------------------------------------------------
class HeatmapNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(2048, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256,  256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256,  256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.final_layer = nn.Conv2d(256, NUM_KPS, 1)

    def forward(self, x):
        return self.final_layer(self.deconv(self.backbone(x)))


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def decode_heatmap(hm: torch.Tensor) -> List[List[float]]:
    hm_np = hm.cpu().numpy()
    return [[float(x), float(y)]
            for y, x in (np.unravel_index(np.argmax(hm_np[i]), hm_np[i].shape)
                         for i in range(NUM_KPS))]


def softargmax_coords(
    logits: torch.Tensor, grid_x: torch.Tensor, grid_y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs   = torch.softmax(
        logits.view(logits.size(0), logits.size(1), -1) * SOFTARGMAX_TEMP, dim=2
    ).view_as(logits)
    coord_x = (probs * grid_x).sum(dim=(2, 3)) * IMG_SIZE / HEATMAP_SIZE
    coord_y = (probs * grid_y).sum(dim=(2, 3)) * IMG_SIZE / HEATMAP_SIZE
    return coord_x, coord_y


def line_len(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p1, np.float32) - np.array(p2, np.float32)))


def find_train_assets(fold_root: Path) -> Tuple[Path, Path]:
    aug_img = fold_root / "train_augmented" / "images"
    aug_ann = fold_root / "train_annotations_augmented.json"
    if aug_img.exists() and aug_ann.exists():
        return aug_img, aug_ann
    return fold_root / "train" / "images", fold_root / "train_annotations.json"


def infer_arch(entry: dict) -> str:
    arch = str(entry.get("arch", "")).strip().lower()
    if arch in {"maxilla", "mandible"}:
        return arch
    return "maxilla" if len(entry.get("lines", [])) <= 3 else "mandible"


def estimate_arc_margin_px(train_data: Dict[str, dict]) -> float:
    gaps_max, gaps_man = [], []
    for item in train_data.values():
        lines = item.get("lines", [])
        if len(lines) == 3:
            a, b = lines[1], lines[2]; bucket = gaps_max
        elif len(lines) == 7:
            a, b = lines[5], lines[6]; bucket = gaps_man
        else:
            continue
        bucket.append(abs(0.5*(a[0][0]+a[1][0]) - 0.5*(b[0][0]+b[1][0])))
    pools = []
    if gaps_max: pools.append(float(np.percentile(np.array(gaps_max, np.float32), 10)))
    if gaps_man: pools.append(float(np.percentile(np.array(gaps_man, np.float32), 10)))
    return float(np.clip(0.12 * min(pools), 6.0, 14.0)) if pools else ARC_MARGIN_PX_DEFAULT


# ---------------------------------------------------------------------------
# Loss computation — always compute all terms, return dict
# ---------------------------------------------------------------------------
def compute_all_losses(
    preds:        torch.Tensor,
    heatmaps:     torch.Tensor,
    gt_pts:       torch.Tensor,
    grid_x:       torch.Tensor,
    grid_y:       torch.Tensor,
    arc_margin_px: float,
) -> Dict[str, torch.Tensor]:
    """
    Compute every loss component independently.
    Returns a dict so training loop can selectively sum and always log all values.
    """
    # 1. Heatmap MSE
    diff       = (preds - heatmaps) ** 2
    valid_mask = (heatmaps.sum(dim=(2, 3)) > 0).float().unsqueeze(2).unsqueeze(3).expand_as(diff)
    heatmap_loss = (diff * valid_mask).sum() / (valid_mask.sum() + 1e-6)

    coord_x, coord_y = softargmax_coords(preds, grid_x, grid_y)

    # 2. Coordinate regression
    pred_xy    = torch.stack([coord_x, coord_y], dim=2)
    valid_kp   = (gt_pts[:, :, 0] >= 0)
    coord_res  = F.smooth_l1_loss(pred_xy, gt_pts, reduction="none").sum(dim=2) / IMG_SIZE
    coord_loss = (coord_res * valid_kp.float()).sum() / (valid_kp.sum() + 1e-6)

    # 3. Line length
    length_loss = torch.tensor(0.0, device=DEVICE)
    line_count  = torch.tensor(0.0, device=DEVICE)
    for j in range(0, NUM_KPS, 2):
        vl = gt_pts[:, j, 0] >= 0
        if not vl.any():
            continue
        pl = torch.sqrt((coord_x[:,j]-coord_x[:,j+1])**2 + (coord_y[:,j]-coord_y[:,j+1])**2 + 1e-6)
        gl = torch.sqrt((gt_pts[:,j,0]-gt_pts[:,j+1,0])**2 + (gt_pts[:,j,1]-gt_pts[:,j+1,1])**2 + 1e-6)
        length_loss = length_loss + ((((pl - gl) / IMG_SIZE)**2) * vl.float()).sum()
        line_count  = line_count  + vl.float().sum()
    length_loss = length_loss / (line_count + 1e-6)

    # 4-6. Arc losses — compute each separately for logging
    arc_side_loss  = torch.tensor(0.0, device=DEVICE)
    arc_align_loss = torch.tensor(0.0, device=DEVICE)
    arc_ctrst_loss = torch.tensor(0.0, device=DEVICE)

    def _arc(mask, ci, cj, gi, gj):
        nonlocal arc_side_loss, arc_align_loss, arc_ctrst_loss
        if not mask.any():
            return
        pred_arcs = torch.stack([0.5*(coord_x[:,ci]+coord_x[:,cj]),
                                  0.5*(coord_x[:,gi]+coord_x[:,gj])], dim=1)
        gt_arcs   = torch.stack([0.5*(gt_pts[:,ci,0]+gt_pts[:,cj,0]),
                                  0.5*(gt_pts[:,gi,0]+gt_pts[:,gj,0])], dim=1)
        ps, _ = torch.sort(pred_arcs, dim=1)
        gs, _ = torch.sort(gt_arcs,   dim=1)
        lpx, rpx = ps[:,0], ps[:,1]
        lgx, rgx = gs[:,0], gs[:,1]

        side    = F.relu(lpx - rpx + arc_margin_px)
        align   = (F.smooth_l1_loss(lpx,lgx,reduction="none") +
                   F.smooth_l1_loss(rpx,rgx,reduction="none"))
        d_lo, d_la = torch.abs(lpx-lgx), torch.abs(lpx-rgx)
        d_ro, d_ra = torch.abs(rpx-rgx), torch.abs(rpx-lgx)
        contrast = (F.relu(d_lo+arc_margin_px-d_la) +
                    F.relu(d_ro+arc_margin_px-d_ra))

        n = mask.sum() + 1e-6
        arc_side_loss  = arc_side_loss  + (side     * mask).sum() / n
        arc_align_loss = arc_align_loss + (align    * mask).sum() / n
        arc_ctrst_loss = arc_ctrst_loss + (contrast * mask).sum() / n

    max_mask = (gt_pts[:,:,0] >= 0).sum(dim=1) == 6
    _arc(max_mask, 2, 3, 4, 5)
    man_mask = (gt_pts[:,:,0] >= 0).sum(dim=1) == 14
    _arc(man_mask, 10, 11, 12, 13)

    return {
        "heatmap":     heatmap_loss,
        "coord":       coord_loss,
        "length":      length_loss,
        "arc_side":    arc_side_loss  / IMG_SIZE,
        "arc_align":   arc_align_loss / IMG_SIZE,
        "arc_contrast": arc_ctrst_loss / IMG_SIZE,
    }


def build_total_loss(losses: Dict[str, torch.Tensor], cfg: dict) -> torch.Tensor:
    """Combine only the active loss terms for backprop."""
    total = losses["heatmap"]                                                   # always active
    if cfg["use_coord"]:       total = total + COORD_LOSS_W       * losses["coord"]
    if cfg["use_length"]:      total = total + LENGTH_LOSS_W      * losses["length"]
    if cfg["use_arc_side"]:    total = total + ARC_SIDE_LOSS_W    * losses["arc_side"]
    if cfg["use_arc_align"]:   total = total + ARC_CENTER_ALIGN_W * losses["arc_align"]
    if cfg["use_arc_contrast"]: total = total + ARC_CONTRAST_W    * losses["arc_contrast"]
    return total


def active_loss_label(cfg: dict) -> str:
    parts = ["heatmap"]
    if cfg["use_coord"]:       parts.append("coord")
    if cfg["use_length"]:      parts.append("length")
    if cfg["use_arc_side"]:    parts.append("arc_side")
    if cfg["use_arc_align"]:   parts.append("arc_align")
    if cfg["use_arc_contrast"]: parts.append("arc_contrast")
    return "+".join(parts)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model: nn.Module, loader: DataLoader, pred_json_path: Path) -> Dict[str, float]:
    model.eval()
    predictions = {}
    errors_px   = {"scale": [], "incisor_sum": [], "left_arc": [], "right_arc": []}
    errors_mm   = {"scale": [], "incisor_sum": [], "left_arc": [], "right_arc": []}

    with torch.no_grad():
        for imgs, _, gt_pts, keys in tqdm(loader, leave=False, desc="  eval"):
            imgs  = imgs.to(DEVICE)
            preds = model(imgs)
            for i in range(len(imgs)):
                pred_coords = decode_heatmap(preds[i])
                pred_pts    = [[x*IMG_SIZE/HEATMAP_SIZE, y*IMG_SIZE/HEATMAP_SIZE]
                               for x, y in pred_coords]
                gt          = gt_pts[i].numpy()
                valid_kps   = int(np.sum(gt[:,0] >= 0))

                sc_pred = line_len(pred_pts[0], pred_pts[1])
                sc_gt   = line_len(gt[0], gt[1])
                sc_err  = abs(sc_pred - sc_gt)
                errors_px["scale"].append(sc_err)
                px2mm = 5.0 / sc_gt if sc_gt > 1e-6 else None
                if px2mm: errors_mm["scale"].append(sc_err * px2mm)

                if valid_kps == 6:
                    pa, ga = [], []
                    for li in [1,2]:
                        p1,p2=pred_pts[2*li],pred_pts[2*li+1]; g1,g2=gt[2*li],gt[2*li+1]
                        pa.append(((p1[0]+p2[0])/2, line_len(p1,p2)))
                        ga.append(((g1[0]+g2[0])/2, line_len(g1,g2)))
                    pa.sort(key=lambda a:a[0]); ga.sort(key=lambda a:a[0])
                    le=abs(pa[0][1]-ga[0][1]); re=abs(pa[1][1]-ga[1][1])
                    errors_px["left_arc"].append(le); errors_px["right_arc"].append(re)
                    if px2mm:
                        errors_mm["left_arc"].append(le*px2mm); errors_mm["right_arc"].append(re*px2mm)
                else:
                    ie = abs(sum(line_len(pred_pts[2*li],pred_pts[2*li+1]) for li in [1,2,3,4]) -
                             sum(line_len(gt[2*li],gt[2*li+1]) for li in [1,2,3,4]))
                    errors_px["incisor_sum"].append(ie)
                    if px2mm: errors_mm["incisor_sum"].append(ie*px2mm)
                    pa, ga = [], []
                    for li in [5,6]:
                        p1,p2=pred_pts[2*li],pred_pts[2*li+1]; g1,g2=gt[2*li],gt[2*li+1]
                        pa.append(((p1[0]+p2[0])/2, line_len(p1,p2)))
                        ga.append(((g1[0]+g2[0])/2, line_len(g1,g2)))
                    pa.sort(key=lambda a:a[0]); ga.sort(key=lambda a:a[0])
                    le=abs(pa[0][1]-ga[0][1]); re=abs(pa[1][1]-ga[1][1])
                    errors_px["left_arc"].append(le); errors_px["right_arc"].append(re)
                    if px2mm:
                        errors_mm["left_arc"].append(le*px2mm); errors_mm["right_arc"].append(re*px2mm)

                predictions[keys[i]] = {"predicted_keypoints": pred_pts}

    with pred_json_path.open("w") as f:
        json.dump(predictions, f, indent=2)

    means: Dict[str, float] = {}
    for k, v in errors_px.items():
        if v: means[f"{k}_px"] = float(np.mean(v))
    for k, v in errors_mm.items():
        if v: means[f"{k}_mm"] = float(np.mean(v))
    if all(x in means for x in ["incisor_sum_px","left_arc_px","right_arc_px"]):
        means["sum3_px"] = means["incisor_sum_px"]+means["left_arc_px"]+means["right_arc_px"]
    if all(x in means for x in ["incisor_sum_mm","left_arc_mm","right_arc_mm"]):
        means["sum3_mm"] = means["incisor_sum_mm"]+means["left_arc_mm"]+means["right_arc_mm"]
    return means


# ---------------------------------------------------------------------------
# Train one config
# ---------------------------------------------------------------------------
def train_one_config(
    config_name: str,
    cfg:         dict,
    fold_root:   Path,
    save_dir:    Path,
    seed:        int,
    arc_margin_px: float,
    train_loader:  DataLoader,
    test_loader:   DataLoader,
    train_data:    Dict[str, dict],
    test_data:     Dict[str, dict],
) -> Dict[str, float]:

    save_dir.mkdir(parents=True, exist_ok=True)
    active_label = active_loss_label(cfg)

    print(f"\n{'='*65}")
    print(f"  Config : {config_name}")
    print(f"  Desc   : {cfg['description']}")
    print(f"  Active : {active_label}")
    print(f"{'='*65}")

    log_csv         = save_dir / "training_log.csv"
    best_model_path = save_dir / "heatmap_best.pth"
    last_model_path = save_dir / "heatmap_last.pth"
    pred_json_path  = save_dir / "test_predictions.json"

    with (save_dir / "run_config.json").open("w") as f:
        json.dump({
            "config_name":   config_name,
            "description":   cfg["description"],
            "active_losses": active_label,
            "use_coord":     cfg["use_coord"],
            "use_length":    cfg["use_length"],
            "use_arc_side":  cfg["use_arc_side"],
            "use_arc_align": cfg["use_arc_align"],
            "use_arc_contrast": cfg["use_arc_contrast"],
            "weights": {
                "coord":       COORD_LOSS_W       if cfg["use_coord"]       else 0,
                "length":      LENGTH_LOSS_W      if cfg["use_length"]      else 0,
                "arc_side":    ARC_SIDE_LOSS_W    if cfg["use_arc_side"]    else 0,
                "arc_align":   ARC_CENTER_ALIGN_W if cfg["use_arc_align"]   else 0,
                "arc_contrast": ARC_CONTRAST_W    if cfg["use_arc_contrast"] else 0,
            },
            "arc_margin_px": arc_margin_px,
            "epochs":       EPOCHS,
            "batch_size":   BATCH_SIZE,
            "lr":           LR,
            "weight_decay": WEIGHT_DECAY,
            "patience":     PATIENCE,
            "seed":         seed,
            "train_samples": len(train_data),
            "test_samples":  len(test_data),
            "device":        str(DEVICE),
        }, f, indent=2)

    # Write CSV header
    with log_csv.open("w", newline="") as f:
        csv.writer(f).writerow(LOG_FIELDS)

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
    grid_x = grid_x.float(); grid_y = grid_y.float()

    for epoch in range(EPOCHS):
        # ---- Train ----
        model.train()
        # Accumulators for per-epoch averages of each individual loss
        acc = {k: 0.0 for k in ["total","heatmap","coord","length","arc_side","arc_align","arc_contrast"]}

        for imgs, heatmaps, gt_pts, _ in train_loader:
            imgs, heatmaps, gt_pts = imgs.to(DEVICE), heatmaps.to(DEVICE), gt_pts.to(DEVICE)
            optimizer.zero_grad()

            preds = model(imgs)

            # Compute ALL individual losses
            losses = compute_all_losses(preds, heatmaps, gt_pts, grid_x, grid_y, arc_margin_px)

            # Build total using only active terms
            total = build_total_loss(losses, cfg)

            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate all values for logging (detach to avoid holding graph)
            acc["total"]        += total.item()
            acc["heatmap"]      += losses["heatmap"].item()
            acc["coord"]        += losses["coord"].item()
            acc["length"]       += losses["length"].item()
            acc["arc_side"]     += losses["arc_side"].item()
            acc["arc_align"]    += losses["arc_align"].item()
            acc["arc_contrast"] += losses["arc_contrast"].item()

        n_batches = max(1, len(train_loader))
        for k in acc: acc[k] /= n_batches

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, heatmaps, _, _ in test_loader:
                imgs, heatmaps = imgs.to(DEVICE), heatmaps.to(DEVICE)
                preds      = model(imgs)
                diff       = (preds - heatmaps) ** 2
                vm         = (heatmaps.sum(dim=(2,3))>0).float().unsqueeze(2).unsqueeze(3).expand_as(diff)
                val_loss  += ((diff * vm).sum() / (vm.sum() + 1e-6)).item()
        val_loss /= max(1, len(test_loader))

        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        # ---- Log ----
        with log_csv.open("a", newline="") as f:
            csv.writer(f).writerow([
                config_name,
                epoch,
                f"{acc['total']:.8f}",
                f"{acc['heatmap']:.8f}",
                f"{acc['coord']:.8f}",
                f"{acc['length']:.8f}",
                f"{acc['arc_side']:.8f}",
                f"{acc['arc_align']:.8f}",
                f"{acc['arc_contrast']:.8f}",
                f"{val_loss:.8f}",
                f"{lr_now:.2e}",
                active_label,
            ])

        print(
            f"  [{config_name}] Ep {epoch:03d} | "
            f"total={acc['total']:.5f}  heatmap={acc['heatmap']:.5f}  "
            f"coord={acc['coord']:.5f}  len={acc['length']:.5f}  "
            f"arc_side={acc['arc_side']:.5f}  arc_align={acc['arc_align']:.5f}  "
            f"arc_ctrst={acc['arc_contrast']:.5f} | "
            f"val={val_loss:.5f}"
        )

        if val_loss < best_val:
            best_val = val_loss; patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  [{config_name}] Early stopping at epoch {epoch}")
            break

    torch.save(model.state_dict(), last_model_path)

    # Evaluate best checkpoint on test set
    best_model = HeatmapNet().to(DEVICE)
    best_model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    metrics = evaluate(best_model, test_loader, pred_json_path)
    metrics["best_val_loss"] = float(best_val)
    metrics["final_epoch"]   = epoch
    metrics["config"]        = config_name
    metrics["description"]   = cfg["description"]
    metrics["active_losses"] = active_label
    return metrics


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Additive loss ablation study — ResNet50, single fold.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--save-root",    type=Path, default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--fold",         type=str,  default="fold_1")
    parser.add_argument("--seed",         type=int,  default=42)
    parser.add_argument("--epochs",       type=int,  default=EPOCHS)
    parser.add_argument("--batch-size",   type=int,  default=BATCH_SIZE)
    parser.add_argument("--run-name",     type=str,  default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    global EPOCHS, BATCH_SIZE
    EPOCHS = args.epochs; BATCH_SIZE = args.batch_size

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    fold_root = args.dataset_root / args.fold
    if not fold_root.exists():
        raise FileNotFoundError(f"{fold_root} not found. Run prepare_folds.py first.")

    run_name = args.run_name or f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = args.save_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    # Load fold data once — shared across all configs
    train_img_dir, train_ann_file = find_train_assets(fold_root)
    test_img_dir  = fold_root / "test"  / "images"
    test_ann_file = fold_root / "test_annotations.json"

    with train_ann_file.open() as f: train_data = json.load(f)
    with test_ann_file.open()  as f: test_data  = json.load(f)

    arc_margin_px = estimate_arc_margin_px(train_data)
    print(f"Fold          : {args.fold}")
    print(f"Arc margin px : {arc_margin_px:.2f}")
    print(f"Train samples : {len(train_data)}")
    print(f"Test samples  : {len(test_data)}")
    print(f"Output        : {run_root}")
    print(f"Configs to run: {len(ABLATION_CONFIGS)}")

    # Build dataloaders
    train_keys  = list(train_data.keys())
    test_keys   = list(test_data.keys())
    arch_labels = [infer_arch(train_data[k]) for k in train_keys]
    count_max   = max(1, sum(1 for a in arch_labels if a == "maxilla"))
    count_man   = max(1, sum(1 for a in arch_labels if a == "mandible"))
    weights     = [1.0/(count_max if a=="maxilla" else count_man) for a in arch_labels]
    sampler     = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True)

    train_ds = HeatmapDentalDataset(train_keys, train_data, train_img_dir)
    test_ds  = HeatmapDentalDataset(test_keys,  test_data,  test_img_dir)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Save ablation plan for reference
    with (run_root / "ablation_plan.json").open("w") as f:
        json.dump({
            "fold":       args.fold,
            "seed":       args.seed,
            "epochs":     EPOCHS,
            "batch_size": BATCH_SIZE,
            "configs":    {k: v for k, v in ABLATION_CONFIGS.items()},
        }, f, indent=2)

    # ---- Run all configs ----
    all_metrics: List[Dict] = []
    for config_name, cfg in ABLATION_CONFIGS.items():
        metrics = train_one_config(
            config_name=config_name,
            cfg=cfg,
            fold_root=fold_root,
            save_dir=run_root / config_name,
            seed=args.seed,
            arc_margin_px=arc_margin_px,
            train_loader=train_loader,
            test_loader=test_loader,
            train_data=train_data,
            test_data=test_data,
        )
        all_metrics.append(metrics)

    # ---- Consolidate training logs (all configs into one CSV) ----
    combined_log = run_root / "ablation_training_log.csv"
    with combined_log.open("w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(LOG_FIELDS)
        for config_name in ABLATION_CONFIGS:
            fold_log = run_root / config_name / "training_log.csv"
            if not fold_log.exists():
                continue
            with fold_log.open("r") as fin:
                reader = csv.DictReader(fin)
                for row in reader:
                    writer.writerow([row.get(f, "") for f in LOG_FIELDS])

    # ---- Summary CSV ----
    summary_fields = [
        "config", "description", "active_losses",
        "final_epoch", "best_val_loss",
        "scale_px",        "scale_mm",
        "incisor_sum_px",  "incisor_sum_mm",
        "left_arc_px",     "left_arc_mm",
        "right_arc_px",    "right_arc_mm",
        "sum3_px",         "sum3_mm",
    ]
    summary_csv = run_root / "ablation_summary.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
        w.writeheader()
        for row in all_metrics:
            w.writerow({k: row.get(k, "") for k in summary_fields})

    # ---- Human-readable report ----
    sep   = "=" * 90
    hsep  = "-" * 90
    lines = [
        sep,
        "ADDITIVE LOSS ABLATION REPORT — ResNet50",
        f"Fold: {args.fold}   |   Epochs per config: {EPOCHS}   |   Run: {run_root.name}",
        sep,
        "",
        f"{'Config':<42} {'BestVal':>8} {'Ep':>4} {'scale_mm':>9} {'inc_mm':>8} {'l_arc_mm':>9} {'r_arc_mm':>9} {'sum3_mm':>8}",
        hsep,
    ]
    for row in all_metrics:
        lines.append(
            f"{row.get('config',''):<42} "
            f"{row.get('best_val_loss', float('nan')):>8.5f} "
            f"{row.get('final_epoch', 0):>4d} "
            f"{row.get('scale_mm',        float('nan')):>9.3f} "
            f"{row.get('incisor_sum_mm',  float('nan')):>8.3f} "
            f"{row.get('left_arc_mm',     float('nan')):>9.3f} "
            f"{row.get('right_arc_mm',    float('nan')):>9.3f} "
            f"{row.get('sum3_mm',         float('nan')):>8.3f}"
        )
    lines += [
        hsep,
        "",
        "Individual loss values logged every epoch (always computed, even when inactive):",
        "  Columns: config, epoch, train_total, train_heatmap, train_coord, train_length,",
        "           train_arc_side, train_arc_align, train_arc_contrast, val_loss, lr, active_losses",
        f"  File: {combined_log}",
        "",
        "Per-config model weights:",
    ]
    for config_name in ABLATION_CONFIGS:
        pth = run_root / config_name / "heatmap_best.pth"
        lines.append(f"  {config_name}: {pth}")
    lines += ["", sep]

    report_text = "\n".join(lines)
    print("\n\n" + report_text)
    with (run_root / "ablation_report.txt").open("w") as f:
        f.write(report_text + "\n")

    print(f"\nAll outputs at: {run_root}")
    print(f"Combined log  : {combined_log}")
    print(f"Summary CSV   : {summary_csv}")


if __name__ == "__main__":
    main()
