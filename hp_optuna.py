#!/usr/bin/env python3
"""
Hyperparameter tuning with Optuna (TPE + MedianPruner).

Each trial trains fold_1. Every 10 epochs the current sum3_mm is reported
to Optuna — unpromising trials are pruned early, so bad configs die fast
and compute concentrates on good regions of the search space.

Usage:
  python hp_optuna.py                     # 20 trials, 150 epochs max
  python hp_optuna.py --n-trials 30 --max-epochs 200
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
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Fixed constants
# ---------------------------------------------------------------------------
IMG_SIZE     = 512
HEATMAP_SIZE = 128
NUM_KPS      = 14
BATCH_SIZE   = 8
WEIGHT_DECAY = 1e-4
PATIENCE     = 8          # eval windows (each = 10 epochs) without improvement
EVAL_EVERY   = 10

DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DSAA_ROOT        = Path(__file__).resolve().parent
FOLD_ROOT        = DSAA_ROOT / "dataset_v2" / "fold_1"
SAVE_ROOT        = DSAA_ROOT / "results" / "hp_optuna"
_PRETRAINED_PATH = ""   # set via --pretrained arg in main()


# ---------------------------------------------------------------------------
# Arc sort (same as train_resnet50.py)
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
# Dataset
# ---------------------------------------------------------------------------
class HeatmapDentalDataset(Dataset):
    def __init__(self, keys, data_dict, img_dir, sigma: int = 3):
        self.keys      = keys
        self.data      = data_dict
        self.img_dir   = img_dir
        self.sigma     = sigma
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    def __len__(self): return len(self.keys)

    def _heatmap(self, pts):
        hm = np.zeros((NUM_KPS, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
        xx, yy = np.meshgrid(np.arange(HEATMAP_SIZE), np.arange(HEATMAP_SIZE))
        for i, (x, y) in enumerate(pts):
            if x < 0: continue
            hx = int(x * HEATMAP_SIZE / IMG_SIZE)
            hy = int(y * HEATMAP_SIZE / IMG_SIZE)
            hm[i] = np.exp(-((xx-hx)**2+(yy-hy)**2)/(2*self.sigma**2))
        return hm

    def __getitem__(self, idx):
        key  = self.keys[idx]
        item = self.data[key]
        img  = cv2.imread(str(self.img_dir / f"{key}.jpg"))
        h, w = img.shape[:2]
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lines = _sort_arc_lines(item["lines"])
        pts   = [p for line in lines for p in line]
        while len(pts) < NUM_KPS: pts.append([-1,-1])
        pts = pts[:NUM_KPS]
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        sx, sy = IMG_SIZE/w, IMG_SIZE/h
        scaled = [[-1.,-1.] if x<0 else [x*sx, y*sy] for x,y in pts]
        return self.normalize(img), torch.tensor(self._heatmap(scaled)), torch.tensor(scaled), key


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class HeatmapNet(nn.Module):
    def __init__(self):
        super().__init__()
        bb = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(bb.children())[:-2])
        self.deconv   = nn.Sequential(
            nn.ConvTranspose2d(2048,256,4,2,1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d( 256,256,4,2,1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d( 256,256,4,2,1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.final_layer = nn.Conv2d(256, NUM_KPS, 1)

    def forward(self, x):
        return self.final_layer(self.deconv(self.backbone(x)))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def line_len(p1, p2):
    return float(np.linalg.norm(np.array(p1,dtype=np.float32)-np.array(p2,dtype=np.float32)))

def infer_arch(entry):
    return "maxilla" if len(entry.get("lines",[])) <= 3 else "mandible"

def find_train_assets(fold_root):
    ai = fold_root/"train_augmented"/"images"
    aa = fold_root/"train_annotations_augmented.json"
    if ai.exists() and aa.exists(): return ai, aa
    return fold_root/"train"/"images", fold_root/"train_annotations.json"

def is_vlm(key): return bool(__import__("re").search(r'__.+__', key))

def estimate_arc_margin(train_data):
    gaps = []
    for item in train_data.values():
        lines = item.get("lines",[])
        if len(lines)==3:   a,b = lines[1],lines[2]
        elif len(lines)==7: a,b = lines[5],lines[6]
        else: continue
        gaps.append(abs(0.5*(a[0][0]+a[1][0])-0.5*(b[0][0]+b[1][0])))
    return float(np.clip(0.12*np.percentile(gaps,10),6.,14.)) if gaps else 12.

def softargmax(logits, gx, gy, temp):
    p = torch.softmax(logits.view(*logits.shape[:2],-1)*temp, 2).view_as(logits)
    return (p*gx).sum((2,3))*IMG_SIZE/HEATMAP_SIZE, (p*gy).sum((2,3))*IMG_SIZE/HEATMAP_SIZE


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def total_loss(preds, heatmaps, gt_pts, gx, gy, cfg):
    diff  = (preds-heatmaps)**2
    vmask = (heatmaps.sum((2,3))>0).float().unsqueeze(2).unsqueeze(3).expand_as(diff)
    loss  = (diff*vmask).sum()/(vmask.sum()+1e-6)

    cx, cy = softargmax(preds, gx, gy, cfg["temp"])
    vkp    = (gt_pts[:,:,0]>=0)
    pxy    = torch.stack([cx,cy],2)
    closs  = (F.smooth_l1_loss(pxy,gt_pts,reduction="none").sum(2)/IMG_SIZE * vkp.float()).sum()/(vkp.sum()+1e-6)

    lloss = lc = torch.tensor(0.,device=preds.device)
    for j in range(0,NUM_KPS,2):
        vl = gt_pts[:,j,0]>=0
        if not vl.any(): continue
        pl = torch.sqrt((cx[:,j]-cx[:,j+1])**2+(cy[:,j]-cy[:,j+1])**2+1e-6)
        gl = torch.sqrt((gt_pts[:,j,0]-gt_pts[:,j+1,0])**2+(gt_pts[:,j,1]-gt_pts[:,j+1,1])**2+1e-6)
        lloss = lloss+(((pl-gl)/IMG_SIZE)**2*vl.float()).sum()
        lc    = lc+vl.float().sum()
    lloss = lloss/(lc+1e-6)

    margin = cfg["arc_margin"]
    sloss = cen = con = torch.tensor(0.,device=preds.device)
    for mask,(li,ri) in [((gt_pts[:,:,0]>=0).sum(1)==6,(2,4)),
                          ((gt_pts[:,:,0]>=0).sum(1)==14,(10,12))]:
        if not mask.any(): continue
        pa = torch.stack([0.5*(cx[:,li]+cx[:,li+1]), 0.5*(cx[:,ri]+cx[:,ri+1])],1)
        ga = torch.stack([0.5*(gt_pts[:,li,0]+gt_pts[:,li+1,0]), 0.5*(gt_pts[:,ri,0]+gt_pts[:,ri+1,0])],1)
        ps,_ = torch.sort(pa,1); gs,_ = torch.sort(ga,1)
        lp,rp = ps[:,0],ps[:,1]; lg,rg = gs[:,0],gs[:,1]
        n = mask.sum()+1e-6
        sloss = sloss+(F.relu(lp-rp+margin)*mask).sum()/n/IMG_SIZE
        cen   = cen+((F.smooth_l1_loss(lp,lg,"none")+F.smooth_l1_loss(rp,rg,"none"))*mask).sum()/n/IMG_SIZE
        dlo,dlx = torch.abs(lp-lg),torch.abs(lp-rg)
        dro,drx = torch.abs(rp-rg),torch.abs(rp-lg)
        con = con+((F.relu(dlo+margin-dlx)+F.relu(dro+margin-drx))*mask).sum()/n/IMG_SIZE

    return (loss
            + cfg["coord_w"]   * closs
            + cfg["len_w"]     * lloss
            + cfg["side_w"]    * sloss
            + cfg["center_w"]  * cen
            + cfg["contrast_w"]* con)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, loader) -> float:
    model.eval()
    errs = {"inc":[], "larc":[], "rarc":[]}
    with torch.no_grad():
        for imgs, _, gt_pts, _ in loader:
            imgs = imgs.to(DEVICE)
            preds = model(imgs)
            for i in range(len(imgs)):
                gt   = gt_pts[i].numpy()
                vkps = int(np.sum(gt[:,0]>=0))
                pc   = [[x*IMG_SIZE/HEATMAP_SIZE, y*IMG_SIZE/HEATMAP_SIZE]
                        for x,y in [np.unravel_index(np.argmax(preds[i,k].cpu().numpy()),
                                    (HEATMAP_SIZE,HEATMAP_SIZE))[::-1] for k in range(NUM_KPS)]]
                sc = line_len(gt[0],gt[1]); px2mm = 5./sc if sc>1e-6 else None
                if not px2mm: continue
                if vkps==6:
                    errs["larc"].append(abs(line_len(pc[2],pc[3])-line_len(gt[2],gt[3]))*px2mm)
                    errs["rarc"].append(abs(line_len(pc[4],pc[5])-line_len(gt[4],gt[5]))*px2mm)
                else:
                    errs["inc"].append(abs(sum(line_len(pc[2*l],pc[2*l+1]) for l in [1,2,3,4])
                                         -sum(line_len(gt[2*l],gt[2*l+1]) for l in [1,2,3,4]))*px2mm)
                    errs["larc"].append(abs(line_len(pc[10],pc[11])-line_len(gt[10],gt[11]))*px2mm)
                    errs["rarc"].append(abs(line_len(pc[12],pc[13])-line_len(gt[12],gt[13]))*px2mm)
    parts = [np.mean(v) for v in errs.values() if v]
    return float(sum(parts)) if parts else 999.


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
def objective(trial: optuna.Trial, max_epochs: int, fold_root: Path) -> float:
    # Search centered around current defaults with ±5× range
    cfg = {
        "lr":         trial.suggest_float("lr",         5e-5,  5e-4,  log=True),
        "coord_w":    trial.suggest_float("coord_w",    0.005, 0.1,   log=True),
        "len_w":      trial.suggest_float("len_w",      0.002, 0.05,  log=True),
        "side_w":     trial.suggest_float("side_w",     0.002, 0.04,  log=True),
        "center_w":   trial.suggest_float("center_w",   0.001, 0.02,  log=True),
        "contrast_w": trial.suggest_float("contrast_w", 0.001, 0.02,  log=True),
        "sigma":      trial.suggest_int  ("sigma",       2,    4),
        "temp":       trial.suggest_float("temp",        8.,   20.),
        "arc_margin": 12.0,
    }

    # Load data — dataset_v2 is already VLM-free
    train_img_dir, train_ann = find_train_assets(fold_root)
    test_img_dir  = fold_root / "test" / "images"
    test_ann      = fold_root / "test_annotations.json"

    with open(train_ann) as f: train_data = json.load(f)
    with open(test_ann)  as f: test_data  = json.load(f)
    cfg["arc_margin"] = estimate_arc_margin(train_data)

    train_keys = list(train_data.keys())
    test_keys  = list(test_data.keys())

    arch_labels = [infer_arch(train_data[k]) for k in train_keys]
    cm = max(1,sum(1 for a in arch_labels if a=="maxilla"))
    cn = max(1,sum(1 for a in arch_labels if a=="mandible"))
    weights  = [1./(cm if a=="maxilla" else cn) for a in arch_labels]
    sampler  = WeightedRandomSampler(torch.as_tensor(weights,dtype=torch.double),len(weights),True)

    train_ds = HeatmapDentalDataset(train_keys, train_data, train_img_dir, sigma=cfg["sigma"])
    test_ds  = HeatmapDentalDataset(test_keys,  test_data,  test_img_dir,  sigma=cfg["sigma"])
    train_loader = DataLoader(train_ds, BATCH_SIZE, sampler=sampler,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False,     num_workers=4, pin_memory=True)

    model = HeatmapNet().to(DEVICE)
    if _PRETRAINED_PATH and Path(_PRETRAINED_PATH).exists():
        model.load_state_dict(torch.load(_PRETRAINED_PATH, map_location=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    gx = torch.arange(HEATMAP_SIZE, dtype=torch.float32, device=DEVICE).view(1,1,1,-1).expand(1,1,HEATMAP_SIZE,-1)
    gy = torch.arange(HEATMAP_SIZE, dtype=torch.float32, device=DEVICE).view(1,1,-1,1).expand(1,1,-1,HEATMAP_SIZE)

    best_sum3    = float("inf")
    patience_ctr = 0

    for epoch in range(max_epochs):
        model.train()
        for imgs, hms, gts, _ in train_loader:
            imgs, hms, gts = imgs.to(DEVICE), hms.to(DEVICE), gts.to(DEVICE)
            optimizer.zero_grad()
            loss = total_loss(model(imgs), hms, gts, gx, gy, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        if (epoch+1) % EVAL_EVERY == 0:
            sum3 = evaluate(model, test_loader)
            trial.report(sum3, epoch)

            if sum3 < best_sum3:
                best_sum3    = sum3
                patience_ctr = 0
            else:
                patience_ctr += 1

            print(f"  [trial {trial.number} epoch {epoch+1}] sum3={sum3:.3f}mm (best={best_sum3:.3f})")

            if trial.should_prune():
                raise optuna.TrialPruned()

            if patience_ctr >= PATIENCE:
                break

    return best_sum3


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials",   type=int,  default=20)
    ap.add_argument("--max-epochs", type=int,  default=150)
    ap.add_argument("--study-name", type=str,  default=None)
    ap.add_argument("--fold-root",   type=Path, default=FOLD_ROOT,
                    help="Fold used for tuning (default: dataset_v2/fold_1)")
    ap.add_argument("--pretrained",  type=str,  default="",
                    help="Path to pretrained .pth to fine-tune from")
    args = ap.parse_args()

    global _PRETRAINED_PATH
    _PRETRAINED_PATH = args.pretrained
    fold_root = args.fold_root
    print(f"Tuning on  : {fold_root}")
    print(f"Pretrained : {_PRETRAINED_PATH or 'none (ImageNet init)'}")

    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    study_name = args.study_name or f"dental_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage    = f"sqlite:///{SAVE_ROOT}/{study_name}.db"

    study = optuna.create_study(
        study_name    = study_name,
        storage       = storage,
        direction     = "minimize",
        pruner        = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        sampler       = optuna.samplers.TPESampler(seed=42),
        load_if_exists= True,
    )

    study.optimize(
        lambda trial: objective(trial, args.max_epochs, fold_root),
        n_trials          = args.n_trials,
        show_progress_bar = False,
    )

    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"BEST: sum3={best.value:.3f}mm  (trial #{best.number})")
    print(f"{'='*60}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    p = best.params
    print(f"\nRun all 4 folds with best config:")
    print(f"  python train_resnet50.py --k-folds 4 \\")
    print(f"    --dataset-root dataset_v2 \\")
    print(f"    --lr {p['lr']:.2e} \\")
    print(f"    --coord-loss-w {p['coord_w']:.4f} \\")
    print(f"    --length-loss-w {p['len_w']:.4f} \\")
    print(f"    --arc-side-w {p['side_w']:.4f} \\")
    print(f"    --arc-center-w {p['center_w']:.4f} \\")
    print(f"    --arc-contrast-w {p['contrast_w']:.4f} \\")
    print(f"    --sigma {p['sigma']} \\")
    print(f"    --softargmax-temp {p['temp']:.1f}")
    print(f"\nStudy DB : {SAVE_ROOT}/{study_name}.db")
    print(f"Visualise: optuna-dashboard {storage}")


if __name__ == "__main__":
    main()
