"""
Heatmap baseline — ResNet50 + 3-stage deconv, masked MSE, NO geometry constraints.
Adapted for DSAA_Dental 5-fold cross-validation dataset.
"""
import os, sys, json, logging, argparse
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from shared_eval import (
    fold_paths, find_image,
    IMG_SIZE, HEATMAP_SIZE, NUM_KPS,
    evaluate_predictions, print_results,
)

# ── config ──────────────────────────────────────────────────────────────────
BATCH_SIZE   = 8
EPOCHS       = 200
LR           = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 20
SIGMA        = 3
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_BASE = os.path.join(os.path.dirname(__file__), "results", "heatmap")
LOGS_DIR     = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(RESULTS_BASE, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


# ── dataset ──────────────────────────────────────────────────────────────────
class DentalHeatmapDataset(Dataset):
    def __init__(self, ann_file, img_dirs, augment=False):
        with open(ann_file) as f:
            self.data = json.load(f)
        self.keys     = list(self.data.keys())
        self.img_dirs = img_dirs if isinstance(img_dirs, list) else [img_dirs]
        self.augment  = augment
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.keys)

    def _find_img(self, key):
        return find_image(key, *self.img_dirs)

    def _gaussian_heatmap(self, pts, valid_mask):
        hm = np.zeros((NUM_KPS, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
        for i, (x, y) in enumerate(pts):
            if i >= NUM_KPS or not valid_mask[i]:
                continue
            hx = int(x * HEATMAP_SIZE / IMG_SIZE)
            hy = int(y * HEATMAP_SIZE / IMG_SIZE)
            xx, yy = np.meshgrid(np.arange(HEATMAP_SIZE), np.arange(HEATMAP_SIZE))
            hm[i] = np.exp(-((xx - hx)**2 + (yy - hy)**2) / (2 * SIGMA**2))
        return hm

    def _augment(self, img, pts):
        h, w = img.shape[:2]
        angle = np.random.uniform(-8, 8)
        scale = np.random.uniform(0.94, 1.06)
        tx = np.random.uniform(-0.04*w, 0.04*w)
        ty = np.random.uniform(-0.04*h, 0.04*h)
        M  = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
        M[:,2] += [tx, ty]
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        new_pts = []
        for (x, y) in pts:
            px = M[0,0]*x + M[0,1]*y + M[0,2]
            py = M[1,0]*x + M[1,1]*y + M[1,2]
            new_pts.append([px, py])
        return img, new_pts

    def __getitem__(self, idx):
        key  = self.keys[idx]
        item = self.data[key]
        path = self._find_img(key)
        if path is None:
            return self.__getitem__((idx+1) % len(self))

        img = cv2.imread(path)
        if img is None:
            return self.__getitem__((idx+1) % len(self))
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]

        flat_pts = [pt for line in item["lines"] for pt in line]
        flat_pts = flat_pts[:NUM_KPS]
        n_valid  = len(flat_pts)

        pts        = list(flat_pts) + [[-1,-1]] * (NUM_KPS - n_valid)
        valid_mask = [1]*n_valid + [0]*(NUM_KPS - n_valid)

        if self.augment:
            img, aug_valid = self._augment(img, pts[:n_valid])
            for i, vp in enumerate(aug_valid):
                pts[i] = list(vp)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        sx, sy = IMG_SIZE/W, IMG_SIZE/H
        pts_scaled = []
        for i, p in enumerate(pts):
            if valid_mask[i]:
                pts_scaled.append([p[0]*sx, p[1]*sy])
            else:
                pts_scaled.append([-1.0, -1.0])

        hm     = self._gaussian_heatmap(pts_scaled, valid_mask)
        gt_pts = np.full((NUM_KPS, 2), -1.0, dtype=np.float32)
        for i, p in enumerate(pts_scaled):
            gt_pts[i] = p

        img_t = self.normalize(img)
        mask  = torch.tensor(valid_mask, dtype=torch.float32)
        return img_t, torch.tensor(hm), gt_pts, mask, key


def collate(batch):
    imgs, hms, gt_pts, masks, keys = zip(*batch)
    gt_arr = np.stack([g if g.shape == (NUM_KPS, 2) else
                       np.full((NUM_KPS, 2), -1.0, dtype=np.float32)
                       for g in gt_pts])
    return (torch.stack(imgs), torch.stack(hms),
            torch.tensor(gt_arr, dtype=torch.float32),
            torch.stack(masks), list(keys))


# ── model ─────────────────────────────────────────────────────────────────────
class HeatmapNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.deconv  = nn.Sequential(
            nn.ConvTranspose2d(2048, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256,  256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256,  256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.head = nn.Conv2d(256, NUM_KPS, 1)

    def forward(self, x):
        return self.head(self.deconv(self.encoder(x)))


# ── training ──────────────────────────────────────────────────────────────────
def masked_mse(pred, gt, mask):
    loss = ((pred - gt)**2).mean(dim=(2,3))   # (B, K)
    return (loss * mask).sum() / (mask.sum() + 1e-6)


def decode_heatmap(hm):
    """hm: (K, H, W) numpy → list of [x, y] in IMG_SIZE space"""
    coords = []
    for i in range(hm.shape[0]):
        y, x = np.unravel_index(np.argmax(hm[i]), hm[i].shape)
        coords.append([float(x) * IMG_SIZE / HEATMAP_SIZE,
                       float(y) * IMG_SIZE / HEATMAP_SIZE])
    return coords


def setup_logger(name, fold_num):
    log_path = os.path.join(LOGS_DIR, f"{name}_fold{fold_num}.log")
    logger   = logging.getLogger(f"{name}_fold{fold_num}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.FileHandler(log_path, mode="w"))
        logger.addHandler(logging.StreamHandler())
    return logger


def save_loss_plot(history, title, path):
    plt.figure(figsize=(8, 4))
    plt.plot(history, label="Train Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def train(fold_num=1):
    results_dir = os.path.join(RESULTS_BASE, f"fold_{fold_num}")
    os.makedirs(results_dir, exist_ok=True)

    logger = setup_logger("heatmap", fold_num)
    logger.info(f"=== Heatmap Baseline — Fold {fold_num} ===")
    logger.info(f"Device: {DEVICE}")

    train_ann, train_aug_ann, test_ann, train_img, train_aug_img, test_img = fold_paths(fold_num)

    train_ds = DentalHeatmapDataset(train_aug_ann, [train_img, train_aug_img], augment=True)
    test_ds  = DentalHeatmapDataset(test_ann,      [test_img],                 augment=False)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, collate_fn=collate)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True, collate_fn=collate)

    logger.info(f"Train: {len(train_ds)} samples, Test: {len(test_ds)} samples")

    model     = HeatmapNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_loss      = float("inf")
    patience_count = 0
    history        = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for imgs, hms, _, masks, _ in tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            imgs, hms, masks = imgs.to(DEVICE), hms.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            loss  = masked_mse(preds, hms, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dl)
        scheduler.step(avg_loss)
        history.append(avg_loss)
        logger.info(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss      = avg_loss
            patience_count = 0
            torch.save(model.state_dict(), os.path.join(results_dir, "heatmap_best.pth"))
            logger.info(f"  → Best saved (loss={best_loss:.6f})")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    save_loss_plot(history, f"Heatmap — Fold {fold_num} Loss",
                   os.path.join(LOGS_DIR, f"heatmap_fold{fold_num}_loss.png"))

    # ── evaluation ────────────────────────────────────────────────────────────
    logger.info("\nEvaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(results_dir, "heatmap_best.pth"), map_location=DEVICE))
    model.eval()

    with open(test_ann) as f:
        test_annotations = json.load(f)

    predictions = {}
    with torch.no_grad():
        for imgs, _, _, masks, keys in tqdm(test_dl, desc="Evaluating"):
            imgs  = imgs.to(DEVICE)
            preds = model(imgs)
            for i, key in enumerate(keys):
                hm_np    = preds[i].cpu().numpy()
                pred_pts = decode_heatmap(hm_np)
                n_valid  = int(masks[i].sum().item())
                predictions[key] = {"predicted_keypoints": pred_pts[:n_valid]}

    metrics = evaluate_predictions(predictions, test_annotations)
    print_results(f"Heatmap Baseline (fold {fold_num})", metrics)
    logger.info(str(metrics))

    with open(os.path.join(results_dir, "predictions.json"), "w") as f:
        json.dump(predictions, f, indent=2)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(results_dir, "train_history.json"), "w") as f:
        json.dump(history, f)

    logger.info(f"Results saved to {results_dir}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=1, choices=range(1, 6),
                        help="Fold number 1-5 (default: 1)")
    parser.add_argument("--all-folds", action="store_true",
                        help="Run all 5 folds sequentially")
    args = parser.parse_args()

    if args.all_folds:
        all_metrics = {}
        for f in range(1, 6):
            all_metrics[f] = train(fold_num=f)
        print("\n=== Cross-fold Summary ===")
        for key in ["incisor_sum", "left_arc", "right_arc", "scale"]:
            vals = [m[key] for m in all_metrics.values() if key in m]
            if vals:
                print(f"  {key}: {np.mean(vals):.2f} ± {np.std(vals):.2f} mm")
    else:
        train(fold_num=args.fold)
