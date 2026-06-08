"""
SimCC baseline — ResNet50 + separate x/y classification heads, KL-divergence loss.
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
    IMG_SIZE, NUM_KPS,
    evaluate_predictions, print_results,
)

# ── config ───────────────────────────────────────────────────────────────────
SIMCC_SPLIT = 2.0
BIN_X = int(IMG_SIZE * SIMCC_SPLIT)   # 1024
BIN_Y = int(IMG_SIZE * SIMCC_SPLIT)   # 1024
SIGMA = 2.0

BATCH_SIZE   = 16
EPOCHS       = 200
LR           = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 20
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_BASE = os.path.join(os.path.dirname(__file__), "results", "simcc")
LOGS_DIR     = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(RESULTS_BASE, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


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


# ── dataset ──────────────────────────────────────────────────────────────────
class SimCCDataset(Dataset):
    def __init__(self, ann_file, img_dirs, augment=False):
        with open(ann_file) as f:
            self.data = json.load(f)
        self.keys     = list(self.data.keys())
        self.img_dirs = img_dirs if isinstance(img_dirs, list) else [img_dirs]
        self.augment  = augment
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.05) if augment else transforms.Lambda(lambda x: x),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.keys)

    def _find_img(self, key):
        return find_image(key, *self.img_dirs)

    def _simcc_label(self, coord_norm, bins):
        mu = coord_norm * bins
        x  = torch.arange(bins).float()
        g  = torch.exp(-0.5 * ((x - mu) / SIGMA)**2)
        return g / (g.sum() + 1e-6)

    def __getitem__(self, idx):
        key  = self.keys[idx]
        item = self.data[key]
        path = self._find_img(key)
        if path is None:
            return self.__getitem__((idx+1) % len(self))

        img = cv2.imread(path)
        if img is None:
            return self.__getitem__((idx+1) % len(self))
        h, w = img.shape[:2]
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img  = self.transform(img)

        flat_pts  = [pt for line in item["lines"] for pt in line]
        n_valid   = len(flat_pts)
        arch      = item.get("arch", "").lower()
        arch_flag = torch.tensor(0 if arch == "mandible" else 1, dtype=torch.long)

        tx   = torch.zeros(NUM_KPS, BIN_X)
        ty   = torch.zeros(NUM_KPS, BIN_Y)
        mask = torch.zeros(NUM_KPS)

        for i, (x, y) in enumerate(flat_pts[:NUM_KPS]):
            tx[i]   = self._simcc_label(x / w, BIN_X)
            ty[i]   = self._simcc_label(y / h, BIN_Y)
            mask[i] = 1.0

        raw  = torch.full((NUM_KPS, 2), -1.0)
        sx, sy = IMG_SIZE/w, IMG_SIZE/h
        for i, (x, y) in enumerate(flat_pts[:NUM_KPS]):
            raw[i] = torch.tensor([x*sx, y*sy])

        return img, tx, ty, mask, arch_flag, raw, key


def collate(batch):
    imgs, txs, tys, masks, archs, raws, keys = zip(*batch)
    return (torch.stack(imgs), torch.stack(txs), torch.stack(tys),
            torch.stack(masks), torch.stack(archs),
            torch.stack(raws), list(keys))


# ── model ─────────────────────────────────────────────────────────────────────
class SimCCNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.conv     = nn.Conv2d(2048, 512, 1)
        self.arch_emb = nn.Embedding(2, 32)
        self.fc       = nn.Sequential(nn.Linear(512+32, 1024), nn.ReLU())
        self.x_head   = nn.Linear(1024, NUM_KPS * BIN_X)
        self.y_head   = nn.Linear(1024, NUM_KPS * BIN_Y)

    def forward(self, x, arch_idx):
        feat = self.backbone(x)
        feat = F.adaptive_avg_pool2d(self.conv(feat), 1).flatten(1)
        arch = self.arch_emb(arch_idx)
        feat = self.fc(torch.cat([feat, arch], dim=1))
        px   = self.x_head(feat).view(-1, NUM_KPS, BIN_X)
        py   = self.y_head(feat).view(-1, NUM_KPS, BIN_Y)
        return px, py


# ── loss ──────────────────────────────────────────────────────────────────────
def simcc_loss(px, py, tx, ty, mask):
    kl     = nn.KLDivLoss(reduction="none")
    loss_x = kl(F.log_softmax(px, dim=-1), tx).sum(dim=-1)
    loss_y = kl(F.log_softmax(py, dim=-1), ty).sum(dim=-1)
    return ((loss_x + loss_y) * mask).sum() / (mask.sum() + 1e-6)


def decode_simcc(px, py):
    """px, py: (K, BIN) tensors → list of [x, y] in IMG_SIZE space"""
    coords = []
    for i in range(px.shape[0]):
        xi = px[i].argmax().item() / BIN_X * IMG_SIZE
        yi = py[i].argmax().item() / BIN_Y * IMG_SIZE
        coords.append([float(xi), float(yi)])
    return coords


# ── training ──────────────────────────────────────────────────────────────────
def train(fold_num=1):
    results_dir = os.path.join(RESULTS_BASE, f"fold_{fold_num}")
    os.makedirs(results_dir, exist_ok=True)

    logger = setup_logger("simcc", fold_num)
    logger.info(f"=== SimCC — Fold {fold_num} ===")
    logger.info(f"Device: {DEVICE}")

    train_ann, train_aug_ann, test_ann, train_img, train_aug_img, test_img = fold_paths(fold_num)

    train_ds = SimCCDataset(train_aug_ann, [train_img, train_aug_img], augment=True)
    test_ds  = SimCCDataset(test_ann,      [test_img],                 augment=False)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, collate_fn=collate)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True, collate_fn=collate)

    logger.info(f"Train: {len(train_ds)} samples, Test: {len(test_ds)} samples")

    model     = SimCCNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_loss      = float("inf")
    patience_count = 0
    history        = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for imgs, txs, tys, masks, archs, _, _ in tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            imgs, txs, tys = imgs.to(DEVICE), txs.to(DEVICE), tys.to(DEVICE)
            masks, archs   = masks.to(DEVICE), archs.to(DEVICE)
            px, py = model(imgs, archs)
            loss   = simcc_loss(px, py, txs, tys, masks)
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
            torch.save(model.state_dict(), os.path.join(results_dir, "simcc_best.pth"))
            logger.info(f"  → Best saved (loss={best_loss:.6f})")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    save_loss_plot(history, f"SimCC — Fold {fold_num} Loss",
                   os.path.join(LOGS_DIR, f"simcc_fold{fold_num}_loss.png"))

    # ── evaluation ────────────────────────────────────────────────────────────
    logger.info("Evaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(results_dir, "simcc_best.pth"), map_location=DEVICE))
    model.eval()

    with open(test_ann) as f:
        test_annotations = json.load(f)

    predictions = {}
    with torch.no_grad():
        for imgs, txs, tys, masks, archs, _, keys in tqdm(test_dl, desc="Evaluating"):
            imgs, archs = imgs.to(DEVICE), archs.to(DEVICE)
            px, py = model(imgs, archs)
            for i, key in enumerate(keys):
                pred_pts = decode_simcc(px[i].cpu(), py[i].cpu())
                n_valid  = int(masks[i].sum().item())
                predictions[key] = {"predicted_keypoints": pred_pts[:n_valid]}

    metrics = evaluate_predictions(predictions, test_annotations)
    print_results(f"SimCC (fold {fold_num})", metrics)
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
    parser.add_argument("--fold", type=int, default=1, choices=range(1, 6))
    parser.add_argument("--all-folds", action="store_true")
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
