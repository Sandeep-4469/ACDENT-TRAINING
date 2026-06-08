"""
GNN baseline — ResNet18 tooth encoder + GATv2 graph layers + MLP decoder.
Uses YOLO detector to extract tooth crops as graph nodes.
Trains separate models for mandible (7 lines) and maxilla (3 lines).
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
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torchvision import models, transforms
from ultralytics import YOLO
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from shared_eval import (
    fold_paths, find_image,
    IMG_SIZE, NUM_KPS,
    evaluate_predictions, print_results,
)

# ── config ───────────────────────────────────────────────────────────────────
YOLO_PATH    = "/data1/sandeep_projects/Dental/train_yolo/runs/yolo11m_medium/weights/best.pt"
BATCH_SIZE   = 4
EPOCHS       = 120
LR           = 1e-4
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_BASE = os.path.join(os.path.dirname(__file__), "results", "gnn")
LOGS_DIR     = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(RESULTS_BASE, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

ARCH_CONFIG = {
    "mandible": {"num_lines": 7},
    "maxilla":  {"num_lines": 3},
}


def setup_logger(name, fold_num):
    log_path = os.path.join(LOGS_DIR, f"{name}_fold{fold_num}.log")
    logger   = logging.getLogger(f"{name}_fold{fold_num}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.FileHandler(log_path, mode="w"))
        logger.addHandler(logging.StreamHandler())
    return logger


def save_loss_plot(histories, title, path):
    plt.figure(figsize=(8, 4))
    for label, h in histories.items():
        plt.plot(h, label=label)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# ── models ────────────────────────────────────────────────────────────────────
class ToothEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.resnet18(weights="DEFAULT")
        self.backbone = nn.Sequential(*list(m.children())[:-1])

    def forward(self, x):
        return self.backbone(x).view(x.size(0), -1)   # (N, 512)


class DentalGNN(nn.Module):
    def __init__(self, num_lines):
        super().__init__()
        self.gnn1 = GATv2Conv(516, 256, heads=4)
        self.gnn2 = GATv2Conv(1024, 128, heads=1)
        self.mlp  = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, num_lines * 4)
        )
        self.num_lines = num_lines

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.gnn1(x, ei))
        x = torch.relu(self.gnn2(x, ei))
        g = global_mean_pool(x, batch)
        return self.mlp(g).view(-1, self.num_lines, 2, 2)


# ── dataset ──────────────────────────────────────────────────────────────────
class GNNDentalDataset(torch.utils.data.Dataset):
    def __init__(self, ann_file, img_dirs, arch, detector, encoder):
        with open(ann_file) as f:
            db = json.load(f)
        # arch field is "Mandible"/"Maxilla" — compare case-insensitively
        self.db      = {k: v for k, v in db.items()
                        if v.get("arch", "").lower() == arch}
        self.keys    = list(self.db.keys())
        self.img_dirs = img_dirs if isinstance(img_dirs, list) else [img_dirs]
        self.detector = detector
        self.encoder  = encoder
        self.arch     = arch
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.keys)

    def _find_img(self, key):
        return find_image(key, *self.img_dirs)

    def __getitem__(self, idx):
        key   = self.keys[idx]
        entry = self.db[key]
        path  = self._find_img(key)
        if path is None:
            return self.__getitem__((idx+1) % len(self))

        img = cv2.imread(path)
        if img is None:
            return self.__getitem__((idx+1) % len(self))
        H, W = img.shape[:2]

        res = self.detector(img, verbose=False)[0]
        if res.boxes is None or len(res.boxes) < 2:
            dummy      = torch.zeros(1, 516)
            edge_index = torch.zeros(2, 0, dtype=torch.long)
        else:
            boxes_px = res.boxes.xyxy.cpu().numpy()
            boxes_n  = res.boxes.xywhn.cpu().numpy()
            order    = np.argsort(boxes_px[:, 0])
            boxes_px = boxes_px[order]
            boxes_n  = boxes_n[order]

            feats = []
            for (x1, y1, x2, y2) in boxes_px:
                crop = img[int(y1):int(y2), int(x1):int(x2)]
                if crop.size == 0:
                    crop = np.zeros((64,64,3), np.uint8)
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                with torch.no_grad():
                    f = self.encoder(self.tf(crop).unsqueeze(0).to(DEVICE)).view(-1)
                feats.append(f.cpu())

            feats   = torch.stack(feats)
            spatial = torch.tensor(boxes_n, dtype=torch.float32)
            node_x  = torch.cat([feats, spatial], dim=1)

            edges = []
            for i in range(len(node_x)-1):
                edges += [[i, i+1], [i+1, i]]
            edge_index = torch.tensor(edges).t().contiguous() if edges else torch.zeros(2,0,dtype=torch.long)
            dummy = node_x

        num_lines = ARCH_CONFIG[self.arch]["num_lines"]
        gt = torch.zeros(num_lines, 2, 2, dtype=torch.float32)
        for i, line in enumerate(entry["lines"]):
            if i >= num_lines:
                break
            p1, p2 = line
            gt[i, 0] = torch.tensor([p1[0]/W, p1[1]/H])
            gt[i, 1] = torch.tensor([p2[0]/W, p2[1]/H])

        data         = Data(x=dummy, edge_index=edge_index)
        data.graph_y = gt
        data.key     = key
        return data


def gnn_loss(pred, gt):
    l_pt  = torch.mean((pred - gt)**2)
    l_len = torch.mean(torch.abs(
        torch.norm(pred[:,:,0] - pred[:,:,1], dim=2) -
        torch.norm(gt[:,:,0]   - gt[:,:,1],   dim=2)
    ))
    return l_pt + 2.0 * l_len


# ── train one arch ────────────────────────────────────────────────────────────
def train_arch(arch, detector, encoder, logger, results_dir,
               train_ann, train_img, train_aug_img, test_ann, test_img):
    num_lines = ARCH_CONFIG[arch]["num_lines"]
    logger.info(f"\n--- Training GNN for {arch} ({num_lines} lines) ---")

    train_ds = GNNDentalDataset(train_ann, [train_img, train_aug_img], arch, detector, encoder)
    test_ds  = GNNDentalDataset(test_ann,  [test_img],                 arch, detector, encoder)
    logger.info(f"  Train: {len(train_ds)}, Test: {len(test_ds)}")

    train_dl = GeoDataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl  = GeoDataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    model     = DentalGNN(num_lines).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    best_loss = float("inf")
    history   = []

    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            batch = batch.to(DEVICE)
            gt    = batch.graph_y.view(-1, num_lines, 2, 2)
            pred  = model(batch)
            loss  = gnn_loss(pred, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        avg = total / len(train_dl)
        history.append(avg)
        logger.info(f"  Epoch {epoch+1:3d} | Loss: {avg:.6f}")
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(),
                       os.path.join(results_dir, f"gnn_{arch}_best.pth"))
            logger.info(f"    → Best saved")

    # ── evaluate ──────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(
        os.path.join(results_dir, f"gnn_{arch}_best.pth"), map_location=DEVICE))
    model.eval()

    with open(test_ann) as f:
        test_annotations = json.load(f)

    predictions = {}
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Evaluating"):
            batch = batch.to(DEVICE)
            preds = model(batch)
            keys_batch = batch.key if isinstance(batch.key, list) else [batch.key]

            for i, key in enumerate(keys_batch):
                if key not in test_annotations:
                    continue
                img_path = test_ds._find_img(key)
                if img_path is None:
                    continue
                img_real = cv2.imread(img_path)
                if img_real is None:
                    continue
                H_real, W_real = img_real.shape[:2]

                p = preds[i].cpu().numpy()
                pts = []
                for li in range(num_lines):
                    pts.append([float(p[li,0,0]*W_real), float(p[li,0,1]*H_real)])
                    pts.append([float(p[li,1,0]*W_real), float(p[li,1,1]*H_real)])
                predictions[key] = {"predicted_keypoints": pts}

    return predictions, history


def train(fold_num=1):
    results_dir = os.path.join(RESULTS_BASE, f"fold_{fold_num}")
    os.makedirs(results_dir, exist_ok=True)

    logger = setup_logger("gnn", fold_num)
    logger.info(f"=== GNN — Fold {fold_num} ===")
    logger.info(f"Device: {DEVICE}")

    train_ann, train_aug_ann, test_ann, train_img, train_aug_img, test_img = fold_paths(fold_num)

    detector = YOLO(YOLO_PATH)
    encoder  = ToothEncoder().to(DEVICE).eval()

    all_predictions = {}
    all_histories   = {}
    for arch in ["mandible", "maxilla"]:
        preds, hist = train_arch(
            arch, detector, encoder, logger, results_dir,
            train_ann, train_img, train_aug_img, test_ann, test_img
        )
        all_predictions.update(preds)
        all_histories[arch] = hist

    save_loss_plot(all_histories, f"GNN — Fold {fold_num} Loss",
                   os.path.join(LOGS_DIR, f"gnn_fold{fold_num}_loss.png"))

    with open(test_ann) as f:
        test_annotations = json.load(f)

    metrics = evaluate_predictions(all_predictions, test_annotations)
    print_results(f"GNN (fold {fold_num})", metrics)
    logger.info(str(metrics))

    with open(os.path.join(results_dir, "predictions.json"), "w") as f:
        json.dump(all_predictions, f, indent=2)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

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
