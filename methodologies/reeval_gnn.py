"""
Re-evaluate GNN with the corrected coordinate denormalization.
Uses already-trained weights for a given fold — no retraining needed.
"""
import os, sys, json, argparse
import cv2
import numpy as np
import torch
from torch_geometric.loader import DataLoader as GeoDataLoader
from ultralytics import YOLO
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from train_gnn import (
    DentalGNN, GNNDentalDataset, ToothEncoder,
    YOLO_PATH, RESULTS_BASE, DEVICE, ARCH_CONFIG, BATCH_SIZE,
)
from shared_eval import fold_paths, evaluate_predictions, print_results


def reeval_arch(arch, detector, encoder, fold_num):
    results_dir = os.path.join(RESULTS_BASE, f"fold_{fold_num}")
    num_lines   = ARCH_CONFIG[arch]["num_lines"]
    weights     = os.path.join(results_dir, f"gnn_{arch}_best.pth")

    if not os.path.exists(weights):
        print(f"[SKIP] No weights for fold {fold_num} {arch}: {weights}")
        return {}

    _, _, test_ann, _, _, test_img = fold_paths(fold_num)

    print(f"\n--- Re-evaluating GNN fold {fold_num} {arch} ---")
    test_ds = GNNDentalDataset(test_ann, [test_img], arch, detector, encoder)
    test_dl = GeoDataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = DentalGNN(num_lines).to(DEVICE)
    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    model.eval()

    with open(test_ann) as f:
        test_annotations = json.load(f)

    predictions = {}
    with torch.no_grad():
        for batch in tqdm(test_dl, desc=f"Eval {arch}"):
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

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=1, choices=range(1, 6))
    args = parser.parse_args()

    fold_num = args.fold
    _, _, test_ann, _, _, _ = fold_paths(fold_num)
    results_dir = os.path.join(RESULTS_BASE, f"fold_{fold_num}")

    detector = YOLO(YOLO_PATH)
    encoder  = ToothEncoder().to(DEVICE).eval()

    all_preds = {}
    for arch in ["mandible", "maxilla"]:
        all_preds.update(reeval_arch(arch, detector, encoder, fold_num))

    with open(test_ann) as f:
        test_annotations = json.load(f)

    metrics = evaluate_predictions(all_preds, test_annotations)
    print_results(f"GNN corrected (fold {fold_num})", metrics)

    with open(os.path.join(results_dir, "predictions.json"), "w") as f:
        json.dump(all_preds, f, indent=2)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nCorrected results saved to {results_dir}")
