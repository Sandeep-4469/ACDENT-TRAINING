"""
YOLO-Pose training + evaluation for DSAA_Dental.
Step 1: Run prepare_yolo_dataset.py --fold N first.
Step 2: Train YOLO11s-pose.
Step 3: Evaluate predictions with the shared metric.
"""
import os, sys, json, argparse
import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(__file__))
from shared_eval import (
    fold_paths, find_image,
    IMG_SIZE, NUM_KPS,
    evaluate_predictions, print_results,
)

YOLO_DATASET_BASE = os.path.join(os.path.dirname(__file__), "yolo_dataset")
RESULTS_BASE      = os.path.join(os.path.dirname(__file__), "results", "yolo_pose")
os.makedirs(RESULTS_BASE, exist_ok=True)


def train_fold(fold_num):
    yaml_path   = os.path.join(YOLO_DATASET_BASE, f"fold_{fold_num}", "dental_pose.yaml")
    results_dir = os.path.join(RESULTS_BASE, f"fold_{fold_num}")
    os.makedirs(results_dir, exist_ok=True)

    print(f"=== YOLO-Pose Training — Fold {fold_num} ===")
    model = YOLO("yolo11s-pose.pt")
    model.train(
        data=yaml_path,
        epochs=300,
        imgsz=IMG_SIZE,
        batch=16,
        patience=50,
        pose=15.0, kobj=2.0, box=7.5,
        degrees=15.0, translate=0.1, scale=0.5,
        shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.5,
        mosaic=0.5, mixup=0.1,
        optimizer="AdamW", lr0=0.001, weight_decay=0.0005,
        project=results_dir,
        name="run",
        exist_ok=True,
    )
    best_weights = os.path.join(results_dir, "run", "weights", "best.pt")
    print(f"Best weights: {best_weights}")
    return best_weights


def evaluate_fold(fold_num, weights_path):
    print(f"\n=== Evaluating YOLO-Pose Fold {fold_num}: {weights_path} ===")
    model = YOLO(weights_path)

    _, _, test_ann, _, _, test_img = fold_paths(fold_num)
    results_dir = os.path.join(RESULTS_BASE, f"fold_{fold_num}")
    os.makedirs(results_dir, exist_ok=True)

    with open(test_ann) as f:
        test_annotations = json.load(f)

    predictions = {}
    for key, item in test_annotations.items():
        img_path = find_image(key, test_img)
        if img_path is None:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]

        res = model(img, verbose=False)[0]
        if res.keypoints is None or len(res.keypoints.xy) == 0:
            continue

        kps = res.keypoints.xy[0].cpu().numpy()   # (14, 2) in original image space
        sx, sy = IMG_SIZE / W, IMG_SIZE / H
        pred_pts = [[float(kps[i,0]*sx), float(kps[i,1]*sy)]
                    for i in range(min(NUM_KPS, len(kps)))]

        n_valid = sum(1 for line in item["lines"] for _ in line)
        predictions[key] = {"predicted_keypoints": pred_pts[:n_valid]}

    metrics = evaluate_predictions(predictions, test_annotations)
    print_results(f"YOLO-Pose (fold {fold_num})", metrics)

    with open(os.path.join(results_dir, "predictions.json"), "w") as f:
        json.dump(predictions, f, indent=2)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Results saved to {results_dir}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=1, choices=range(1, 6))
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument("--eval-only", type=str, default=None,
                        help="Path to existing weights; skip training")
    args = parser.parse_args()

    folds = range(1, 6) if args.all_folds else [args.fold]
    for f in folds:
        if args.eval_only:
            evaluate_fold(f, args.eval_only)
        else:
            weights = train_fold(f)
            evaluate_fold(f, weights)
