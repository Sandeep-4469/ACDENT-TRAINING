"""
Shared evaluation utilities for DSAA_Dental methodology comparison.
Supports 5-fold cross-validation; callers pass fold_num (1-5) to get
the correct annotation/image paths for that fold.
"""
import json
import os
import numpy as np

DSAA_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(DSAA_ROOT, "dataset")
SOURCE_IMGS  = os.path.join(DSAA_ROOT, "dataset_source", "images_512")

IMG_SIZE     = 512
HEATMAP_SIZE = 128
NUM_KPS      = 14   # max keypoints (7 lines × 2 pts for mandible)


def fold_paths(fold_num: int):
    """Return (train_ann, train_aug_ann, test_ann, train_img, train_aug_img, test_img)."""
    fold_dir      = os.path.join(DATASET_ROOT, f"fold_{fold_num}")
    train_ann     = os.path.join(fold_dir, "train_annotations.json")
    train_aug_ann = os.path.join(fold_dir, "train_annotations_augmented.json")
    test_ann      = os.path.join(fold_dir, "test_annotations.json")
    train_img     = os.path.join(fold_dir, "train", "images")
    train_aug_img = os.path.join(fold_dir, "train_augmented", "images")
    test_img      = os.path.join(fold_dir, "test", "images")
    return train_ann, train_aug_ann, test_ann, train_img, train_aug_img, test_img


def find_image(key, *img_dirs):
    """Return image path for a given annotation key, searching provided dirs then source."""
    for d in img_dirs:
        for ext in (".jpg", ".png"):
            p = os.path.join(d, f"{key}{ext}")
            if os.path.exists(p):
                return p
    for ext in (".jpg", ".png"):
        p = os.path.join(SOURCE_IMGS, f"{key}{ext}")
        if os.path.exists(p):
            return p
    return None


def line_len(p1, p2):
    return float(np.linalg.norm(np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)))


def evaluate_predictions(predictions, test_annotations):
    """
    Args:
        predictions:      {key: {'predicted_keypoints': [[x, y], ...]}}
                          coordinates in IMG_SIZE (512) pixel space
        test_annotations: loaded test_annotations.json dict
    Returns:
        dict of mean errors (mm)
    """
    errors = {"scale": [], "incisor_sum": [], "left_arc": [], "right_arc": []}

    for key, pred in predictions.items():
        if key not in test_annotations:
            continue
        item      = test_annotations[key]
        pred_pts  = pred["predicted_keypoints"]
        gt_lines  = item["lines"]
        gt_pts    = [pt for line in gt_lines for pt in line]
        valid_kps = len(gt_pts)

        if len(pred_pts) < 2 or valid_kps < 2:
            continue

        # Per-sample calibration
        px_per_mm = item.get("pixel_per_mm", None)
        if px_per_mm is None or px_per_mm < 1e-6:
            continue
        px_to_mm = 1.0 / px_per_mm

        # Scale line (line 0)
        scale_pred = line_len(pred_pts[0], pred_pts[1])
        scale_gt   = line_len(gt_pts[0],   gt_pts[1])
        errors["scale"].append(abs(scale_pred - scale_gt) * px_to_mm)

        arch = item.get("arch", "").lower()

        if arch == "maxilla" and valid_kps == 6:   # 3 lines × 2 pts
            pred_arcs, gt_arcs = [], []
            for li in [1, 2]:
                if len(pred_pts) > 2 * li + 1:
                    p1, p2 = pred_pts[2*li], pred_pts[2*li+1]
                    g1, g2 = gt_pts[2*li],   gt_pts[2*li+1]
                    pred_arcs.append(((p1[0]+p2[0])/2.0, line_len(p1, p2)))
                    gt_arcs.append(  ((g1[0]+g2[0])/2.0, line_len(g1, g2)))
            if len(pred_arcs) == 2:
                pred_arcs.sort(key=lambda x: x[0])
                gt_arcs.sort(  key=lambda x: x[0])
                errors["left_arc"].append( abs(pred_arcs[0][1] - gt_arcs[0][1]) * px_to_mm)
                errors["right_arc"].append(abs(pred_arcs[1][1] - gt_arcs[1][1]) * px_to_mm)

        elif arch == "mandible" and valid_kps == 14:  # 7 lines × 2 pts
            pred_inc = sum(line_len(pred_pts[2*li], pred_pts[2*li+1]) for li in [1,2,3,4]
                           if len(pred_pts) > 2*li+1)
            gt_inc   = sum(line_len(gt_pts[2*li], gt_pts[2*li+1]) for li in [1,2,3,4])
            errors["incisor_sum"].append(abs(pred_inc - gt_inc) * px_to_mm)

            pred_arcs, gt_arcs = [], []
            for li in [5, 6]:
                if len(pred_pts) > 2*li+1:
                    p1, p2 = pred_pts[2*li], pred_pts[2*li+1]
                    g1, g2 = gt_pts[2*li],   gt_pts[2*li+1]
                    pred_arcs.append(((p1[0]+p2[0])/2.0, line_len(p1, p2)))
                    gt_arcs.append(  ((g1[0]+g2[0])/2.0, line_len(g1, g2)))
            if len(pred_arcs) == 2:
                pred_arcs.sort(key=lambda x: x[0])
                gt_arcs.sort(  key=lambda x: x[0])
                errors["left_arc"].append( abs(pred_arcs[0][1] - gt_arcs[0][1]) * px_to_mm)
                errors["right_arc"].append(abs(pred_arcs[1][1] - gt_arcs[1][1]) * px_to_mm)

    means = {k: float(np.mean(v)) for k, v in errors.items() if v}
    return means


def print_results(method_name, metrics):
    print(f"\n{'='*55}")
    print(f"  {method_name}")
    print(f"{'='*55}")
    keys   = ["incisor_sum", "left_arc", "right_arc", "scale"]
    labels = ["Mand. Incisor Sum", "L Arc", "R Arc", "Scale"]
    for k, label in zip(keys, labels):
        val = metrics.get(k, float("nan"))
        print(f"  {label:<20}: {val:.2f} mm")
    print(f"{'='*55}")
