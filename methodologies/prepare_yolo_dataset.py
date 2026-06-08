"""
Convert DSAA_Dental fold annotations → YOLO-Pose dataset format.
Output: methodologies/yolo_dataset/fold_X/{train,val}/{images,labels}/
Run once per fold before train_yolo_pose.py.
"""
import os, sys, json, shutil, argparse
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from shared_eval import (
    fold_paths, find_image,
    IMG_SIZE, NUM_KPS,
)

OUT_BASE = os.path.join(os.path.dirname(__file__), "yolo_dataset")


def ann_to_yolo(key, item, img_w, img_h):
    """
    Returns one YOLO label line:
    class cx cy bw bh [kpx kpy kpv] * 14
    class: 0=mandible, 1=maxilla
    """
    arch = item.get("arch", "").lower()
    cls  = 0 if arch == "mandible" else 1

    flat_pts = [pt for line in item["lines"] for pt in line]
    n_valid  = len(flat_pts)

    xs = [p[0] for p in flat_pts]
    ys = [p[1] for p in flat_pts]
    x1, x2 = max(0, min(xs)-5), min(img_w, max(xs)+5)
    y1, y2 = max(0, min(ys)-5), min(img_h, max(ys)+5)
    cx = ((x1+x2)/2) / img_w
    cy = ((y1+y2)/2) / img_h
    bw = (x2-x1) / img_w
    bh = (y2-y1) / img_h

    kp_str = ""
    for i in range(NUM_KPS):
        if i < n_valid:
            x, y = flat_pts[i]
            kp_str += f" {x/img_w:.6f} {y/img_h:.6f} 2"
        else:
            kp_str += " 0 0 0"

    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}{kp_str}"


def write_split(ann_file, img_dirs, out_dir):
    img_out = os.path.join(out_dir, "images")
    lbl_out = os.path.join(out_dir, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    with open(ann_file) as f:
        ann = json.load(f)

    skipped = 0
    for key, item in ann.items():
        img_path = find_image(key, *img_dirs)
        if img_path is None:
            skipped += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]

        label_line = ann_to_yolo(key, item, w, h)
        safe_key   = key.replace(" ", "_").replace("/", "_")
        dst_img    = os.path.join(img_out, f"{safe_key}.jpg")
        dst_lbl    = os.path.join(lbl_out, f"{safe_key}.txt")

        shutil.copy2(img_path, dst_img)
        with open(dst_lbl, "w") as f:
            f.write(label_line + "\n")

    print(f"  {os.path.basename(out_dir)}: {len(ann)-skipped} written, {skipped} skipped")


def write_yaml(fold_num, out_dir):
    yaml_path = os.path.join(out_dir, "dental_pose.yaml")
    content = f"""path: {out_dir}
train: train/images
val:   val/images
kpt_shape: [{NUM_KPS}, 3]
names:
  0: Mandible
  1: Maxilla
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"YAML written: {yaml_path}")
    return yaml_path


def prepare_fold(fold_num):
    out_dir = os.path.join(OUT_BASE, f"fold_{fold_num}")
    os.makedirs(out_dir, exist_ok=True)

    train_ann, train_aug_ann, test_ann, train_img, train_aug_img, test_img = fold_paths(fold_num)

    print(f"\n=== Preparing YOLO-Pose Dataset — Fold {fold_num} ===")
    write_split(train_aug_ann, [train_img, train_aug_img], os.path.join(out_dir, "train"))
    write_split(test_ann,      [test_img],                 os.path.join(out_dir, "val"))
    yaml_path = write_yaml(fold_num, out_dir)
    print(f"Dataset ready: {out_dir}")
    return yaml_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=1, choices=range(1, 6))
    parser.add_argument("--all-folds", action="store_true")
    args = parser.parse_args()

    if args.all_folds:
        for f in range(1, 6):
            prepare_fold(f)
    else:
        prepare_fold(args.fold)
