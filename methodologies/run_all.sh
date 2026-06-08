#!/bin/bash
# Master script — runs all baselines for one or all folds.
# Usage:
#   bash run_all.sh                   # fold 1 only
#   bash run_all.sh --fold 3          # fold 3 only
#   bash run_all.sh --all-folds       # all 5 folds
#   bash run_all.sh --skip-yolo       # skip YOLO-Pose
#
set -e
cd "$(dirname "$0")"

FOLD=1
ALL_FOLDS=0
SKIP_YOLO=0

for arg in "$@"; do
    [[ "$arg" == "--all-folds" ]] && ALL_FOLDS=1
    [[ "$arg" == "--skip-yolo" ]] && SKIP_YOLO=1
done
# Parse --fold N
for i in "$@"; do
    if [[ "$prev" == "--fold" ]]; then FOLD=$i; fi
    prev=$i
done

FOLD_ARG="--fold $FOLD"
[[ "$ALL_FOLDS" -eq 1 ]] && FOLD_ARG="--all-folds"

mkdir -p logs results/{heatmap,simcc,gnn,yolo_pose}

echo "============================================================"
echo "  DSAA Dental — Baseline Comparison"
echo "  Fold arg: $FOLD_ARG"
echo "============================================================"

# ── 1. Heatmap ───────────────────────────────────────────────────────────────
echo ""
echo "[1/4] Training Heatmap baseline..."
python train_heatmap.py $FOLD_ARG 2>&1 | tee logs/heatmap_run.log
echo "  Done."

# ── 2. SimCC ─────────────────────────────────────────────────────────────────
echo ""
echo "[2/4] Training SimCC..."
python train_simcc.py $FOLD_ARG 2>&1 | tee logs/simcc_run.log
echo "  Done."

# ── 3. GNN ───────────────────────────────────────────────────────────────────
echo ""
echo "[3/4] Training GNN..."
python train_gnn.py $FOLD_ARG 2>&1 | tee logs/gnn_run.log
echo "  Done."

# ── 4. YOLO-Pose ─────────────────────────────────────────────────────────────
if [ "$SKIP_YOLO" -eq 0 ]; then
    echo ""
    echo "[4/4] Preparing YOLO dataset and training YOLO-Pose..."
    python prepare_yolo_dataset.py $FOLD_ARG 2>&1 | tee logs/yolo_prepare.log
    python train_yolo_pose.py $FOLD_ARG 2>&1 | tee logs/yolo_run.log
    echo "  Done."
else
    echo ""
    echo "[4/4] Skipping YOLO-Pose (--skip-yolo)."
fi

# ── 5. Collect results ────────────────────────────────────────────────────────
echo ""
echo "Collecting results..."
python collect_results.py

echo ""
echo "All done! See results/comparison_table.csv"
