"""
Runs Heatmap → SimCC → GNN for a single fold sequentially.
YOLO-Pose must be run separately (needs prepare_yolo_dataset.py first).
Usage: python run_sequential.py --fold 1
"""
import sys, os, argparse

sys.path.insert(0, os.path.dirname(__file__))

import train_heatmap
import train_simcc
import train_gnn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=1, choices=range(1, 6))
    parser.add_argument("--all-folds", action="store_true")
    args = parser.parse_args()

    folds = range(1, 6) if args.all_folds else [args.fold]

    for fold in folds:
        print(f"\n{'='*60}")
        print(f"  FOLD {fold} — Step 1/3: Heatmap Baseline")
        print(f"{'='*60}")
        train_heatmap.train(fold_num=fold)

        print(f"\n{'='*60}")
        print(f"  FOLD {fold} — Step 2/3: SimCC")
        print(f"{'='*60}")
        train_simcc.train(fold_num=fold)

        print(f"\n{'='*60}")
        print(f"  FOLD {fold} — Step 3/3: GNN")
        print(f"{'='*60}")
        train_gnn.train(fold_num=fold)

    print(f"\n{'='*60}")
    print("  Collecting results")
    print(f"{'='*60}")
    import collect_results
    exec(open(os.path.join(os.path.dirname(__file__), "collect_results.py")).read())
