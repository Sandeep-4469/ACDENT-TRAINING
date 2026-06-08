"""
Run all methodology baselines (Heatmap, SimCC, GNN, YOLO-Pose) across all 5 folds.
Usage:
    python run_all_folds.py                        # all methods, all folds
    python run_all_folds.py --skip-yolo            # skip YOLO-Pose
    python run_all_folds.py --methods heatmap gnn  # specific methods only
    python run_all_folds.py --folds 1 2 3          # specific folds only
"""
import sys, os, argparse, traceback, time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import train_heatmap
import train_simcc
import train_gnn
import prepare_yolo_dataset
import train_yolo_pose
import collect_results as cr

ALL_METHODS = ["heatmap", "simcc", "gnn", "yolo_pose"]

LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)


def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def header(text):
    print(f"\n{'='*65}")
    print(f"  {text}")
    print(f"  {timestamp()}")
    print(f"{'='*65}")


def run_fold(fold, methods, summary):
    for method in methods:
        header(f"Fold {fold} — {method.upper()}")
        t0 = time.time()
        try:
            if method == "heatmap":
                train_heatmap.train(fold_num=fold)
            elif method == "simcc":
                train_simcc.train(fold_num=fold)
            elif method == "gnn":
                train_gnn.train(fold_num=fold)
            elif method == "yolo_pose":
                prepare_yolo_dataset.prepare_fold(fold)
                weights = train_yolo_pose.train_fold(fold)
                train_yolo_pose.evaluate_fold(fold, weights)

            elapsed = time.time() - t0
            summary.append((fold, method, "OK", f"{elapsed/60:.1f} min"))
            print(f"\n  Done in {elapsed/60:.1f} min")

        except Exception as e:
            elapsed = time.time() - t0
            summary.append((fold, method, "FAILED", str(e)))
            print(f"\n  FAILED: {e}")
            traceback.print_exc()


def print_summary(summary):
    print(f"\n{'='*65}")
    print("  RUN SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Fold':<6} {'Method':<12} {'Status':<8} {'Info'}")
    print(f"  {'-'*55}")
    for fold, method, status, info in summary:
        icon = "OK" if status == "OK" else "!!"
        print(f"  [{icon}] Fold {fold}  {method:<12} {status:<8} {info}")
    print(f"{'='*65}")

    failed = [(f, m) for f, m, s, _ in summary if s == "FAILED"]
    if failed:
        print(f"\n  {len(failed)} job(s) failed: {failed}")
    else:
        print(f"\n  All {len(summary)} jobs completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds",   type=int, nargs="+", default=list(range(1, 6)),
                        choices=range(1, 6), metavar="N",
                        help="Folds to run (default: 1 2 3 4 5)")
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS,
                        choices=ALL_METHODS, metavar="M",
                        help="Methods to run (default: all)")
    parser.add_argument("--skip-yolo", action="store_true",
                        help="Exclude YOLO-Pose")
    args = parser.parse_args()

    methods = args.methods
    if args.skip_yolo and "yolo_pose" in methods:
        methods = [m for m in methods if m != "yolo_pose"]

    print(f"\n{'='*65}")
    print(f"  DSAA Dental — Methodology Baselines")
    print(f"  Methods : {methods}")
    print(f"  Folds   : {args.folds}")
    print(f"  Started : {timestamp()}")
    print(f"{'='*65}")

    summary = []
    total_t0 = time.time()

    for fold in args.folds:
        run_fold(fold, methods, summary)

    # Collect and compare results
    header("Collecting Results")
    try:
        import importlib
        importlib.reload(cr)
        exec(open(os.path.join(os.path.dirname(__file__), "collect_results.py")).read())
    except Exception as e:
        print(f"  collect_results failed: {e}")

    print_summary(summary)
    total = (time.time() - total_t0) / 60
    print(f"\n  Total time: {total:.1f} min  |  Finished: {timestamp()}\n")
