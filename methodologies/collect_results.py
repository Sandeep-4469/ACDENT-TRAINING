"""
Collect per-fold metrics.json from each method, compute mean ± std across folds,
print comparison table, save CSV and bar-chart.
"""
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_BASE = os.path.join(os.path.dirname(__file__), "results")
NUM_FOLDS    = 5

METHODS = ["heatmap", "simcc", "gnn", "yolo_pose"]
METHOD_LABELS = {
    "heatmap":   "Heatmap (baseline)",
    "simcc":     "SimCC",
    "gnn":       "GNN",
    "yolo_pose": "YOLO-Pose",
}

METRICS_ORDER = [
    ("incisor_sum", "Sum of Mand. Inc."),
    ("left_arc",    "L Arc"),
    ("right_arc",   "R Arc"),
    ("scale",       "Scale"),
]


def load_fold_metrics(method, fold_num):
    path = os.path.join(RESULTS_BASE, method, f"fold_{fold_num}", "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def aggregate(method):
    """Return {metric: (mean, std)} across all available folds."""
    fold_data = []
    for f in range(1, NUM_FOLDS+1):
        m = load_fold_metrics(method, f)
        if m:
            fold_data.append(m)
    if not fold_data:
        return None
    result = {}
    for key, _ in METRICS_ORDER:
        vals = [d[key] for d in fold_data if key in d]
        if vals:
            result[key] = (float(np.mean(vals)), float(np.std(vals)))
    return result


def print_table(all_results):
    col_w = 24
    print("\n" + "="*85)
    print("COMPARISON TABLE — Mean ± Std Line-Length Error (mm), 5-Fold CV")
    print("="*85)
    header = f"{'Method':<{col_w}}" + "".join(f"{label:>18}" for _, label in METRICS_ORDER)
    print(header)
    print("-"*85)
    for method, res in all_results.items():
        label = METHOD_LABELS.get(method, method)
        row   = f"{label:<{col_w}}"
        if res is None:
            row += "".join(f"{'N/A':>18}" for _ in METRICS_ORDER)
        else:
            for key, _ in METRICS_ORDER:
                if key in res:
                    mean, std = res[key]
                    row += f"{mean:>8.2f}±{std:<6.2f}  "
                else:
                    row += f"{'N/A':>18}"
        print(row)
    print("="*85)


def save_csv(all_results, path):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Method"]
        for key, label in METRICS_ORDER:
            header += [f"{label}_mean_mm", f"{label}_std_mm"]
        writer.writerow(header)
        for method, res in all_results.items():
            label = METHOD_LABELS.get(method, method)
            row = [label]
            if res is None:
                row += ["N/A"] * (len(METRICS_ORDER)*2)
            else:
                for key, _ in METRICS_ORDER:
                    if key in res:
                        row += [f"{res[key][0]:.4f}", f"{res[key][1]:.4f}"]
                    else:
                        row += ["N/A", "N/A"]
            writer.writerow(row)
    print(f"\nCSV saved: {path}")


def save_bar_chart(all_results, path):
    methods  = [METHOD_LABELS.get(m, m) for m in all_results]
    x        = np.arange(len(methods))
    width    = 0.2
    colors   = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]

    fig, ax = plt.subplots(figsize=(13, 6))
    for mi, (key, label) in enumerate(METRICS_ORDER):
        means, stds = [], []
        for method in all_results:
            res = all_results[method]
            if res and key in res:
                means.append(res[key][0])
                stds.append(res[key][1])
            else:
                means.append(0); stds.append(0)
        ax.bar(x + mi*width, means, width, yerr=stds, label=label,
               color=colors[mi], capsize=3)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Error (mm)")
    ax.set_title("DSAA Dental — Methodology Comparison (5-Fold CV)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Bar chart saved: {path}")


if __name__ == "__main__":
    os.makedirs(RESULTS_BASE, exist_ok=True)
    all_results = {}
    for method in METHODS:
        res = aggregate(method)
        all_results[method] = res
        if res is None:
            print(f"[MISSING] {method} — no fold results found")

    print_table(all_results)
    save_csv(all_results, os.path.join(RESULTS_BASE, "comparison_table.csv"))
    save_bar_chart(all_results, os.path.join(RESULTS_BASE, "comparison_chart.png"))
