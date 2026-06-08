#!/usr/bin/env python3
"""
Generate all ablation study plots from the training log and summary CSV.

Usage:
  python plot_ablation.py
  python plot_ablation.py --run-dir results/ablation/ablation_20260523_223901
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

DSAA_ROOT   = Path(__file__).resolve().parent
DEFAULT_RUN = DSAA_ROOT / "results" / "ablation" / "ablation_20260523_223901"

# Short display labels for configs
SHORT_LABELS = {
    "01_heatmap_only":                       "01\nheatmap",
    "02_+coord":                             "02\n+coord",
    "03_+coord_+length":                     "03\n+length",
    "04_+coord_+length_+arcside":            "04\n+arc_side",
    "05_+coord_+length_+arcside_+arcalign":  "05\n+arc_align",
    "06_full":                               "06\n+arc_contrast\n(full)",
}

# One colour per config
CONFIG_COLORS = [
    "#e41a1c", "#ff7f00", "#f0c000",
    "#4daf4a", "#377eb8", "#984ea3",
]

# Colours for individual loss curves
LOSS_COLORS = {
    "train_heatmap":     "#e41a1c",
    "train_coord":       "#ff7f00",
    "train_length":      "#f0c000",
    "train_arc_side":    "#4daf4a",
    "train_arc_align":   "#377eb8",
    "train_arc_contrast": "#984ea3",
}

LOSS_LABELS = {
    "train_heatmap":     "Heatmap",
    "train_coord":       "Coord",
    "train_length":      "Length",
    "train_arc_side":    "Arc Side",
    "train_arc_align":   "Arc Align",
    "train_arc_contrast": "Arc Contrast",
}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data(run_dir: Path):
    log_csv     = run_dir / "ablation_training_log.csv"
    summary_csv = run_dir / "ablation_summary.csv"
    if not log_csv.exists():
        raise FileNotFoundError(f"Training log not found: {log_csv}")
    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary not found: {summary_csv}")
    log = pd.read_csv(log_csv)
    summary = pd.read_csv(summary_csv)
    # Ensure numeric
    for col in log.columns:
        if col not in ["config", "active_losses"]:
            log[col] = pd.to_numeric(log[col], errors="coerce")
    return log, summary


# ---------------------------------------------------------------------------
# Plot 1 — Validation loss convergence
# ---------------------------------------------------------------------------
def plot_val_loss(log: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    configs = list(log["config"].unique())
    for color, cfg in zip(CONFIG_COLORS, configs):
        sub = log[log["config"] == cfg].sort_values("epoch")
        ax.plot(sub["epoch"], sub["val_loss"], color=color, linewidth=1.8,
                label=cfg.replace("_", " "))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss (Heatmap MSE)")
    ax.set_title("Validation Loss Convergence — All Configs")
    ax.legend(loc="upper right", framealpha=0.8)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.5f"))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "01_val_loss_convergence.png", dpi=180)
    plt.close(fig)
    print("  Saved: 01_val_loss_convergence.png")


# ---------------------------------------------------------------------------
# Plot 2 — Train total loss convergence
# ---------------------------------------------------------------------------
def plot_train_total(log: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    configs = list(log["config"].unique())
    for color, cfg in zip(CONFIG_COLORS, configs):
        sub = log[log["config"] == cfg].sort_values("epoch")
        ax.plot(sub["epoch"], sub["train_total"], color=color, linewidth=1.8,
                label=cfg.replace("_", " "))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Train Loss")
    ax.set_title("Total Training Loss Convergence — All Configs")
    ax.legend(loc="upper right", framealpha=0.8)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "02_train_total_loss.png", dpi=180)
    plt.close(fig)
    print("  Saved: 02_train_total_loss.png")


# ---------------------------------------------------------------------------
# Plot 3 — Each individual loss component across all configs
#           (always computed, even when inactive — shows what it was tracking)
# ---------------------------------------------------------------------------
def plot_individual_losses(log: pd.DataFrame, out_dir: Path) -> None:
    loss_cols = ["train_heatmap", "train_coord", "train_length",
                 "train_arc_side", "train_arc_align", "train_arc_contrast"]
    configs   = list(log["config"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes      = axes.flatten()

    for ax, loss_col in zip(axes, loss_cols):
        for color, cfg in zip(CONFIG_COLORS, configs):
            sub = log[log["config"] == cfg].sort_values("epoch")
            # dashed if this loss was NOT active in this config
            active = sub["active_losses"].iloc[0]
            loss_key = loss_col.replace("train_", "")
            is_active = loss_key in active
            ax.plot(sub["epoch"], sub[loss_col],
                    color=color, linewidth=1.5 if is_active else 0.8,
                    linestyle="-" if is_active else "--",
                    alpha=1.0 if is_active else 0.4,
                    label=cfg.replace("_", " ") if is_active else None)

        ax.set_title(f"{LOSS_LABELS[loss_col]} Loss\n(dashed = not in backprop)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss value")
        ax.grid(True, alpha=0.2)

    # Shared legend from last axis
    handles = [plt.Line2D([0],[0], color=c, linewidth=2, label=cfg.replace("_"," "))
               for c, cfg in zip(CONFIG_COLORS, configs)]
    fig.legend(handles=handles, loc="lower center", ncol=3, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Individual Loss Components Across All Configs\n"
                 "(solid = active in backprop, dashed = computed but not optimised)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_dir / "03_individual_losses_all_configs.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 03_individual_losses_all_configs.png")


# ---------------------------------------------------------------------------
# Plot 4 — Test metrics bar chart (mm errors per config)
# ---------------------------------------------------------------------------
def plot_test_metrics_bars(summary: pd.DataFrame, out_dir: Path) -> None:
    metrics = ["scale_mm", "incisor_sum_mm", "left_arc_mm", "right_arc_mm"]
    labels  = ["Scale", "Incisor Sum", "Left Arc", "Right Arc"]
    configs = summary["config"].tolist()
    x       = np.arange(len(configs))
    width   = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        vals = summary[metric].values
        bars = ax.bar(x + i * width, vals, width, label=label, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([SHORT_LABELS.get(c, c) for c in configs], fontsize=9)
    ax.set_ylabel("Error (mm)")
    ax.set_title("Test Metrics per Config — Individual mm Errors")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "04_test_metrics_bars.png", dpi=180)
    plt.close(fig)
    print("  Saved: 04_test_metrics_bars.png")


# ---------------------------------------------------------------------------
# Plot 5 — Sum3 mm improvement across additive configs
# ---------------------------------------------------------------------------
def plot_sum3_progression(summary: pd.DataFrame, out_dir: Path) -> None:
    configs  = summary["config"].tolist()
    sum3     = summary["sum3_mm"].values
    baseline = sum3[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left — absolute sum3_mm
    bars = axes[0].bar(range(len(configs)), sum3, color=CONFIG_COLORS, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, sum3):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[0].set_xticks(range(len(configs)))
    axes[0].set_xticklabels([SHORT_LABELS.get(c, c) for c in configs], fontsize=9)
    axes[0].set_ylabel("Sum3 Error (mm)")
    axes[0].set_title("Sum (Incisor + Left Arc + Right Arc) — mm")
    axes[0].axhline(baseline, color="red", linestyle="--", linewidth=1.2, alpha=0.6, label=f"Baseline {baseline:.3f}")
    axes[0].legend(); axes[0].grid(True, axis="y", alpha=0.25)

    # Right — improvement relative to baseline (%)
    improvement = [(baseline - v) / baseline * 100 for v in sum3]
    bar_colors  = ["#aaaaaa" if v <= 0 else "#2ca02c" for v in improvement]
    bars2 = axes[1].bar(range(len(configs)), improvement, color=bar_colors, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars2, improvement):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + (0.3 if v >= 0 else -1.2),
                     f"{v:+.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[1].set_xticks(range(len(configs)))
    axes[1].set_xticklabels([SHORT_LABELS.get(c, c) for c in configs], fontsize=9)
    axes[1].set_ylabel("Improvement over Baseline (%)")
    axes[1].set_title("Relative Improvement in Sum3 vs Heatmap-Only Baseline")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle("Additive Loss Ablation — Sum3 mm Error Progression", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "05_sum3_progression.png", dpi=180)
    plt.close(fig)
    print("  Saved: 05_sum3_progression.png")


# ---------------------------------------------------------------------------
# Plot 6 — Heatmap of test metrics (configs × metrics)
# ---------------------------------------------------------------------------
def plot_metrics_heatmap(summary: pd.DataFrame, out_dir: Path) -> None:
    metrics = ["scale_mm", "incisor_sum_mm", "left_arc_mm", "right_arc_mm", "sum3_mm"]
    mlabels = ["Scale", "Incisor Sum", "Left Arc", "Right Arc", "Sum3"]
    configs = summary["config"].tolist()
    data    = summary[metrics].values.astype(float)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(mlabels))); ax.set_xticklabels(mlabels, fontsize=10)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels([SHORT_LABELS.get(c, c).replace("\n", " ") for c in configs], fontsize=9)
    # Annotate cells
    for i in range(len(configs)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if data[i,j] > np.percentile(data[:,j], 60) else "black")
    plt.colorbar(im, ax=ax, label="Error (mm)", shrink=0.8)
    ax.set_title("Test Metrics Heatmap — All Configs vs All Metrics (mm)\n"
                 "(green = lower error = better)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "06_metrics_heatmap.png", dpi=180)
    plt.close(fig)
    print("  Saved: 06_metrics_heatmap.png")


# ---------------------------------------------------------------------------
# Plot 7 — Final epoch individual loss values per config (bar chart)
#           Shows what each loss was at convergence — even if inactive
# ---------------------------------------------------------------------------
def plot_final_loss_values(log: pd.DataFrame, out_dir: Path) -> None:
    loss_cols = ["train_heatmap", "train_coord", "train_length",
                 "train_arc_side", "train_arc_align", "train_arc_contrast"]
    configs   = list(log["config"].unique())

    # Take last 5 epochs average (more stable than single last epoch)
    final_vals = []
    for cfg in configs:
        sub = log[log["config"] == cfg].sort_values("epoch").tail(5)
        final_vals.append(sub[loss_cols].mean().values)
    final_vals = np.array(final_vals)

    x     = np.arange(len(configs))
    width = 0.13
    fig, ax = plt.subplots(figsize=(15, 6))

    for i, (col, label, color) in enumerate(zip(loss_cols,
                                                  [LOSS_LABELS[c] for c in loss_cols],
                                                  LOSS_COLORS.values())):
        # Check activity per config
        activity = []
        for cfg in configs:
            active = log[log["config"]==cfg]["active_losses"].iloc[0]
            activity.append(col.replace("train_","") in active)

        vals = final_vals[:, i]
        bars = ax.bar(x + i*width, vals, width, label=label, color=color,
                      alpha=0.85, edgecolor="white")
        # Hatch inactive bars
        for bar, is_active in zip(bars, activity):
            if not is_active:
                bar.set_hatch("///")
                bar.set_alpha(0.4)

    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels([SHORT_LABELS.get(c, c) for c in configs], fontsize=9)
    ax.set_ylabel("Loss Value (avg last 5 epochs)")
    ax.set_title("Individual Loss Values at Convergence — All Configs\n"
                 "(hatched = computed but NOT in backprop)")
    ax.legend(loc="upper right", ncol=3)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_dir / "07_final_loss_values_per_config.png", dpi=180)
    plt.close(fig)
    print("  Saved: 07_final_loss_values_per_config.png")


# ---------------------------------------------------------------------------
# Plot 8 — Val loss + each individual train loss side-by-side per config
#           (2-row grid: top=val_loss, bottom=train components)
# ---------------------------------------------------------------------------
def plot_per_config_breakdown(log: pd.DataFrame, out_dir: Path) -> None:
    configs   = list(log["config"].unique())
    loss_cols = ["train_heatmap", "train_coord", "train_length",
                 "train_arc_side", "train_arc_align", "train_arc_contrast"]

    fig, axes = plt.subplots(2, len(configs), figsize=(20, 8),
                              gridspec_kw={"height_ratios": [1, 1.5]})

    for col_idx, cfg in enumerate(configs):
        sub    = log[log["config"] == cfg].sort_values("epoch")
        active = sub["active_losses"].iloc[0]
        color  = CONFIG_COLORS[col_idx]

        # Top row — val loss
        ax_top = axes[0, col_idx]
        ax_top.plot(sub["epoch"], sub["val_loss"], color=color, linewidth=1.8)
        ax_top.set_title(SHORT_LABELS.get(cfg, cfg), fontsize=9)
        ax_top.set_xlabel("Epoch", fontsize=8)
        if col_idx == 0:
            ax_top.set_ylabel("Val Loss", fontsize=9)
        ax_top.tick_params(labelsize=7)
        ax_top.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.5f"))
        ax_top.grid(True, alpha=0.2)

        # Bottom row — all individual losses
        ax_bot = axes[1, col_idx]
        for loss_col, lcolor in LOSS_COLORS.items():
            is_active = loss_col.replace("train_", "") in active
            ax_bot.plot(sub["epoch"], sub[loss_col],
                        color=lcolor, linewidth=1.5 if is_active else 0.7,
                        linestyle="-" if is_active else "--",
                        alpha=1.0 if is_active else 0.35,
                        label=LOSS_LABELS[loss_col])
        ax_bot.set_xlabel("Epoch", fontsize=8)
        if col_idx == 0:
            ax_bot.set_ylabel("Individual Loss", fontsize=9)
        ax_bot.tick_params(labelsize=7)
        ax_bot.grid(True, alpha=0.2)
        ax_bot.set_yscale("log")

    # Shared legend for individual losses
    handles = [plt.Line2D([0],[0], color=c, linewidth=2, label=LOSS_LABELS[k])
               for k, c in LOSS_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=6, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Per-Config Breakdown: Val Loss (top) & Individual Losses (bottom)\n"
                 "solid = in backprop, dashed = monitored only", fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_dir / "08_per_config_breakdown.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 08_per_config_breakdown.png")


# ---------------------------------------------------------------------------
# Parse args & run
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ablation study results.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    return parser.parse_args()


def main() -> None:
    args    = parse_args()
    run_dir = args.run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    out_dir = run_dir / "plots"
    out_dir.mkdir(exist_ok=True)
    print(f"Run dir : {run_dir}")
    print(f"Plots   : {out_dir}\n")

    log, summary = load_data(run_dir)

    print("Generating plots...")
    plot_val_loss(log, out_dir)
    plot_train_total(log, out_dir)
    plot_individual_losses(log, out_dir)
    plot_test_metrics_bars(summary, out_dir)
    plot_sum3_progression(summary, out_dir)
    plot_metrics_heatmap(summary, out_dir)
    plot_final_loss_values(log, out_dir)
    plot_per_config_breakdown(log, out_dir)

    print(f"\nAll 8 plots saved to: {out_dir}")
    print("\nPlot summary:")
    print("  01_val_loss_convergence.png       — val loss curves all configs")
    print("  02_train_total_loss.png           — total train loss (log scale)")
    print("  03_individual_losses_all_configs  — each loss component, solid=active dashed=inactive")
    print("  04_test_metrics_bars.png          — grouped bars: scale/incisor/arc errors in mm")
    print("  05_sum3_progression.png           — sum3 absolute + % improvement over baseline")
    print("  06_metrics_heatmap.png            — heatmap configs × metrics")
    print("  07_final_loss_values_per_config   — convergence values of each loss (hatched=inactive)")
    print("  08_per_config_breakdown.png       — per-config val + all individual losses")


if __name__ == "__main__":
    main()
