"""Plot mIoU training curves for all experiments — paper Figure 2.

Usage:
    python -m raw2task.paper_training_curves \\
        --kitti-runs ./runs/kitti360_sfb4 \\
        --acdc-runs  ./runs/acdc_sfb4 \\
        --out-dir    ./paper_figures
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from typing import Dict, List, Optional, Tuple

EXP_ORDER = ["codesign", "rgb", "ablate_no_optics", "ablate_fixed_camera"]
EXP_LABELS = {
    "codesign":               "Co-design (ours)",
    "rgb":                    "RGB baseline",
    "ablate_no_optics":       "No optics",
    "ablate_fixed_camera":    "Fixed camera",
}
EXP_COLORS = {
    "codesign":               "#e63946",   # red  — our method
    "rgb":                    "#2196F3",   # blue
    "ablate_no_optics":       "#4CAF50",   # green
    "ablate_fixed_camera":    "#FF9800",   # orange
}
EXP_LINESTYLE = {
    "codesign":               "-",
    "rgb":                    "--",
    "ablate_no_optics":       "-.",
    "ablate_fixed_camera":    ":",
}


def load_curve(runs_dir: str, exp_name: str, suffix: str) -> Optional[Tuple[List[int], List[float]]]:
    patterns = [
        f"{runs_dir}/{exp_name}{suffix}_seed0_s0/metrics_log.csv",
        f"{runs_dir}/{exp_name}{suffix}_seed0/metrics_log.csv",
        f"{runs_dir}/{exp_name}{suffix}/metrics_log.csv",
    ]
    for pat in patterns:
        files = sorted(glob.glob(pat))
        if files:
            epochs, mious = [], []
            with open(files[0]) as f:
                for row in csv.DictReader(f):
                    if row.get("mIoU"):
                        try:
                            epochs.append(int(row["epoch"]))
                            mious.append(float(row["mIoU"]))
                        except ValueError:
                            pass
            if epochs:
                return epochs, mious
    return None


def plot_curves(
    datasets: List[Tuple[str, str, str, str]],   # (label, runs_dir, suffix, total_ep_str)
    out_dir: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping curve plots")
        return

    os.makedirs(out_dir, exist_ok=True)

    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(7 * n_ds, 5), squeeze=False)

    for col, (ds_label, runs_dir, suffix, total_ep) in enumerate(datasets):
        ax = axes[0][col]
        found_any = False
        for exp_key in EXP_ORDER:
            result = load_curve(runs_dir, exp_key, suffix)
            if result is None:
                continue
            found_any = True
            epochs, mious = result
            ax.plot(
                epochs, mious,
                label=EXP_LABELS[exp_key],
                color=EXP_COLORS[exp_key],
                linestyle=EXP_LINESTYLE[exp_key],
                linewidth=2.0,
                marker="o", markersize=3,
            )
            # annotate final value
            ax.annotate(
                f"{mious[-1]:.3f}",
                xy=(epochs[-1], mious[-1]),
                xytext=(4, 0), textcoords="offset points",
                fontsize=8, color=EXP_COLORS[exp_key],
            )

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Val mIoU", fontsize=12)
        ax.set_title(f"{ds_label} — SegFormer-B4", fontsize=13)
        if found_any:
            ax.legend(fontsize=10, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves → {out_path}")

    # also save per-dataset for inline figures
    for ds_label, runs_dir, suffix, total_ep in datasets:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        for exp_key in EXP_ORDER:
            result = load_curve(runs_dir, exp_key, suffix)
            if result is None:
                continue
            epochs, mious = result
            ax2.plot(epochs, mious,
                     label=EXP_LABELS[exp_key],
                     color=EXP_COLORS[exp_key],
                     linestyle=EXP_LINESTYLE[exp_key],
                     linewidth=2.0, marker="o", markersize=3)
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val mIoU")
        ax2.set_title(f"{ds_label}")
        ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)
        plt.tight_layout()
        name = ds_label.lower().replace(" ", "_").replace("-", "")
        p = os.path.join(out_dir, f"training_curves_{name}.png")
        fig2.savefig(p, dpi=180, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti-runs", default=None)
    ap.add_argument("--acdc-runs",  default=None)
    ap.add_argument("--out-dir",    default="./paper_figures")
    args = ap.parse_args()

    datasets = []
    if args.kitti_runs:
        datasets.append(("KITTI-360", args.kitti_runs, "_sfb4", "40"))
    if args.acdc_runs:
        datasets.append(("ACDC", args.acdc_runs, "_acdc", "60"))

    if not datasets:
        print("Provide --kitti-runs and/or --acdc-runs")
        return

    plot_curves(datasets, args.out_dir)


if __name__ == "__main__":
    main()
