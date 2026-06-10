"""Analyze per-class IoU and confusion for segmentation runs."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from typing import Any, Dict, List

import numpy as np

from raw2task.data.kitti360_seg import TRAIN_ID_CLASS_NAMES


def _latest_confusion(run_dir: str) -> str:
    files = glob.glob(os.path.join(run_dir, "confusion_epoch*.npy"))
    if not files:
        raise FileNotFoundError(f"No confusion_epoch*.npy under {run_dir}")

    def epoch(path: str) -> int:
        m = re.search(r"confusion_epoch(\d+)\.npy", os.path.basename(path))
        return int(m.group(1)) if m else -1

    return max(files, key=epoch)


def _rows_from_confusion(conf: np.ndarray) -> List[Dict[str, Any]]:
    tp = np.diag(conf).astype(np.float64)
    gt = conf.sum(axis=1).astype(np.float64)
    pred = conf.sum(axis=0).astype(np.float64)
    fp = pred - tp
    fn = gt - tp
    denom = np.maximum(tp + fp + fn, 1.0)
    iou = tp / denom
    recall = tp / np.maximum(gt, 1.0)
    precision = tp / np.maximum(pred, 1.0)
    total_gt = max(float(gt.sum()), 1.0)

    rows: List[Dict[str, Any]] = []
    for cls_id, name in enumerate(TRAIN_ID_CLASS_NAMES):
        row_conf = conf[cls_id].astype(np.float64).copy()
        row_conf[cls_id] = 0.0
        top_id = int(row_conf.argmax()) if row_conf.sum() > 0 else -1
        rows.append(
            {
                "class_id": cls_id,
                "class_name": name,
                "IoU": float(iou[cls_id]),
                "recall": float(recall[cls_id]),
                "precision": float(precision[cls_id]),
                "gt_pixels": int(gt[cls_id]),
                "pred_pixels": int(pred[cls_id]),
                "gt_percent": float(100.0 * gt[cls_id] / total_gt),
                "pred_percent": float(100.0 * pred[cls_id] / total_gt),
                "top_confused_with": TRAIN_ID_CLASS_NAMES[top_id] if top_id >= 0 else "",
                "top_confusion_percent_of_gt": float(100.0 * row_conf[top_id] / max(gt[cls_id], 1.0)) if top_id >= 0 else 0.0,
            }
        )
    return rows


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _write_md(path: str, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "class_id",
        "class_name",
        "IoU",
        "recall",
        "precision",
        "gt_percent",
        "pred_percent",
        "top_confused_with",
        "top_confusion_percent_of_gt",
    ]
    with open(path, "w") as f:
        f.write("# Per-Class IoU Audit\n\n")
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fields)) + " |\n")
        for row in sorted(rows, key=lambda r: r["IoU"]):
            vals = []
            for key in fields:
                val = row[key]
                vals.append(f"{val:.4f}" if isinstance(val, float) else str(val))
            f.write("| " + " | ".join(vals) + " |\n")


def _plot(path: str, rows: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows_sorted = sorted(rows, key=lambda r: r["IoU"])
    labels = [r["class_name"] for r in rows_sorted]
    vals = [r["IoU"] for r in rows_sorted]
    colors = ["#c73e3a" if v < 0.15 else "#d98c25" if v < 0.35 else "#2f7d4f" for v in vals]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.barh(range(len(labels)), vals, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("IoU")
    ax.set_xlim(0, 1)
    ax.set_title("Per-Class IoU")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir or os.path.join(run_dir, "class_iou_audit")))
    os.makedirs(out_dir, exist_ok=True)
    conf_path = _latest_confusion(run_dir)
    rows = _rows_from_confusion(np.load(conf_path))
    _write_csv(os.path.join(out_dir, "class_iou.csv"), rows)
    _write_md(os.path.join(out_dir, "class_iou.md"), rows)
    _plot(os.path.join(out_dir, "class_iou.png"), rows)
    print(f"Read {conf_path}")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
