"""Summarize matched KITTI-360 / Cityscapes fast comparison runs."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _as_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _parse_experiment(name: str) -> Dict[str, str]:
    parts = name.split("_")
    dataset = parts[0] if parts else "unknown"
    condition = "clean"
    frontend = "unknown"
    backbone = "unknown"
    for item in ("lowlight", "lowbit4", "clean"):
        if item in parts:
            condition = item
            break
    if "codesign" in parts and "tasktokens" in parts:
        frontend = "codesign_task_tokens"
    elif "codesign" in parts:
        frontend = "codesign_soft_rgb"
    else:
        for item in ("rgb", "fixed"):
            if item in parts:
                frontend = item
                break
    for item in ("segformer", "ddrlite", "biselite", "b0", "b1", "b2"):
        if item in parts:
            idx = parts.index(item)
            if item == "segformer" and idx + 1 < len(parts):
                backbone = f"{item}_{parts[idx + 1]}"
            else:
                backbone = item
            break
    return {"dataset": dataset, "condition": condition, "frontend": frontend, "backbone": backbone}


def _write_csv(path: Path, rows: List[Dict[str, str]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _plot(rows: List[Dict[str, str]], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return

    condition_order = {"clean": 0, "lowlight": 1, "lowbit4": 2}
    frontend_order = {"codesign_soft_rgb": 0, "codesign_task_tokens": 1, "rgb": 2, "fixed": 3}
    datasets = ["kitti360", "cityscapes"]
    lookup = {
        (r["dataset"], r["condition"], r["frontend"], r["backbone"]): _as_float(r.get("mIoU", ""))
        for r in rows
    }
    combos = sorted(
        {(r["condition"], r["frontend"], r["backbone"]) for r in rows},
        key=lambda x: (condition_order.get(x[0], 99), frontend_order.get(x[1], 99), x[2]),
    )
    labels = []
    values = {ds: [] for ds in datasets}
    for cond, frontend, backbone in combos:
        label = frontend.replace("codesign_", "co-").replace("_", "\n")
        labels.append(f"{cond}\n{label}\n{backbone}")
        for ds in datasets:
            values[ds].append(lookup.get((ds, cond, frontend, backbone), float("nan")) * 100.0)

    x = np.arange(len(labels))
    width = 0.36
    fig_w = max(10.5, 0.72 * max(1, len(labels)))
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    ax.bar(x - width / 2, values["kitti360"], width, label="KITTI-360", color="#3b82f6")
    ax.bar(x + width / 2, values["cityscapes"], width, label="Cityscapes", color="#f97316")
    ax.set_ylabel("mIoU (%)")
    ax.set_title("Matched Backbone Cross-Dataset Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti-summary", default="runs/dataset_compare_fast/kitti360/summary.csv")
    parser.add_argument("--city-summary", default="runs/dataset_compare_fast/cityscapes/summary.csv")
    parser.add_argument("--out-dir", default="runs/dataset_compare_fast/paper_tables")
    args = parser.parse_args()

    rows: List[Dict[str, str]] = []
    for summary_path, dataset_name in [
        (Path(args.kitti_summary), "kitti360"),
        (Path(args.city_summary), "cityscapes"),
    ]:
        for row in _read_csv(summary_path):
            parsed = _parse_experiment(row.get("experiment", ""))
            parsed["dataset"] = dataset_name
            rows.append(
                {
                    **parsed,
                    "experiment": row.get("experiment", ""),
                    "seed": row.get("seed", ""),
                    "mIoU": row.get("mIoU") or row.get("best_val", ""),
                    "pixel_acc": row.get("pixel_acc", ""),
                    "params_model_m": row.get("params_model_m", ""),
                    "latency_chain_ms": row.get("latency_chain_ms", ""),
                    "ckpt_dir": row.get("ckpt_dir", ""),
                }
            )

    out_dir = Path(args.out_dir)
    fields = [
        "dataset",
        "condition",
        "frontend",
        "backbone",
        "experiment",
        "seed",
        "mIoU",
        "pixel_acc",
        "params_model_m",
        "latency_chain_ms",
        "ckpt_dir",
    ]
    rows = sorted(rows, key=lambda r: (r["condition"], r["frontend"], r["dataset"], r["experiment"]))
    _write_csv(out_dir / "cross_dataset_comparison.csv", rows, fields)
    _plot(rows, out_dir / "cross_dataset_comparison_miou.png")
    print(f"[dataset-compare] wrote {out_dir / 'cross_dataset_comparison.csv'}")
    print(f"[dataset-compare] wrote {out_dir / 'cross_dataset_comparison_miou.png'}")


if __name__ == "__main__":
    main()
