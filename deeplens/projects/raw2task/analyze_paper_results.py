"""Audit current paper experiment results, including in-progress runs.

This script is intentionally independent of ``summary.csv`` because the full
orchestrator only writes summary rows after a job exits. During long training
runs, this audit lets us see whether the evidence is strong enough for the
paper claim and whether the plots are usable.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from typing import Any, Dict, List


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _flatten(value: Any) -> List[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        out: List[float] = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return []


def _rms_delta(a: Any, b: Any) -> float | None:
    av, bv = _flatten(a), _flatten(b)
    if not av or len(av) != len(bv):
        return None
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(av, bv)) / len(av))


def _l1_delta(a: Any, b: Any) -> float | None:
    av, bv = _flatten(a), _flatten(b)
    if not av or len(av) != len(bv):
        return None
    return sum(abs(x - y) for x, y in zip(av, bv)) / len(av)


def _latest_log_tail(path: str, n: int = 20) -> str:
    if not os.path.isfile(path):
        return ""
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-n:])


def _run_row(run_dir: str) -> Dict[str, Any]:
    name_seed = os.path.basename(run_dir)
    metrics = _read_csv(os.path.join(run_dir, "metrics_log.csv"))
    epoch_summary = _read_csv(os.path.join(run_dir, "epoch_summary.csv"))
    best_row: Dict[str, str] = {}
    last_row: Dict[str, str] = {}
    if metrics:
        best_row = max(metrics, key=lambda r: _as_float(r.get("mIoU")) or -1.0)
        last_row = metrics[-1]

    initial = _load_json(os.path.join(run_dir, "camera_design_initial.json"))
    best_design = _load_json(os.path.join(run_dir, "camera_design_best.json"))
    init_opt = initial.get("optics", {}) if initial else {}
    best_opt = best_design.get("optics", {}) if best_design else {}
    init_noise = initial.get("noise_quantization", {}) if initial else {}
    best_noise = best_design.get("noise_quantization", {}) if best_design else {}

    status = "empty"
    if os.path.isfile(os.path.join(run_dir, "last.pt")):
        status = "has_last"
    if metrics:
        status = "has_metrics"
    if os.path.isfile(os.path.join(run_dir, "train.log")) and not os.path.isfile(os.path.join(run_dir, "last.pt")):
        status = "running_or_interrupted"

    return {
        "run": name_seed,
        "run_dir": run_dir,
        "status": status,
        "epochs_logged": len(metrics),
        "best_epoch": best_row.get("epoch", ""),
        "best_mIoU": _as_float(best_row.get("mIoU")),
        "last_epoch": last_row.get("epoch", ""),
        "last_mIoU": _as_float(last_row.get("mIoU")),
        "last_pixel_acc": _as_float(last_row.get("pixel_acc")),
        "last_train_loss": _as_float(epoch_summary[-1].get("train_loss")) if epoch_summary else None,
        "last_sensor_reg": _as_float(epoch_summary[-1].get("train_sensor_reg")) if epoch_summary else None,
        "has_best_design": bool(best_design),
        "psf_coeff_rms_delta": _rms_delta(init_opt.get("coefficients"), best_opt.get("coefficients")),
        "cfa_l1_delta": _l1_delta(initial.get("cfa_weights_rgb") if initial else None, best_design.get("cfa_weights_rgb") if best_design else None),
        "exposure_initial": _as_float(initial.get("exposure_gain")) if initial else None,
        "exposure_best": _as_float(best_design.get("exposure_gain")) if best_design else None,
        "bit_depth_best": _as_float(best_noise.get("bit_depth_continuous")) if best_noise else None,
        "read_noise_best": _as_float(best_noise.get("read_noise_std")) if best_noise else None,
        "shot_noise_best": _as_float(best_noise.get("shot_noise_scale")) if best_noise else None,
        "train_log_tail": _latest_log_tail(os.path.join(run_dir, "train.log"), n=10),
    }


def _write_csv(path: str, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.{digits}f}"
    return "" if value is None else str(value)


def _write_markdown(path: str, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "run",
        "status",
        "epochs_logged",
        "best_epoch",
        "best_mIoU",
        "last_mIoU",
        "last_pixel_acc",
        "psf_coeff_rms_delta",
        "cfa_l1_delta",
        "exposure_best",
        "bit_depth_best",
        "read_noise_best",
        "shot_noise_best",
    ]
    with open(path, "w") as f:
        f.write("# Interim Paper Result Audit\n\n")
        f.write("This report includes in-progress runs and is not a final paper table.\n\n")
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fields)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(_fmt(row.get(k), 6 if "delta" in k or "noise" in k else 4) for k in fields) + " |\n")
        f.write("\n## Claim Readiness\n\n")
        completed = [r for r in rows if r.get("epochs_logged", 0)]
        has_codesign = any("codesign" in r["run"] and r.get("best_mIoU") is not None for r in completed)
        has_fixed = any("fixed_camera_frontend" in r["run"] and r.get("best_mIoU") is not None for r in completed)
        has_rgb = any("rgb_liteseg" in r["run"] and r.get("best_mIoU") is not None for r in completed)
        if not (has_codesign and has_fixed and has_rgb):
            f.write("- Not ready for the main claim yet: need full co-design, fixed-camera, and RGB baseline rows on the same split.\n")
        if completed:
            best = max((r.get("best_mIoU") or 0.0) for r in completed)
            if best < 0.55:
                f.write(f"- Current best mIoU is {_fmt(best)}. This is below a strong KITTI-360 paper result; treat as incomplete/undertrained unless later epochs improve.\n")
        cfa_moves = [r.get("cfa_l1_delta") for r in completed if r.get("cfa_l1_delta") is not None]
        if cfa_moves and max(cfa_moves) < 0.01:
            f.write("- CFA barely changes from Bayer. Do not claim strong learned-CFA contribution unless the fixed-CFA ablation proves otherwise.\n")


def _plot(rows: List[Dict[str, Any]], out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    plot_rows = [r for r in rows if r.get("best_mIoU") is not None]
    if not plot_rows:
        return
    labels = [r["run"].replace("_seed", "\nseed") for r in plot_rows]
    vals = [r.get("best_mIoU") or 0.0 for r in plot_rows]
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(labels)), 4.0))
    colors = ["#2f6f9f" if "codesign" in r["run"] else "#888888" for r in plot_rows]
    ax.bar(range(len(labels)), vals, color=colors)
    ax.set_ylabel("Best validation mIoU")
    ax.set_ylim(0, max(0.7, max(vals) + 0.05))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Interim Results: Best mIoU So Far")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "interim_best_miou.png"), dpi=220)
    plt.close(fig)

    xs = [r.get("psf_coeff_rms_delta") or 0.0 for r in plot_rows]
    ys = vals
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.scatter(xs, ys, s=60, color="#2f6f9f")
    for r, x, y in zip(plot_rows, xs, ys):
        ax.annotate(r["run"].replace("kitti360_", "").replace("_liteseg", ""), (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("PSF coefficient RMS change")
    ax.set_ylabel("Best validation mIoU")
    ax.set_title("Camera Movement vs Accuracy")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "interim_camera_delta_vs_miou.png"), dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="runs/industry_paper_matrix")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    runs_root = os.path.abspath(os.path.expanduser(args.runs_root))
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir or os.path.join(runs_root, "paper_tables", "interim_audit")))
    os.makedirs(out_dir, exist_ok=True)
    run_dirs = sorted(d for d in glob.glob(os.path.join(runs_root, "*_seed*")) if os.path.isdir(d))
    rows = [_run_row(d) for d in run_dirs]

    fields = [
        "run",
        "status",
        "epochs_logged",
        "best_epoch",
        "best_mIoU",
        "last_epoch",
        "last_mIoU",
        "last_pixel_acc",
        "last_train_loss",
        "last_sensor_reg",
        "has_best_design",
        "psf_coeff_rms_delta",
        "cfa_l1_delta",
        "exposure_initial",
        "exposure_best",
        "bit_depth_best",
        "read_noise_best",
        "shot_noise_best",
        "run_dir",
    ]
    _write_csv(os.path.join(out_dir, "interim_run_status.csv"), rows, fields)
    _write_markdown(os.path.join(out_dir, "interim_report.md"), rows)
    _plot(rows, out_dir)
    print(f"Wrote interim audit to {out_dir}")


if __name__ == "__main__":
    main()
