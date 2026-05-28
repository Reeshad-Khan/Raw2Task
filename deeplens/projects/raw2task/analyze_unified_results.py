"""Create paper-facing artifacts for unified 2D/3D/occupancy runs.

The unified model is the strongest claim path: one learned camera frontend and
one shared model are trained for 2D semantics, observed 3D semantics, and
occupancy. This collector keeps fast experiments lightweight while still
exporting the same evidence types expected from the full protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

from deeplens.projects.raw2task.animate_camera_learning import make_learning_gifs


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return {}


def _as_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


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


def _latest_or_best(rows: List[Dict[str, str]], key: str) -> Dict[str, str]:
    valid = [r for r in rows if _as_float(r.get(key)) is not None]
    if not valid:
        return rows[-1] if rows else {}
    return max(valid, key=lambda r: _as_float(r.get(key)) or -1.0)


def _run_kind(name: str) -> str:
    if "codesign" in name:
        return "latest_co_design"
    if "fixed_camera" in name:
        return "fixed_camera"
    if "rgb" in name:
        return "rgb_baseline"
    return "other"


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.{digits}f}"
    if value is None:
        return ""
    return str(value)


def _find_files(root: Path, patterns: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for pattern in patterns:
        out.extend(root.glob(pattern))
    return sorted(p for p in out if p.is_file())


def _row(run_dir: Path) -> Dict[str, Any]:
    rows = _read_csv(run_dir / "metrics_log.csv")
    best = _latest_or_best(rows, "unified_score")
    initial = _load_json(run_dir / "camera_design_initial.json")
    best_design = _load_json(run_dir / "camera_design_best.json")
    initial_optics = initial.get("optics", {}) if initial else {}
    best_optics = best_design.get("optics", {}) if best_design else {}
    initial_noise = initial.get("noise_quantization", {}) if initial else {}
    best_noise = best_design.get("noise_quantization", {}) if best_design else {}
    viz = _find_files(run_dir, ["viz/*.png", "plots/*.png"])
    return {
        "run": run_dir.name,
        "kind": _run_kind(run_dir.name),
        "run_dir": str(run_dir),
        "epochs_logged": len(rows),
        "best_epoch": best.get("epoch", ""),
        "unified_score": _as_float(best.get("unified_score")),
        "mIoU_2d": _as_float(best.get("mIoU_2d")),
        "mIoU_3d": _as_float(best.get("mIoU_3d")),
        "semantic_occupied_IoU": _as_float(best.get("semantic_occupied_IoU")),
        "occupancy_IoU": _as_float(best.get("occupancy_IoU")),
        "occupancy_degenerate": best.get("occupancy_degenerate", ""),
        "occupancy_positive_fraction": _as_float(best.get("occupancy_positive_fraction")),
        "psf_coeff_rms_delta": _rms_delta(initial_optics.get("coefficients"), best_optics.get("coefficients")),
        "cfa_l1_delta": _l1_delta(
            initial.get("cfa_weights_rgb") if initial else None,
            best_design.get("cfa_weights_rgb") if best_design else None,
        ),
        "exposure_initial": _as_float(initial.get("exposure_gain")) if initial else None,
        "exposure_best": _as_float(best_design.get("exposure_gain")) if best_design else None,
        "bit_depth_best": _as_float(best_noise.get("bit_depth_continuous")) if best_noise else None,
        "read_noise_best": _as_float(best_noise.get("read_noise_std")) if best_noise else None,
        "shot_noise_best": _as_float(best_noise.get("shot_noise_scale")) if best_noise else None,
        "has_2d_viz": any("_2d_" in p.name for p in viz),
        "has_3d_viz": any("_3d_" in p.name for p in viz),
        "has_plots": any(p.parent.name == "plots" for p in viz),
        "num_evidence_png": len(viz),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _write_md(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    by_kind = {row["kind"]: row for row in rows}
    latest = by_kind.get("latest_co_design")
    fixed = by_kind.get("fixed_camera")
    rgb = by_kind.get("rgb_baseline")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# Unified Multi-Task Evidence\n\n")
        f.write("One shared optics-sensor frontend and one shared model are evaluated for 2D segmentation, observed 3D semantic segmentation, and occupancy.\n\n")
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fields)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(_fmt(row.get(k), 6 if "delta" in k or "noise" in k else 4) for k in fields) + " |\n")
        f.write("\n## Feasibility Read\n\n")
        if latest and fixed:
            delta = (latest.get("unified_score") or 0.0) - (fixed.get("unified_score") or 0.0)
            f.write(f"- Latest co-design vs fixed camera unified-score delta: `{delta:.4f}`.\n")
        if latest and rgb:
            delta = (latest.get("unified_score") or 0.0) - (rgb.get("unified_score") or 0.0)
            f.write(f"- Latest co-design vs RGB unified-score delta: `{delta:.4f}`.\n")
        if latest and latest.get("occupancy_degenerate") in (True, "True", "true", "1"):
            f.write("- Occupancy metric is degenerate for this observed-voxel split; use 2D and observed 3D semantic metrics as the primary fast evidence.\n")
        if latest and latest.get("psf_coeff_rms_delta") is None:
            f.write("- Optics movement could not be measured from exported coefficients; inspect camera preview artifacts before making a learned-lens claim.\n")
        elif latest and (latest.get("psf_coeff_rms_delta") or 0.0) < 1e-4:
            f.write("- Learned optics barely moved in this run; do not claim a strong learned-lens contribution from this fast evidence alone.\n")
        if latest and not (latest.get("has_2d_viz") and latest.get("has_3d_viz") and latest.get("has_plots")):
            f.write("- Evidence is incomplete: missing 2D/3D visualization panels or plots.\n")
        f.write("\nFast experiments are feasibility evidence; the paper table should use the full protocol with repeated seeds.\n")


def _plot(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [r["kind"].replace("_", "\n") for r in rows]

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = range(len(rows))
    for offset, key, label, color in [
        (-0.22, "mIoU_2d", "2D mIoU", "#1f77b4"),
        (0.00, "mIoU_3d", "3D mIoU", "#2ca02c"),
        (0.22, "unified_score", "unified score", "#9467bd"),
    ]:
        vals = [r.get(key) or 0.0 for r in rows]
        ax.bar([i + offset for i in x], vals, width=0.2, label=label, color=color)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("validation score")
    ax.set_ylim(0.0, max(0.7, max([(r.get("unified_score") or 0.0) for r in rows] + [0.0]) + 0.05))
    ax.set_title("Unified Fast Evidence: Shared Camera and Model")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "unified_fast_scores.png", dpi=240)
    plt.close(fig)

    xs = [r.get("psf_coeff_rms_delta") or 0.0 for r in rows]
    ys = [r.get("unified_score") or 0.0 for r in rows]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.scatter(xs, ys, s=64, color="#1f77b4")
    for row, xval, yval in zip(rows, xs, ys):
        ax.annotate(row["kind"].replace("_", " "), (xval, yval), xytext=(5, 3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("PSF coefficient RMS change")
    ax.set_ylabel("unified score")
    ax.set_title("Camera Movement vs Unified Accuracy")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "unified_camera_delta_vs_score.png", dpi=240)
    plt.close(fig)


def _copy_key_evidence(runs_root: Path, out_dir: Path) -> None:
    evidence_dir = out_dir / "evidence_gallery"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    copied: List[Dict[str, str]] = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        preview_dir = run_dir / "design_preview"
        if preview_dir.is_dir():
            try:
                gif_dir = run_dir / "learning_gifs"
                for gif in make_learning_gifs(preview_dir, gif_dir, imgs_dir=None):
                    dst = evidence_dir / f"{run_dir.name}_{gif.name}"
                    shutil.copy2(gif, dst)
                    copied.append({"run": run_dir.name, "artifact": str(dst), "source": str(gif)})
            except Exception as exc:
                copied.append({"run": run_dir.name, "artifact": "", "source": f"GIF generation skipped: {exc}"})
        for pattern in [
            "viz/epoch*_2d_sensor_task_story.png",
            "viz/epoch*_3d_semantic_bev_story.png",
            "plots/unified_validation_curves.png",
            "plots/unified_training_curves.png",
            "design_preview/best_psf_kernels.png",
            "design_preview/best_psf_rms_by_field.png",
            "design_preview/best_cfa_bars.png",
            "design_preview/best_noise_adc_curve.png",
        ]:
            matches = sorted(run_dir.glob(pattern))
            if not matches:
                continue
            src = matches[-1]
            dst = evidence_dir / f"{run_dir.name}_{src.name}"
            shutil.copy2(src, dst)
            copied.append({"run": run_dir.name, "artifact": str(dst), "source": str(src)})
    if copied:
        _write_csv(evidence_dir / "manifest.csv", copied, ["run", "artifact", "source"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="runs/unified_multitask")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    runs_root = Path(args.runs_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else runs_root / "paper_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(p for p in runs_root.iterdir() if p.is_dir() and (p / "metrics_log.csv").is_file()) if runs_root.is_dir() else []
    rows = [_row(d) for d in run_dirs]
    order = {"latest_co_design": 0, "fixed_camera": 1, "rgb_baseline": 2, "other": 3}
    rows.sort(key=lambda r: (order.get(r["kind"], 9), r["run"]))
    fields = [
        "run",
        "kind",
        "epochs_logged",
        "best_epoch",
        "unified_score",
        "mIoU_2d",
        "mIoU_3d",
        "semantic_occupied_IoU",
        "occupancy_IoU",
        "occupancy_degenerate",
        "occupancy_positive_fraction",
        "psf_coeff_rms_delta",
        "cfa_l1_delta",
        "exposure_initial",
        "exposure_best",
        "bit_depth_best",
        "read_noise_best",
        "shot_noise_best",
        "has_2d_viz",
        "has_3d_viz",
        "has_plots",
        "num_evidence_png",
        "run_dir",
    ]
    _write_csv(out_dir / "unified_summary.csv", rows, fields)
    _write_md(out_dir / "unified_claim_readiness.md", rows, fields)
    _plot(rows, out_dir)
    _copy_key_evidence(runs_root, out_dir)
    print(f"[unified-analysis] wrote {out_dir}")


if __name__ == "__main__":
    main()
