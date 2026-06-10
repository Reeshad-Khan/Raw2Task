"""Export paper-facing measurements for a learned optics-sensor checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch

from raw2task.train_extended import CoDesignSensor, _save_sensor_preview


def _torch_load(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _make_evidence_panel(out_dir: Path, stem: str, title: str, imgs_dir: Path | None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("Learned field PSFs", out_dir / f"{stem}_psf_kernels.png"),
        ("Field PSF width", out_dir / f"{stem}_psf_rms_by_field.png"),
        ("Learned CFA response", out_dir / f"{stem}_cfa_bars.png"),
        ("Noise and ADC response", out_dir / f"{stem}_noise_adc_curve.png"),
    ]
    panels = [(name, path) for name, path in panels if path.is_file()]

    fig = plt.figure(figsize=(14.0, 5.6))
    gs = fig.add_gridspec(2, max(4, len(panels)), height_ratios=[0.55, 1.45], hspace=0.28, wspace=0.18)
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    blocks = [
        ("RGB proxy", "#f2f2f2"),
        ("RAW-linear\nunprocess", "#d9ead3"),
        ("learned field PSF", "#cfe2f3"),
        ("exposure + CFA", "#fff2cc"),
        ("noise + n-bit ADC", "#f4cccc"),
        ("task backbone", "#d0e0e3"),
        ("segmentation\nmetrics", "#eeeeee"),
    ]
    for i, (text, color) in enumerate(blocks):
        x = 0.015 + i * 0.14
        rect = plt.Rectangle((x, 0.36), 0.112, 0.34, facecolor=color, edgecolor="#222222", linewidth=1.0)
        ax.add_patch(rect)
        ax.text(x + 0.056, 0.53, text, ha="center", va="center", fontsize=9)
        if i < len(blocks) - 1:
            ax.annotate("", xy=(x + 0.134, 0.53), xytext=(x + 0.114, 0.53), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(
        0.5,
        0.12,
        "Checkpoint-exported evidence for the fixed-at-inference learned camera: optics, sensor, and ADC are measurable.",
        ha="center",
        va="center",
        fontsize=10,
    )

    for idx, (panel_title, path) in enumerate(panels[:4]):
        panel_ax = fig.add_subplot(gs[1, idx])
        panel_ax.imshow(plt.imread(path), aspect="auto")
        panel_ax.set_title(panel_title, fontsize=10)
        panel_ax.set_xticks([])
        panel_ax.set_yticks([])
        for spine in panel_ax.spines.values():
            spine.set_color("#444444")
            spine.set_linewidth(0.8)

    fig.suptitle(title, fontsize=13)
    fig.subplots_adjust(top=0.88, bottom=0.06, left=0.025, right=0.985, hspace=0.34, wspace=0.16)
    evidence_path = out_dir / f"{stem}_learned_camera_evidence.png"
    fig.savefig(evidence_path, dpi=240)
    plt.close(fig)

    if imgs_dir is not None:
        imgs_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(evidence_path, imgs_dir / "raw2task_learned_camera_evidence.png")
    return evidence_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Path to last.pt or best.pt.")
    parser.add_argument("--out", default="", help="Output directory. Defaults to <checkpoint>/design_preview.")
    parser.add_argument("--stem", default="best", help="Artifact prefix, e.g. best or checkpoint.")
    parser.add_argument("--imgs-dir", default="imgs", help="Optional repo image directory for the combined panel.")
    parser.add_argument("--write-design", action="store_true", help="Re-export camera_design_<stem>.json beside the checkpoint.")
    args = parser.parse_args()

    ckpt_path = os.path.abspath(os.path.expanduser(args.ckpt))
    payload = _torch_load(ckpt_path)
    cfg = payload.get("cfg")
    if not isinstance(cfg, dict):
        raise ValueError(f"Checkpoint has no config dictionary: {ckpt_path}")

    sensor_state = payload.get("sensor")
    if not isinstance(sensor_state, dict):
        raise ValueError(f"Checkpoint has no sensor state dictionary: {ckpt_path}")

    sensor = CoDesignSensor(cfg).cpu()
    sensor.load_state_dict(sensor_state, strict=False)
    sensor.eval()

    out_dir = Path(args.out or os.path.join(os.path.dirname(ckpt_path), "design_preview")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_sensor_preview(sensor, str(out_dir), args.stem)

    if args.write_design and hasattr(sensor, "export_design"):
        design_path = Path(os.path.dirname(ckpt_path)) / f"camera_design_{args.stem}.json"
        with open(design_path, "w") as f:
            json.dump(sensor.export_design(include_kernels=True), f, indent=2)
        if args.stem == "best":
            with open(Path(os.path.dirname(ckpt_path)) / "camera_design_best.json", "w") as f:
                json.dump(sensor.export_design(include_kernels=True), f, indent=2)

    imgs_dir = Path(args.imgs_dir).resolve() if args.imgs_dir else None
    evidence = _make_evidence_panel(out_dir, args.stem, f"Learned Optics-Sensor Measurements: {Path(ckpt_path).parent.name}", imgs_dir)
    print(f"Saved camera measurements to {out_dir}")
    print(f"Saved evidence panel to {evidence}")


if __name__ == "__main__":
    main()
