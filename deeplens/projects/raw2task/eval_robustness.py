"""Robustness sweeps for KITTI-360 raw-to-task checkpoints.

This script produces the quantitative blur/noise/bit-depth tables reviewers
asked for. It keeps metric/split consistent with training: dataset-level mIoU
from the same validation split, plus pixel accuracy, exported as CSV and JSON.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from deeplens.projects.raw2task.train_extended import (
    CoDesignSensor,
    PassThroughSensor,
    align_logits_to_labels,
    build_seg_model,
    get_dataset,
    iou_from_confusion,
    update_confusion,
)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _gaussian_kernel(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    k = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    k = k / k.sum().clamp_min(1e-8)
    return k


def _blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    size = max(3, int(round(sigma * 6)) | 1)
    k = _gaussian_kernel(size, sigma, x.device, x.dtype).view(1, 1, size, size)
    k = k.repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, k, padding=size // 2, groups=x.shape[1]).clamp(0, 1)


def _warp_labels_like(labels: torch.Tensor, theta: torch.Tensor, ignore_index: int) -> torch.Tensor:
    labels_f = labels.unsqueeze(1).float()
    grid = F.affine_grid(theta, labels_f.shape, align_corners=False)
    warped = F.grid_sample(labels_f, grid, mode="nearest", padding_mode="zeros", align_corners=False).squeeze(1).long()
    valid = torch.ones_like(labels_f)
    valid = F.grid_sample(valid, grid, mode="nearest", padding_mode="zeros", align_corners=False).squeeze(1)
    return torch.where(valid > 0.5, warped, torch.full_like(warped, int(ignore_index)))


def _apply_viewpoint_corruption(
    x: torch.Tensor,
    labels: torch.Tensor,
    spec: Dict[str, Any],
    ignore_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    kind = spec["kind"]
    level = float(spec.get("level", 0.0))
    b = x.shape[0]
    theta = torch.zeros(b, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    if kind == "viewpoint_pitch":
        theta[:, 1, 1] = max(0.70, min(1.30, 1.0 + level))
    elif kind == "viewpoint_height":
        theta[:, 1, 2] = level
    elif kind == "viewpoint_shift_x":
        theta[:, 0, 2] = level
    elif kind == "viewpoint_yaw":
        theta[:, 0, 1] = level
    else:
        raise ValueError(f"Unsupported viewpoint corruption: {kind}")
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x_warped = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False).clamp(0, 1)
    labels_warped = _warp_labels_like(labels, theta, ignore_index=ignore_index)
    return x_warped, labels_warped


def _apply_input_corruption(
    x: torch.Tensor,
    labels: torch.Tensor,
    spec: Dict[str, Any],
    ignore_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    kind = spec["kind"]
    level = float(spec.get("level", 0.0))
    if kind == "clean":
        return x, labels
    if kind == "blur":
        return _blur(x, sigma=level), labels
    if kind == "gaussian":
        return (x + torch.randn_like(x) * level).clamp(0, 1), labels
    if kind == "exposure":
        return (x * level).clamp(0, 1), labels
    if kind in ("bit_depth", "read_noise", "shot_noise"):
        return x, labels
    if kind in ("viewpoint_pitch", "viewpoint_height", "viewpoint_shift_x", "viewpoint_yaw"):
        return _apply_viewpoint_corruption(x, labels, spec, ignore_index=ignore_index)
    raise ValueError(f"Unsupported input corruption: {kind}")


def _set_sensor_stress(sensor: torch.nn.Module, spec: Dict[str, Any]) -> None:
    kind = spec["kind"]
    if kind == "bit_depth" and hasattr(sensor, "noise"):
        sensor.noise.bit_depth = int(spec["level"])
    elif kind == "read_noise" and hasattr(sensor, "noise"):
        sensor.noise.read_noise_std = float(spec["level"])
    elif kind == "shot_noise" and hasattr(sensor, "noise"):
        sensor.noise.shot_noise_scale = float(spec["level"])


def _sensor_scalar_state(sensor: torch.nn.Module) -> Dict[str, Any]:
    if not hasattr(sensor, "noise"):
        return {}
    return {
        "bit_depth": int(sensor.noise.bit_depth),
        "read_noise_std": float(sensor.noise.read_noise_std),
        "shot_noise_scale": float(sensor.noise.shot_noise_scale),
    }


def _restore_sensor_scalar_state(sensor: torch.nn.Module, state: Dict[str, Any]) -> None:
    if not state or not hasattr(sensor, "noise"):
        return
    sensor.noise.bit_depth = int(state["bit_depth"])
    sensor.noise.read_noise_std = float(state["read_noise_std"])
    sensor.noise.shot_noise_scale = float(state["shot_noise_scale"])


def _default_sweep() -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = [{"name": "clean", "kind": "clean", "level": 0.0}]
    specs += [{"name": f"blur_sigma_{v}", "kind": "blur", "level": v} for v in [0.5, 1.0, 1.5, 2.0]]
    specs += [{"name": f"gaussian_{v}", "kind": "gaussian", "level": v} for v in [0.01, 0.03, 0.05, 0.08]]
    specs += [{"name": f"exposure_{v}", "kind": "exposure", "level": v} for v in [0.5, 0.75, 1.25, 1.5]]
    specs += [{"name": f"viewpoint_pitch_{v}", "kind": "viewpoint_pitch", "level": v} for v in [-0.12, 0.12]]
    specs += [{"name": f"viewpoint_height_{v}", "kind": "viewpoint_height", "level": v} for v in [-0.08, 0.08]]
    specs += [{"name": f"viewpoint_shift_x_{v}", "kind": "viewpoint_shift_x", "level": v} for v in [-0.08, 0.08]]
    specs += [{"name": f"viewpoint_yaw_{v}", "kind": "viewpoint_yaw", "level": v} for v in [-0.06, 0.06]]
    specs += [{"name": f"bit_depth_{v}", "kind": "bit_depth", "level": v} for v in [4, 6, 8, 10]]
    specs += [{"name": f"read_noise_{v}", "kind": "read_noise", "level": v} for v in [0.002, 0.005, 0.01]]
    specs += [{"name": f"shot_noise_{v}", "kind": "shot_noise", "level": v} for v in [0.01, 0.02, 0.04]]
    return specs


def _load_sweep(path: str | None) -> List[Dict[str, Any]]:
    if not path:
        return _default_sweep()
    with open(os.path.expanduser(path), "r") as f:
        data = yaml.safe_load(f)
    return data["sweep"] if isinstance(data, dict) and "sweep" in data else data


def _build_from_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = copy.deepcopy(ckpt["cfg"])
    cfg["device"] = device
    cfg["num_gpus"] = int(torch.cuda.device_count()) if device.type == "cuda" else 0
    _, val_loader, num_classes = get_dataset(cfg)

    input_mode = str(cfg.get("pipeline", {}).get("input", "sensor")).lower()
    if input_mode in ("rgb", "rgb_baseline", "processed_rgb"):
        sensor = PassThroughSensor(cfg).to(device)
    else:
        sensor = CoDesignSensor(cfg).to(device)
    in_ch = sensor.output_channels
    model = build_seg_model(cfg, in_channels=in_ch, num_classes=num_classes).to(device)
    model.ignore_index = int(cfg.get("model", {}).get("ignore_index", 255))
    sensor.load_state_dict(ckpt["sensor"], strict=True)
    model.load_state_dict(ckpt["model"], strict=True)
    return cfg, sensor, model, val_loader, num_classes


@torch.no_grad()
def evaluate_one(
    sensor: torch.nn.Module,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    spec: Dict[str, Any],
    max_batches: int = 0,
) -> Dict[str, Any]:
    sensor.eval()
    model.eval()
    ignore_index = int(getattr(model, "ignore_index", 255))
    conf = torch.zeros(num_classes, num_classes, dtype=torch.int64).numpy()
    total_correct = 0
    total_labeled = 0

    _set_sensor_stress(sensor, spec)
    old_force_noise = bool(getattr(sensor, "force_stochastic_noise", False))
    if hasattr(sensor, "force_stochastic_noise"):
        sensor.force_stochastic_noise = spec["kind"] in ("read_noise", "shot_noise", "gaussian")
    try:
        for batch_idx, (imgs, labels) in enumerate(loader, 1):
            if max_batches > 0 and batch_idx > max_batches:
                break
            imgs = imgs.to(device)
            labels = labels.to(device)
            imgs, labels = _apply_input_corruption(imgs, labels, spec, ignore_index=ignore_index)
            logits = align_logits_to_labels(model(sensor(imgs)), labels)
            pred = logits.argmax(1)
            valid = labels != ignore_index
            total_correct += (pred[valid] == labels[valid]).sum().item()
            total_labeled += valid.sum().item()
            update_confusion(conf, pred, labels, num_classes=num_classes, ignore_index=ignore_index)
    finally:
        if hasattr(sensor, "force_stochastic_noise"):
            sensor.force_stochastic_noise = old_force_noise

    miou, per_class_iou = iou_from_confusion(conf)
    return {
        "name": spec["name"],
        "kind": spec["kind"],
        "level": spec.get("level", ""),
        "pixel_acc": float(total_correct / max(total_labeled, 1)),
        "mIoU": float(miou),
        "per_class_IoU": [float(x) for x in per_class_iou],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sweep", default="")
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = _device()
    cfg, sensor, model, val_loader, num_classes = _build_from_checkpoint(args.ckpt, device)
    specs = _load_sweep(args.sweep)

    rows = []
    base_sensor_state = copy.deepcopy(sensor.state_dict())
    base_scalar_state = _sensor_scalar_state(sensor)
    for spec in specs:
        sensor.load_state_dict(base_sensor_state, strict=True)
        _restore_sensor_scalar_state(sensor, base_scalar_state)
        rows.append(evaluate_one(sensor, model, val_loader, device, num_classes, spec, args.max_batches))
        print(f"{rows[-1]['name']}: mIoU={rows[-1]['mIoU']:.4f} pixel_acc={rows[-1]['pixel_acc']:.4f}")

    json_path = os.path.join(args.out_dir, "robustness.json")
    with open(json_path, "w") as f:
        json.dump({"checkpoint": args.ckpt, "config_name": cfg.get("name", ""), "results": rows}, f, indent=2)

    csv_path = os.path.join(args.out_dir, "robustness.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "kind", "level", "pixel_acc", "mIoU"] + [f"IoU_{i}" for i in range(num_classes)])
        for r in rows:
            writer.writerow([r["name"], r["kind"], r["level"], r["pixel_acc"], r["mIoU"], *r["per_class_IoU"]])

    print(f"Saved robustness results to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
