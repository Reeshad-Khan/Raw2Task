# deeplens/projects/raw2task/train_extended.py

# Copyright (c) 2025 DeepLens Authors. All rights reserved.
#
# This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC).
#     The license is only for non-commercial use (commercial licenses can be obtained from authors).
#     The material is provided as-is, with no warranties whatsoever.
#     If you publish any code, data, or scientific work based on this, please cite our work.

"""
Task-driven camera co-design: DeepLens optics + exposure + learnable CFA + quantization → small CNN / tiny UNet.

This script is a practical, fast-to-run pipeline:
- Reuses DeepLens optics (GeoLens if a lens JSON is provided, else identity optics).
- Adds learnable exposure, 2×2 CFA, Poisson+Gaussian noise, and N-bit quantization.
- Supports:
    * Classification: CIFAR-10 / ImageNet(-local)   -> SmallCNN
    * Segmentation:  KITTI-360 (exported Image/Mask) -> UNetTiny

Usage:
  python -m raw2task.train_extended --config deeplens/projects/raw2task/configs/kitti360_seg_identity.yaml
"""

import argparse
import contextlib
import copy
import logging
import math
import os
import random
import sys
import time
from typing import Tuple, Dict, Any, List, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import yaml
from PIL import Image, ImageDraw, ImageFont

from deeplens.utils import set_seed

# Our modules
from raw2task.sensors.cfa import LearnableCFA, soft_cfa_demosaic, superpixel_demux
from raw2task.sensors.noise import PoissonGaussianNoiseQuant
from raw2task.sensors.trainable_optics import ConstrainedFieldPSF
from raw2task.models.cnn_small import SmallCNN
from raw2task.data.kitti360_seg import (
    KITTI360Seg,
    TRAIN_ID_CLASS_NAMES,
    _maybe_remap_to_trainid,
    _remap_original_id_to_trainid,
    _sanitize_trainid,
)
from raw2task.data.seg_jsonl import SegmentationJSONLDataset
from raw2task.data.occupancy_proxy import (
    OCCUPANCY_PROXY_CLASS_NAMES,
    OCCUPANCY_PROXY_PALETTE,
    OccupancyProxyDataset,
    occupancy_proxy_metadata,
    trainid_to_occupancy_proxy_np,
)
from raw2task.models.segmentation import build_segmentation_model

try:
    from deeplens.sensor.rgb_sensor import RGBSensor
except Exception:
    RGBSensor = None

# Try efficiency utils (optional)
try:
    from raw2task.utils.efficiency import (
        Chain, measure_model_efficiency, save_efficiency
    )
except Exception:
    Chain = None
    measure_model_efficiency = None
    save_efficiency = None

# ---------- KITTI-360 / Cityscapes-like 19-class palette ----------
PALETTE_19 = [
    (128,  64, 128),  # road
    (244,  35, 232),  # sidewalk
    ( 70,  70,  70),  # building
    (102, 102, 156),  # wall
    (190, 153, 153),  # fence
    (153, 153, 153),  # pole
    (250, 170,  30),  # traffic light
    (220, 220,   0),  # traffic sign
    (107, 142,  35),  # vegetation
    (152, 251, 152),  # terrain
    ( 70, 130, 180),  # sky
    (220,  20,  60),  # person
    (255,   0,   0),  # rider
    (  0,   0, 142),  # car
    (  0,   0,  70),  # truck
    (  0,  60, 100),  # bus
    (  0,  80, 100),  # train
    (  0,   0, 230),  # motorcycle
    (119,  11,  32),  # bicycle
]


def is_dense_prediction_task(task: str) -> bool:
    return str(task).lower() in ("seg", "segmentation", "occupancy", "occupancy_proxy")


def dense_task_palette(cfg: Dict[str, Any], num_classes: int) -> List[Tuple[int, int, int]]:
    task = str(cfg.get("task", "seg")).lower()
    if task in ("occupancy", "occupancy_proxy") or int(cfg.get("model", {}).get("num_classes", num_classes)) == 3:
        return OCCUPANCY_PROXY_PALETTE
    return PALETTE_19[:num_classes]


def dense_task_class_names(cfg: Dict[str, Any], num_classes: int) -> List[str]:
    task = str(cfg.get("task", "seg")).lower()
    if task in ("occupancy", "occupancy_proxy") or int(cfg.get("model", {}).get("num_classes", num_classes)) == 3:
        return OCCUPANCY_PROXY_CLASS_NAMES
    return TRAIN_ID_CLASS_NAMES[:num_classes]

# -------------------------
# Small helpers: checkpoint & AMP
# -------------------------
def _make_ckpt(sensor, model, optim, scaler, scheduler, cfg, epoch, step, best, ema_model=None, feedback_controller=None):
    payload = {
        "sensor": sensor.state_dict(),
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "scaler": getattr(scaler, "state_dict", lambda: {})(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "cfg": cfg,
        "epoch": epoch,
        "step": step,
        "best": best,
    }
    if ema_model is not None:
        payload["ema_model"] = ema_model.state_dict()
    if feedback_controller is not None:
        payload["feedback_controller"] = feedback_controller.state_dict()
    return payload

def _torch_load_checkpoint(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_ckpt(path, device):
    return _torch_load_checkpoint(path, map_location=device)

def _find_default_resume_path(ckpt_dir):
    lp = os.path.join(ckpt_dir, "last.pt")
    return lp if os.path.isfile(lp) else ""


def _state_dict_average(state_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not state_dicts:
        return {}
    avg: Dict[str, Any] = {}
    for key in state_dicts[0].keys():
        values = [sd[key] for sd in state_dicts if key in sd]
        if not values:
            continue
        first = values[0]
        if torch.is_tensor(first):
            if torch.is_floating_point(first):
                stacked = torch.stack([v.detach().to(dtype=torch.float32) for v in values], dim=0)
                avg[key] = stacked.mean(dim=0).to(dtype=first.dtype)
            else:
                avg[key] = first.detach().clone()
        else:
            avg[key] = copy.deepcopy(first)
    return avg


def _checkpoint_epoch(path: str) -> int:
    import re
    base = os.path.basename(path)
    for pattern in (r"last_ep(\d+)\.pt", r"best_ep(\d+)_", r"epoch(\d+)"):
        match = re.search(pattern, base)
        if match:
            return int(match.group(1))
    return -1


def _checkpoint_state_for_average(ckpt: Dict[str, Any], prefer_ema: bool) -> Dict[str, Any]:
    if prefer_ema and ckpt.get("ema_model") is not None:
        return ckpt["ema_model"]
    return ckpt["model"]


def _build_checkpoint_average(ckpt_dir: str, source: str, k: int, device: torch.device, prefer_ema: bool = True) -> Dict[str, Any] | None:
    import glob
    if k < 2:
        return None
    if source == "best":
        paths = sorted(glob.glob(os.path.join(ckpt_dir, "best_ep*.pt")), key=_checkpoint_epoch)
    else:
        paths = sorted(glob.glob(os.path.join(ckpt_dir, "last_ep*.pt")), key=_checkpoint_epoch)
    if len(paths) < k:
        return None
    selected = paths[-k:]
    ckpts = [_torch_load_checkpoint(p, map_location=device) for p in selected]
    sensor_states = [ckpt["sensor"] for ckpt in ckpts if "sensor" in ckpt]
    model_states = [_checkpoint_state_for_average(ckpt, prefer_ema=prefer_ema) for ckpt in ckpts if "model" in ckpt]
    if len(sensor_states) < k or len(model_states) < k:
        return None
    return {
        "sensor": _state_dict_average(sensor_states),
        "model": _state_dict_average(model_states),
        "source_paths": selected,
    }


def _maybe_eval_checkpoint_average(
    sensor: nn.Module,
    model: nn.Module,
    cfg: Dict[str, Any],
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    ckpt_dir: str,
    best: float,
    task: str,
):
    import json

    tcfg = cfg.get("train", {})
    avg_k = int(tcfg.get("checkpoint_average_k", 0))
    if not is_dense_prediction_task(task) or avg_k < 2:
        return best

    avg_source = str(tcfg.get("checkpoint_average_source", "best")).lower()
    avg_use_ema = bool(tcfg.get("checkpoint_average_use_ema", True))
    payload = _build_checkpoint_average(ckpt_dir, avg_source, avg_k, device, prefer_ema=avg_use_ema)
    if payload is None:
        logging.info(f"[AvgCkpt] skipped: need at least {avg_k} checkpoints from source={avg_source!r}")
        return best

    avg_sensor = copy.deepcopy(sensor).to(device)
    avg_model = copy.deepcopy(model).to(device)
    try:
        avg_sensor.load_state_dict(payload["sensor"], strict=False)
        avg_model.load_state_dict(payload["model"], strict=False)
    except Exception as exc:
        logging.warning(f"[AvgCkpt] failed to load averaged weights: {exc}")
        return best

    avg_epoch_tag = f"avg_{avg_source}k{avg_k}"
    metrics = evaluate_seg(avg_sensor, avg_model, val_loader, device, num_classes=num_classes, out_dir=ckpt_dir, epoch=avg_epoch_tag)
    avg_score = float(metrics["mIoU"])
    logging.info(f"[AvgCkpt] source={avg_source} k={avg_k} | mIoU {avg_score:.4f}")

    metrics_path = os.path.join(ckpt_dir, f"metrics_{avg_epoch_tag}.json")
    with open(metrics_path, "w") as f:
        json.dump({**metrics, "average_of": payload["source_paths"]}, f, indent=2)

    if avg_score > best + 1e-6:
        metric_tag = "miou"
        save_path = os.path.join(ckpt_dir, f"best_avg_{avg_source}k{avg_k}_{metric_tag}{avg_score:.4f}.pt")
        torch.save({"sensor": avg_sensor.state_dict(), "model": avg_model.state_dict(), "cfg": cfg, "average_of": payload["source_paths"]}, save_path)
        logging.info(f"[AvgCkpt] Saved improved averaged checkpoint to {save_path}")
        return avg_score

    return best


def _build_ema_model(model: nn.Module) -> nn.Module:
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    return ema_model


@torch.no_grad()
def _ema_update(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    msd = model.state_dict()
    for key, ema_v in ema_model.state_dict().items():
        model_v = msd[key]
        if torch.is_floating_point(ema_v):
            ema_v.mul_(decay).add_(model_v.detach(), alpha=1.0 - decay)
        else:
            ema_v.copy_(model_v)


class TaskAwareISPAdapter(nn.Module):
    """Compact task-oriented ISP bridge for camera-sim/RAW-like tensors.

    The adapter applies local filtering plus global and regional modulation,
    then emits a bounded residual correction. This gives pretrained sRGB
    perception backbones enough capacity to absorb camera-domain statistics
    without turning the frontend into a large learned image enhancer.
    """

    def __init__(
        self,
        channels: int = 3,
        hidden_channels: int = 24,
        residual_scale: float = 0.20,
        regional_grid: int = 8,
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("TaskAwareISPAdapter requires a positive channel count")
        hidden = max(int(hidden_channels), channels)
        groups = max(1, math.gcd(hidden, 8))
        self.channels = int(channels)
        self.residual_scale = float(residual_scale)
        self.regional_grid = max(int(regional_grid), 1)
        self.clamp_output = bool(clamp_output)
        self.stem = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=hidden),
            nn.SiLU(inplace=True),
        )
        self.pixel_mod = nn.Conv2d(hidden, channels * 2, kernel_size=1)
        self.regional_mod = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels * 2, kernel_size=1),
        )
        self.global_mod = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels * 2, kernel_size=1),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        self._init_near_identity()

    def _init_near_identity(self) -> None:
        for module in (self.pixel_mod, self.regional_mod[-1], self.global_mod[-1], self.residual[-1]):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        pixel_scale, pixel_bias = self.pixel_mod(feat).chunk(2, dim=1)
        pooled = F.adaptive_avg_pool2d(x, (self.regional_grid, self.regional_grid))
        regional = F.interpolate(self.regional_mod(pooled), size=x.shape[-2:], mode="bilinear", align_corners=False)
        regional_scale, regional_bias = regional.chunk(2, dim=1)
        global_scale, global_bias = self.global_mod(x).chunk(2, dim=1)
        scale = 1.0 + self.residual_scale * torch.tanh(pixel_scale + regional_scale + global_scale)
        bias = self.residual_scale * torch.tanh(pixel_bias + regional_bias + global_bias)
        residual = self.residual_scale * torch.tanh(self.residual(feat))
        out = x * scale + bias + residual
        return out.clamp(0.0, 1.0) if self.clamp_output else out


class PerceptionWithTaskISP(nn.Module):
    """Wrap a task-aware ISP adapter in front of a perception backbone."""

    def __init__(self, adapter: nn.Module, backbone: nn.Module) -> None:
        super().__init__()
        self.adapter = adapter
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.adapter(x))

    @property
    def ignore_index(self):
        return getattr(self.backbone, "ignore_index", None)

    @ignore_index.setter
    def ignore_index(self, value) -> None:
        setattr(self.backbone, "ignore_index", value)

    def adapter_parameters(self) -> Iterable[nn.Parameter]:
        return self.adapter.parameters()

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        return self.backbone.parameters()


def build_task_isp_adapter(cfg: Dict[str, Any], channels: int) -> nn.Module | None:
    adapter_cfg = cfg.get("task_isp", {}) or {}
    if not bool(adapter_cfg.get("enabled", False)):
        return None
    kind = str(adapter_cfg.get("type", "task_aware_modulation")).lower()
    if kind not in ("task_aware_modulation", "ta_isp", "compact"):
        raise ValueError(f"Unsupported task_isp.type={kind!r}")
    return TaskAwareISPAdapter(
        channels=channels,
        hidden_channels=int(adapter_cfg.get("hidden_channels", 24)),
        residual_scale=float(adapter_cfg.get("residual_scale", 0.20)),
        regional_grid=int(adapter_cfg.get("regional_grid", 8)),
        clamp_output=bool(adapter_cfg.get("clamp_output", True)),
    )


def _set_requires_grad(
    params: Iterable[nn.Parameter],
    enabled: bool,
    base_requires: Dict[int, bool] | None = None,
) -> None:
    for p in params:
        p.requires_grad_(bool(enabled) and (True if base_requires is None else bool(base_requires.get(id(p), False))))


@contextlib.contextmanager
def _temporary_model_trainability(model: nn.Module, *, allow_adapter: bool = False):
    params = list(model.parameters())
    old = [p.requires_grad for p in params]
    adapter_ids = set()
    if allow_adapter and hasattr(model, "adapter_parameters"):
        adapter_ids = {id(p) for p in model.adapter_parameters()}
    try:
        for p, req in zip(params, old):
            p.requires_grad_(bool(req) and id(p) in adapter_ids)
        yield
    finally:
        for p, req in zip(params, old):
            p.requires_grad_(req)

def _export_sensor_design(sensor: nn.Module, out_json: str, include_kernels: bool = True) -> None:
    if not hasattr(sensor, "export_design"):
        return
    import json

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    payload = sensor.export_design(include_kernels=include_kernels)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

def _save_sensor_preview(sensor: nn.Module, out_dir: str, stem: str) -> None:
    """Save paper-facing numeric and visual camera measurements."""
    try:
        import csv
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logging.debug(f"[Monitor] matplotlib unavailable; skipping sensor preview: {e}")
        return

    os.makedirs(out_dir, exist_ok=True)
    try:
        cfa = sensor.cfa.effective_weights().detach().cpu().numpy() if hasattr(sensor, "cfa") else None
        exposure = float(sensor.exposure_value().detach().cpu()) if hasattr(sensor, "exposure_value") else None
        kernels = None
        if getattr(sensor, "optics", None) is not None and hasattr(sensor.optics, "psf_kernels"):
            kernels = sensor.optics.psf_kernels(device=torch.device("cpu"), dtype=torch.float32).detach().cpu().numpy()

        if kernels is not None:
            zones, channels = kernels.shape[:2]
            psf_rows = []
            fig, axes = plt.subplots(zones, channels, figsize=(3.0 * channels, 2.2 * zones), squeeze=False)
            for z in range(zones):
                for c in range(channels):
                    ax = axes[z][c]
                    k = kernels[z, c]
                    h, w = k.shape
                    yy, xx = np.mgrid[:h, :w].astype(np.float32)
                    mass = float(k.sum()) if float(k.sum()) > 0 else 1.0
                    cx = float((k * xx).sum() / mass)
                    cy = float((k * yy).sum() / mass)
                    x0 = (w - 1) / 2.0
                    y0 = (h - 1) / 2.0
                    rr2 = (xx - cx) ** 2 + (yy - cy) ** 2
                    second_moment = float((k * rr2).sum() / mass)
                    rms_radius = float(np.sqrt(max(second_moment, 0.0)))
                    peak = float(k.max())
                    entropy = float(-(k * np.log(np.clip(k, 1e-12, 1.0))).sum())
                    psf_rows.append(
                        {
                            "stem": stem,
                            "zone": z,
                            "channel": c,
                            "centroid_x_px": f"{cx:.6f}",
                            "centroid_y_px": f"{cy:.6f}",
                            "shift_x_px": f"{cx - x0:.6f}",
                            "shift_y_px": f"{cy - y0:.6f}",
                            "rms_radius_px": f"{rms_radius:.6f}",
                            "second_moment_px2": f"{second_moment:.6f}",
                            "peak": f"{peak:.8f}",
                            "entropy": f"{entropy:.6f}",
                        }
                    )
                    ax.imshow(k, cmap="magma")
                    ax.plot([cx], [cy], marker="+", color="cyan", markersize=8, mew=1.5)
                    ax.set_title(f"zone {z} ch {c}\nrms {rms_radius:.2f}px shift {cx-x0:+.2f},{cy-y0:+.2f}", fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])
            title = "Learned constrained PSFs"
            if exposure is not None:
                title += f" | exposure {exposure:.3f}x"
            fig.suptitle(title)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{stem}_psf_kernels.png"), dpi=160)
            plt.close(fig)

            with open(os.path.join(out_dir, f"{stem}_psf_metrics.csv"), "w", newline="") as f:
                fields = [
                    "stem",
                    "zone",
                    "channel",
                    "centroid_x_px",
                    "centroid_y_px",
                    "shift_x_px",
                    "shift_y_px",
                    "rms_radius_px",
                    "second_moment_px2",
                    "peak",
                    "entropy",
                ]
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(psf_rows)

            fig, ax = plt.subplots(figsize=(6.0, 3.6))
            zone_ids = np.arange(zones)
            for c in range(channels):
                vals = [float(r["rms_radius_px"]) for r in psf_rows if int(r["channel"]) == c]
                ax.plot(zone_ids, vals, marker="o", linewidth=1.8, label=f"ch {c}")
            ax.set_xlabel("field zone")
            ax.set_ylabel("PSF RMS radius (px)")
            ax.set_title("Field-dependent PSF width")
            ax.grid(True, alpha=0.3)
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{stem}_psf_rms_by_field.png"), dpi=180)
            plt.close(fig)

        if cfa is not None:
            fig, ax = plt.subplots(figsize=(5.0, 2.8))
            im = ax.imshow(cfa, cmap="viridis", vmin=0.0, vmax=1.0)
            _T = int(round(cfa.shape[0] ** 0.5))
            ax.set_title(f"Learned CFA spectral weights ({_T}×{_T} tile)")
            ax.set_xlabel("RGB channel")
            ax.set_ylabel(f"{_T}×{_T} CFA site")
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(["R", "G", "B"])
            ax.set_yticks(list(range(cfa.shape[0])))
            for i in range(cfa.shape[0]):
                for j in range(cfa.shape[1]):
                    ax.text(j, i, f"{cfa[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{stem}_cfa_weights.png"), dpi=160)
            plt.close(fig)

            with open(os.path.join(out_dir, f"{stem}_cfa_weights.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["stem", "site", "R", "G", "B", "dominant_channel", "dominant_weight"])
                writer.writeheader()
                labels = ["R", "G", "B"]
                for i, row in enumerate(cfa):
                    dom = int(np.argmax(row))
                    writer.writerow(
                        {
                            "stem": stem,
                            "site": i,
                            "R": f"{float(row[0]):.8f}",
                            "G": f"{float(row[1]):.8f}",
                            "B": f"{float(row[2]):.8f}",
                            "dominant_channel": labels[dom],
                            "dominant_weight": f"{float(row[dom]):.8f}",
                        }
                    )

            fig, ax = plt.subplots(figsize=(5.2, 3.2))
            x = np.arange(cfa.shape[0])
            width = 0.24
            for j, label in enumerate(["R", "G", "B"]):
                ax.bar(x + (j - 1) * width, cfa[:, j], width=width, label=label)
            ax.set_xticks(x)
            ax.set_xticklabels([f"s{i}" for i in x])
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("spectral weight")
            ax.set_title(f"CFA spectral response by {_T}×{_T} site")
            ax.legend(frameon=False, ncol=3)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{stem}_cfa_bars.png"), dpi=180)
            plt.close(fig)

        if hasattr(sensor, "noise"):
            noise = sensor.noise
            xs = np.linspace(0.0, 1.0, 128, dtype=np.float32)
            bit = float(noise.bit_depth_value().detach().cpu())
            read = float(noise.read_noise_std_value().detach().cpu())
            shot = float(noise.shot_noise_scale_value().detach().cpu())
            black = float(getattr(sensor, "black_level", 0.0))
            iso = float(getattr(sensor, "iso_value", 100.0))
            iso_base = max(float(getattr(sensor, "iso_base", 100.0)), 1e-6)
            levels_total = 2.0 ** bit - 1.0
            signal_levels = max(levels_total - black, 1.0)
            signal_nbit = xs * signal_levels
            gain = iso / iso_base
            if str(getattr(sensor, "sensor_model", "normalized")).lower() in ("deeplens", "deeplens_nbit", "unprocess_nbit", "raw_nbit"):
                noise_std_nbit = np.sqrt((shot * np.sqrt(np.maximum(signal_nbit, 1e-6))) ** 2 * gain + (read ** 2) * gain * gain)
                noise_std_norm = noise_std_nbit / signal_levels
                quant_step = np.full_like(xs, 1.0 / signal_levels)
            else:
                noise_std_norm = np.sqrt(np.maximum(shot * xs, 1e-6) + read * read)
                quant_step = np.full_like(xs, 1.0 / max(2.0 ** bit - 1.0, 1.0))

            with open(os.path.join(out_dir, f"{stem}_noise_adc_curve.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["stem", "signal", "noise_std_normalized", "quant_step_normalized"])
                writer.writeheader()
                for sig, ns, qs in zip(xs, noise_std_norm, quant_step):
                    writer.writerow(
                        {
                            "stem": stem,
                            "signal": f"{float(sig):.6f}",
                            "noise_std_normalized": f"{float(ns):.8f}",
                            "quant_step_normalized": f"{float(qs):.8f}",
                        }
                    )

            fig, ax = plt.subplots(figsize=(5.8, 3.5))
            ax.plot(xs, noise_std_norm, linewidth=2.0, label="noise std")
            ax.plot(xs, quant_step, linewidth=1.6, linestyle="--", label="ADC step")
            ax.set_xlabel("normalized RAW signal")
            ax.set_ylabel("normalized magnitude")
            ax.set_title(f"Noise/ADC response | bit {bit:.2f}, ISO {iso:.0f}")
            ax.grid(True, alpha=0.3)
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{stem}_noise_adc_curve.png"), dpi=180)
            plt.close(fig)

        summary = {
            "stem": stem,
            "exposure_gain": exposure,
            "sensor_model": getattr(sensor, "sensor_model", ""),
            "raw_output": getattr(sensor, "raw_output", ""),
            "black_level": getattr(sensor, "black_level", ""),
            "iso": getattr(sensor, "iso_value", ""),
            "iso_base": getattr(sensor, "iso_base", ""),
        }
        if hasattr(sensor, "noise"):
            summary.update(
                {
                    "bit_depth_continuous": float(sensor.noise.bit_depth_value().detach().cpu()),
                    "bit_depth_deploy": int(sensor.noise.bit_depth),
                    "read_noise_std": float(sensor.noise.read_noise_std),
                    "shot_noise_scale": float(sensor.noise.shot_noise_scale),
                }
            )
        with open(os.path.join(out_dir, f"{stem}_camera_summary.csv"), "w", newline="") as f:
            fields = list(summary.keys())
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerow(summary)
    except Exception as e:
        logging.warning(f"[Monitor] sensor preview failed: {e}")


def _tensor_to_pil_image(img: torch.Tensor) -> Image.Image:
    if img.dim() == 4:
        if img.size(0) != 1:
            raise ValueError("Expected a single image tensor or batch size 1.")
        img = img[0]
    if img.dim() != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(img.shape)}")
    img = img.detach().cpu().float().clamp(0, 1)
    if img.size(0) == 1:
        arr = (img[0].numpy() * 255.0).round().astype("uint8")
        return Image.fromarray(arr, mode="L")
    if img.size(0) != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {img.size(0)}")
    arr = (img.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def _pil_image_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, copy=True)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


def _make_grid(tiles: torch.Tensor, nrow: int, padding: int = 4) -> torch.Tensor:
    if tiles.dim() != 4:
        raise ValueError(f"Expected NCHW tensor, got {tuple(tiles.shape)}")
    n, c, h, w = tiles.shape
    nrow = max(1, int(nrow))
    ncol = (n + nrow - 1) // nrow
    canvas_h = ncol * h + padding * (ncol + 1)
    canvas_w = nrow * w + padding * (nrow + 1)
    canvas = torch.zeros((c, canvas_h, canvas_w), dtype=tiles.dtype)
    idx = 0
    for r in range(ncol):
        for col in range(nrow):
            if idx >= n:
                break
            y = padding + r * (h + padding)
            x = padding + col * (w + padding)
            canvas[:, y : y + h, x : x + w] = tiles[idx]
            idx += 1
    return canvas


def _save_image(img: torch.Tensor, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _tensor_to_pil_image(img).save(path)

def _plot_training_curves(ckpt_dir: str) -> None:
    """Save loss and validation curves after each epoch."""
    try:
        import csv
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logging.debug(f"[Monitor] matplotlib unavailable; skipping curves: {e}")
        return

    def _read_rows(path: str) -> List[Dict[str, str]]:
        if not os.path.isfile(path):
            return []
        with open(path, "r", newline="") as f:
            return list(csv.DictReader(f))

    def _as_float(row: Dict[str, str], key: str) -> float:
        try:
            return float(row.get(key, "nan"))
        except Exception:
            return float("nan")

    os.makedirs(os.path.join(ckpt_dir, "plots"), exist_ok=True)
    summary = _read_rows(os.path.join(ckpt_dir, "epoch_summary.csv"))
    metrics = _read_rows(os.path.join(ckpt_dir, "metrics_log.csv"))

    if summary:
        xs = [_as_float(r, "epoch") for r in summary]
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        for key, label in [
            ("train_loss", "total"),
            ("train_ce", "CE/OHEM"),
            ("train_overlap", "Lovasz/Dice"),
            ("train_smooth", "smooth"),
            ("train_feedback", "task feedback"),
            ("train_distill", "distillation"),
            ("train_semantic_snr", "semantic SNR"),
            ("train_deploy", "deployable camera"),
            ("train_fixed_advantage", "fixed-camera hinge"),
            ("train_sensor_reg", "sensor reg"),
        ]:
            vals = [_as_float(r, key) for r in summary]
            if np.isfinite(vals).any():
                ax.plot(xs, vals, marker="o", linewidth=1.5, label=label)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("Training losses")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(ckpt_dir, "plots", "loss_curves.png"), dpi=160)
        plt.close(fig)

    if metrics:
        xs = [_as_float(r, "epoch") for r in metrics]
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        for key, label in [("mIoU", "mIoU"), ("pixel_acc", "pixel acc")]:
            vals = [_as_float(r, key) for r in metrics]
            ax.plot(xs, vals, marker="o", linewidth=1.5, label=label)
        ax.set_xlabel("epoch")
        ax.set_ylabel("score")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Validation metrics")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(ckpt_dir, "plots", "validation_curves.png"), dpi=160)
        plt.close(fig)

    trust_rows = _read_rows(os.path.join(ckpt_dir, "monitoring", "sensor_trust_region_steps.csv"))
    if trust_rows:
        comps = ["optics", "cfa", "exposure", "noise_adc", "readout", "other"]
        xs = [_as_float(r, "optimizer_step") for r in trust_rows]

        fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.0), sharex=True)
        for comp in comps:
            vals = [_as_float(r, f"lr_scale_{comp}") for r in trust_rows]
            if np.isfinite(vals).any():
                axes[0].plot(xs, vals, linewidth=1.5, label=comp)
        axes[0].set_ylabel("LR scale")
        axes[0].set_title("Per-component sensor trust-region learning rates")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(frameon=False, ncol=3)

        for comp in comps:
            vals = [_as_float(r, f"grad_{comp}") for r in trust_rows]
            if np.isfinite(vals).any():
                axes[1].plot(xs, vals, linewidth=1.2, label=comp)
        axes[1].set_yscale("symlog", linthresh=1e-8)
        axes[1].set_ylabel("grad norm")
        axes[1].set_title("Optics/sensor gradient evidence")
        axes[1].grid(True, alpha=0.3)

        for comp in comps:
            vals = [_as_float(r, f"step_{comp}_rel") for r in trust_rows]
            if np.isfinite(vals).any():
                axes[2].plot(xs, vals, linewidth=1.2, label=comp)
        axes[2].set_yscale("symlog", linthresh=1e-10)
        axes[2].set_xlabel("optimizer step")
        axes[2].set_ylabel("relative step")
        axes[2].set_title("Actual physical-parameter update size")
        axes[2].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(ckpt_dir, "plots", "sensor_trust_region_components.png"), dpi=180)
        plt.close(fig)

        probe_x = []
        probe_delta = []
        for r in trust_rows:
            delta = _as_float(r, "counterfactual_delta_new_minus_old")
            if math.isfinite(delta):
                probe_x.append(_as_float(r, "optimizer_step"))
                probe_delta.append(delta)
        if probe_x:
            fig, ax = plt.subplots(figsize=(7.5, 3.8))
            colors = ["#b91c1c" if d > 0 else "#047857" for d in probe_delta]
            ax.bar(probe_x, probe_delta, color=colors, width=0.8)
            ax.axhline(0.0, color="black", linewidth=1.0)
            ax.set_xlabel("optimizer step")
            ax.set_ylabel("new sensor loss - old sensor loss")
            ax.set_title("Same-batch counterfactual: did the sensor update help?")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(ckpt_dir, "plots", "sensor_counterfactual_delta.png"), dpi=180)
            plt.close(fig)

def _amp_context(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)

def _grad_scaler(enabled: bool):
    try:
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            return torch.amp.GradScaler(enabled=enabled)
    except TypeError:
        pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


@contextlib.contextmanager
def _temporary_camera_alpha(sensor: nn.Module, alpha: float | None):
    """Temporarily evaluate the fixed deployable camera view."""
    if alpha is None or not hasattr(sensor, "set_camera_blend_alpha"):
        yield
        return
    old_alpha = getattr(sensor, "camera_blend_alpha", None)
    sensor.set_camera_blend_alpha(float(alpha))
    try:
        yield
    finally:
        if old_alpha is not None:
            sensor.set_camera_blend_alpha(float(old_alpha))

# -------------------------
# Visualization helpers
# -------------------------
def colorize_mask(
    mask_tensor: torch.Tensor,
    palette=PALETTE_19,
    ignore_index: int = 255,
    ignore_color: Tuple[int, int, int] = (0, 0, 0),
) -> torch.Tensor:
    """
    mask_tensor: (H,W) long with class ids (0..K-1 or ignore=255).
    Returns (3,H,W) float in [0,1].
    """
    mask = mask_tensor.detach().cpu().numpy()
    H, W = mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[:, :] = ignore_color
    for k, (r, g, b) in enumerate(palette):
        rgb[mask == k] = (r, g, b)
    rgb[mask == ignore_index] = ignore_color
    rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return rgb


def mask_prediction_for_viz(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    """Use the same void/ignore mask for GT and prediction visualization."""
    pred_viz = pred.detach().clone()
    pred_viz[target == ignore_index] = ignore_index
    return pred_viz

def _robust_normalize_1ch(
    x: torch.Tensor,
    lo_q: float = 0.01,
    hi_q: float = 0.99,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-image robust normalization for RAW/delta visualization."""
    if x.dim() != 4 or x.shape[1] != 1:
        raise ValueError(f"Expected BCHW single-channel tensor, got {tuple(x.shape)}")
    b = x.shape[0]
    flat = x.detach().reshape(b, -1).float()
    lo_q = min(max(float(lo_q), 0.0), 1.0)
    hi_q = min(max(float(hi_q), lo_q + 1e-4), 1.0)
    lo = torch.quantile(flat, lo_q, dim=1).view(b, 1, 1, 1).to(x.device, x.dtype)
    hi = torch.quantile(flat, hi_q, dim=1).view(b, 1, 1, 1).to(x.device, x.dtype)
    fallback_lo = flat.min(dim=1)[0].view(b, 1, 1, 1).to(x.device, x.dtype)
    fallback_hi = flat.max(dim=1)[0].view(b, 1, 1, 1).to(x.device, x.dtype)
    too_flat = (hi - lo).abs() < eps
    lo = torch.where(too_flat, fallback_lo, lo)
    hi = torch.where(too_flat, fallback_hi, hi)
    return ((x - lo) / torch.clamp(hi - lo, min=eps)).clamp(0, 1)


def apply_colormap_1ch(
    x: torch.Tensor,
    cmap_name: str = "magma",
    robust: bool = True,
    lo_q: float = 0.01,
    hi_q: float = 0.99,
) -> torch.Tensor:
    """
    x: (B,1,H,W) in [0,1] or arbitrary range; returns (B,3,H,W) in [0,1]
    """
    try:
        import matplotlib
        cmap = matplotlib.colormaps.get_cmap(cmap_name)
    except Exception:
        cmap = None

    B, _, H, W = x.shape
    with torch.no_grad():
        if robust:
            x01 = _robust_normalize_1ch(x, lo_q=lo_q, hi_q=hi_q)
        else:
            x_flat = x.view(B, -1)
            minv = x_flat.min(dim=1, keepdim=True)[0]
            maxv = x_flat.max(dim=1, keepdim=True)[0]
            x01 = (x - minv.view(B,1,1,1)) / torch.clamp(
                (maxv - minv).view(B,1,1,1), min=1e-8
            )
            x01 = x01.clamp(0, 1)

    if cmap is None:
        return torch.cat([x01, x01.sqrt(), x01**2], dim=1).clamp(0, 1)

    x_np = x01.detach().cpu().numpy()[:, 0]  # (B,H,W)
    out = []
    for b in range(B):
        rgba = cmap(x_np[b])  # (H,W,4)
        rgb = torch.from_numpy(rgba[..., :3]).permute(2,0,1).float()  # (3,H,W)
        out.append(rgb)
    out = torch.stack(out, dim=0)
    return out.to(x.device)

def _make_labeled_tile_grid(
    tiles: torch.Tensor,
    nrow: int,
    titles: List[str],
    padding: int = 4,
    footer_h: int = 34,
) -> torch.Tensor:
    """Make a grid with labels in a footer below each tile, outside the image."""
    if tiles.dim() != 4:
        raise ValueError(f"Expected NCHW tensor, got {tuple(tiles.shape)}")
    n, c, h, w = tiles.shape
    if c != 3:
        raise ValueError("Labeled grids expect RGB tiles.")
    nrow = max(1, int(nrow))
    ncol = (n + nrow - 1) // nrow
    canvas_w = nrow * w + padding * (nrow + 1)
    canvas_h = ncol * (h + footer_h) + padding * (ncol + 1)
    img = Image.new("RGB", (canvas_w, canvas_h), (18, 18, 18))
    draw = ImageDraw.Draw(img)
    try:
        font_size = max(12, min(18, int(h * 0.075)))
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception:
        font = None

    for idx in range(n):
        r = idx // nrow
        col = idx % nrow
        x = padding + col * (w + padding)
        y = padding + r * (h + footer_h + padding)
        img.paste(_tensor_to_pil_image(tiles[idx].cpu()), (x, y))
        title = titles[col % len(titles)]
        label_y = y + h + max(4, footer_h // 5)
        draw.rectangle((x, y + h, x + w, y + h + footer_h), fill=(18, 18, 18))
        draw.text(
            (x + w // 2, label_y),
            title,
            fill=(245, 245, 245),
            anchor="ma",
            font=font,
        )

    return _pil_image_to_tensor(img).to(tiles.device)


def _stage_delta_visual(curr: torch.Tensor, prev: torch.Tensor, gain: float, cmap_name: str = "turbo") -> torch.Tensor:
    """False-color robust absolute change map for subtle camera stages."""
    if curr.shape[1] != prev.shape[1]:
        if curr.shape[1] == 1:
            prev = prev.mean(dim=1, keepdim=True)
        elif prev.shape[1] == 1:
            prev = prev.repeat(1, curr.shape[1], 1, 1)
    delta = (curr - prev).abs()
    if delta.shape[1] != 1:
        delta = delta.mean(dim=1, keepdim=True)
    return apply_colormap_1ch(delta * float(gain), cmap_name=cmap_name, robust=True, lo_q=0.02, hi_q=0.995)


def _stage_delta_or_image(
    curr: torch.Tensor,
    prev: torch.Tensor,
    gain: float,
    fallback_eps: float = 1e-4,
    cmap_name: str = "turbo",
) -> torch.Tensor:
    """Use a delta map when visible; otherwise show the actual stage image.

    Global exposure often initializes at exactly 1.0 gain, so the diagnostic
    delta is mathematically zero. A blank tile is misleading in paper figures;
    the fallback keeps the panel inspectable while preserving delta plots once
    the parameter moves.
    """
    if curr.shape[1] != prev.shape[1]:
        prev_cmp = prev.mean(dim=1, keepdim=True) if curr.shape[1] == 1 else prev.repeat(1, curr.shape[1], 1, 1)
    else:
        prev_cmp = prev
    delta = (curr - prev_cmp).abs()
    if float(delta.detach().amax().cpu()) < float(fallback_eps):
        if curr.dim() == 4 and curr.shape[1] == 1:
            return curr.repeat(1, 3, 1, 1).clamp(0, 1)
        return curr.clamp(0, 1)
    return _stage_delta_visual(curr, prev, gain=gain, cmap_name=cmap_name)

def vis_batch_with_stages(
    rgb: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    stages: Dict[str, torch.Tensor],
    out_png: str,
    palette=PALETTE_19,
    ignore_index: int = 255,
    max_items: int = 4,
    colorize_raw: bool = True,
    raw_cmap: str = "magma",
    titles: List[str] | None = None,
    diagnostic_stages: bool = True,
    panel_caption: str | None = None,
):
    """
    Save grid columns:
      [Input RGB | PSF residual | Exposure image | CFA/RAW view | Noise/ADC residual | GT | Pred]
    """
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    rgb_in       = torch.clamp(rgb.detach(), 0, 1)
    viz_size = rgb_in.shape[-2:]
    after_optics = torch.clamp(
        (stages.get("after_optics", rgb_in)).detach(), 0, 1
    )
    after_expo   = torch.clamp(
        (stages.get("after_exposure", after_optics)).detach(), 0, 1
    )
    cfa_stage    = (stages.get("after_cfa", after_expo)).detach()
    noise_stage  = (stages.get("after_noise", cfa_stage)).detach()

    def _resize_stage(t: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
        if t.dim() != 4 or t.shape[-2:] == viz_size:
            return t
        if mode == "nearest":
            return F.interpolate(t, size=viz_size, mode="nearest")
        return F.interpolate(t, size=viz_size, mode="bilinear", align_corners=False)

    after_optics = _resize_stage(after_optics)
    after_expo = _resize_stage(after_expo)
    cfa_stage = _resize_stage(cfa_stage)
    noise_stage = _resize_stage(noise_stage)

    def _ensure_color(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 4 and t.size(1) == 1:
            return apply_colormap_1ch(t, cmap_name=raw_cmap) if colorize_raw else t.repeat(1,3,1,1)
        if t.dim() == 4 and t.size(1) != 3:
            view = t[:, :3] if t.size(1) >= 3 else t.repeat(1, 3, 1, 1)
            flat = view.flatten(2)
            lo = flat.quantile(0.02, dim=2).view(view.shape[0], view.shape[1], 1, 1)
            hi = flat.quantile(0.98, dim=2).view(view.shape[0], view.shape[1], 1, 1)
            return ((view - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)
        return t

    token_stage = stages.get("task_tokens", None)
    if token_stage is not None:
        token_stage = _resize_stage(token_stage.detach())
    after_cfa   = torch.clamp(_ensure_color(cfa_stage),   0, 1)
    after_noise = torch.clamp(_ensure_color(noise_stage), 0, 1)
    token_panel = torch.clamp(_ensure_color(token_stage), 0, 1) if token_stage is not None else None
    is_rgb_passthrough = bool(stages.get("rgb_passthrough", False))

    if is_rgb_passthrough:
        optics_panel = rgb_in
        exposure_panel = rgb_in
        noise_panel = rgb_in
        after_cfa = rgb_in
    elif diagnostic_stages:
        optics_panel = _stage_delta_visual(after_optics, rgb_in, gain=10.0)
        exposure_panel = _stage_delta_or_image(after_expo, after_optics, gain=5.0)
        noise_panel = _stage_delta_visual(noise_stage, cfa_stage, gain=12.0)
    else:
        optics_panel = after_optics
        exposure_panel = after_expo
        noise_panel = after_noise
    if token_panel is not None:
        after_cfa = token_panel

    B = rgb_in.size(0)
    keep = min(B, max_items)
    rows = []
    for i in range(keep):
        pred_i = mask_prediction_for_viz(pred[i], target[i], ignore_index=ignore_index)
        gt_i = colorize_mask(target[i], palette, ignore_index=ignore_index)
        pr_i = colorize_mask(pred_i, palette, ignore_index=ignore_index)
        row = [
            rgb_in[i].detach().cpu(),
            optics_panel[i].detach().cpu(),
            exposure_panel[i].detach().cpu(),
            after_cfa[i].detach().cpu(),
            noise_panel[i].detach().cpu(),
            gt_i.cpu(),
            pr_i.cpu(),
        ]
        rows.append(torch.stack(row, dim=0))

    tiles = torch.cat(rows, dim=0)
    ncols = 7
    exposure_gain = stages.get("exposure_gain", None)
    exposure_title = "Exposure image"
    if exposure_gain is not None:
        try:
            exposure_title = f"Exposure {float(exposure_gain.detach().mean().cpu()):.3f}x"
        except Exception:
            exposure_title = "Exposure image"
    if titles is None:
        if is_rgb_passthrough:
            titles = [
                "Input RGB",
                "No optics",
                "No exposure",
                "No CFA/RAW",
                "No noise/ADC",
                "GT",
                "Pred",
            ]
        elif diagnostic_stages:
            titles = [
                "Input RGB",
                "PSF residual",
                exposure_title,
                "Task tokens" if token_panel is not None else "CFA/RAW view",
                "Noise/ADC residual",
                "GT",
                "Pred",
            ]
        else:
            titles = [
                "Input RGB",
                "After optics",
                exposure_title,
                "Task tokens" if token_panel is not None else "CFA/RAW view",
                "Noisy RAW",
                "GT",
                "Pred",
            ]
    grid_labeled = _make_labeled_tile_grid(tiles, nrow=ncols, titles=titles, padding=4, footer_h=34)

    if panel_caption:
        img = _tensor_to_pil_image(grid_labeled.cpu())
        caption_h = 54
        canvas = Image.new("RGB", (img.width, img.height + caption_h), (18, 18, 18))
        canvas.paste(img, (0, 0))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except Exception:
            font = None
        draw.text((14, img.height + 10), panel_caption, fill=(245, 245, 245), font=font)
        grid_labeled = _pil_image_to_tensor(canvas).to(tiles.device)

    _save_image(grid_labeled, out_png)

def vis_batch(rgb: torch.Tensor, pred: torch.Tensor, target: torch.Tensor,
              out_png: str, palette=PALETTE_19, ignore_index: int = 255, max_items: int = 4):
    B = rgb.size(0)
    keep = min(B, max_items)
    tiles = []
    for i in range(keep):
        rgb_i = torch.clamp(rgb[i].detach().cpu(), 0, 1)
        pred_i = mask_prediction_for_viz(pred[i], target[i], ignore_index=ignore_index)
        gt_i  = colorize_mask(target[i], palette, ignore_index=ignore_index)
        pr_i  = colorize_mask(pred_i, palette, ignore_index=ignore_index)
        row = torch.stack([rgb_i, gt_i, pr_i], dim=0)
        tiles.append(row)
    tiles = torch.cat(tiles, dim=0)
    grid = _make_grid(tiles, nrow=3, padding=4)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    _save_image(grid, out_png)

# -------------------------
# Metrics: confusion / IoU
# -------------------------
def update_confusion(conf: np.ndarray, pred: torch.Tensor, gt: torch.Tensor,
                     num_classes: int, ignore_index: int = 255):
    p = pred.detach().cpu().numpy().astype(np.int64)
    g = gt.detach().cpu().numpy().astype(np.int64)
    valid = (g != ignore_index)
    p = p[valid]; g = g[valid]
    idx = g * num_classes + p
    binc = np.bincount(idx, minlength=num_classes * num_classes)
    conf += binc.reshape(num_classes, num_classes)

def iou_from_confusion(conf: np.ndarray):
    tp = np.diag(conf)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    denom = (tp + fp + fn)
    present = denom > 0
    iou = np.zeros_like(tp, dtype=np.float64)
    np.divide(tp, denom, out=iou, where=present)
    miou = float(iou[present].mean()) if present.any() else 0.0
    return miou, iou

@torch.no_grad()
def imagewise_miou(pred: torch.Tensor, gt: torch.Tensor, num_classes: int, ignore_index: int = 255) -> float:
    """Per-image mIoU averaged over a mini-batch."""
    if pred.ndim != 3 or gt.ndim != 3:
        raise ValueError("Expected tensors with shape (B,H,W) for pred and gt.")
    vals: List[float] = []
    for b in range(pred.shape[0]):
        conf = np.zeros((num_classes, num_classes), dtype=np.int64)
        update_confusion(conf, pred[b : b + 1], gt[b : b + 1], num_classes=num_classes, ignore_index=ignore_index)
        miou, _ = iou_from_confusion(conf)
        if np.isfinite(miou):
            vals.append(float(miou))
    return float(np.mean(vals)) if vals else 0.0


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    if minutes:
        return f"{minutes:d}m{sec:02d}s"
    return f"{sec:d}s"


def _progress_bar(step_idx: int, total_steps: int, width: int = 24) -> str:
    total_steps = max(1, int(total_steps))
    frac = min(1.0, max(0.0, float(step_idx) / float(total_steps)))
    filled = int(round(frac * width))
    return "[" + "=" * filled + "." * (width - filled) + f"] {100.0 * frac:5.1f}%"

# -------------------------
# Class weights & Dice
# -------------------------
def compute_class_weights(
    loader: DataLoader,
    num_classes: int,
    ignore_index: int = 255,
    max_batches: int = 200,
    max_weight: float = 20.0,
    min_weight: float = 0.05,
    rare_class_boost: Dict[int, float] | None = None,
) -> torch.Tensor:
    """
    Histogram-based inverse-frequency weights with smoothing + clamping.
    """
    hist = torch.zeros(num_classes, dtype=torch.float64)
    with torch.no_grad():
        for b, (_, labels) in enumerate(loader):
            if b >= max_batches:
                break
            if labels.ndim == 4:  # (B,1,H,W) -> (B,H,W)
                labels = labels[:, 0]
            labels = labels.view(-1)
            mask = (labels != ignore_index)
            vals = labels[mask]
            vals = vals[(vals >= 0) & (vals < num_classes)]
            if vals.numel():
                hist += torch.bincount(vals, minlength=num_classes).double()

    eps = 1e-3
    freq = (hist + eps)
    inv = 1.0 / freq
    w = inv / inv.mean()  # normalize around 1
    w = torch.clamp(w, min=float(min_weight), max=float(max_weight))
    if rare_class_boost:
        for cls_id, boost in rare_class_boost.items():
            if 0 <= int(cls_id) < num_classes:
                w[int(cls_id)] = torch.clamp(w[int(cls_id)] * float(boost), min=float(min_weight), max=float(max_weight))
    return w.float()


def _parse_class_boosts(raw: Any) -> Dict[int, float]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return {int(k): float(v) for k, v in raw.items()}
    if isinstance(raw, list):
        out: Dict[int, float] = {}
        for item in raw:
            if isinstance(item, dict):
                out.update({int(k): float(v) for k, v in item.items()})
        return out
    return {}

def dice_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = 255,
              eps: float = 1e-6) -> torch.Tensor:
    """
    Multi-class soft Dice loss (1 - mean Dice) with ignore_index.
    logits:  (B, C, H, W)
    targets: (B, H, W) or (B,1,H,W) long with values in [0..C-1] or ignore_index
    """
    C = logits.shape[1]
    probs = F.softmax(logits, dim=1)

    if targets.ndim == 4 and targets.size(1) == 1:
        targets = targets[:, 0]
    targets = targets.long()

    valid = (targets != ignore_index)
    safe = targets.clone()
    safe[~valid] = 0
    safe = torch.clamp(safe, 0, C - 1)

    one_hot = F.one_hot(safe, num_classes=C).permute(0, 3, 1, 2).float()

    valid = valid.unsqueeze(1)
    probs = probs * valid
    one_hot = one_hot * valid

    dims = (0, 2, 3)
    intersect = (probs * one_hot).sum(dims)
    denom = probs.sum(dims) + one_hot.sum(dims) + eps
    dice = (2.0 * intersect + eps) / denom

    present = (one_hot.sum(dims) > 0)
    if present.any():
        return 1.0 - dice[present].mean()
    else:
        return logits.new_tensor(0.0)

def ohem_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor | None = None,
    ignore_index: int = 255,
    keep_frac: float = 1.0,
) -> torch.Tensor:
    ce = F.cross_entropy(
        logits,
        targets,
        weight=weight,
        ignore_index=ignore_index,
        reduction="none",
    )
    valid = (targets != ignore_index)
    if not valid.any():
        return ce.mean() * 0.0
    losses = ce[valid]
    keep_frac = float(keep_frac)
    if keep_frac <= 0.0 or keep_frac >= 1.0:
        return losses.mean()
    k = max(1, int(round(losses.numel() * keep_frac)))
    return torch.topk(losses, k=k, largest=True, sorted=False).values.mean()

def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = gt_sorted.numel()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / torch.clamp(union, min=1.0)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_softmax_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    C = probs.shape[1]
    probs = probs.permute(0, 2, 3, 1).reshape(-1, C)
    labels = targets.reshape(-1)
    valid = labels != ignore_index
    if not valid.any():
        return logits.mean() * 0.0
    probs = probs[valid]
    labels = labels[valid]
    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        errors = (fg - probs[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    if not losses:
        return logits.mean() * 0.0
    return torch.stack(losses).mean()

def edge_aware_smoothness_loss(
    logits: torch.Tensor,
    rgb: torch.Tensor,
    ignore_index: int | None = None,
    targets: torch.Tensor | None = None,
    sigma_rgb: float = 0.10,
) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    dx = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1])
    dy = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :])
    wx = torch.exp(-torch.mean(torch.abs(rgb[:, :, :, 1:] - rgb[:, :, :, :-1]), dim=1, keepdim=True) / max(sigma_rgb, 1e-6))
    wy = torch.exp(-torch.mean(torch.abs(rgb[:, :, 1:, :] - rgb[:, :, :-1, :]), dim=1, keepdim=True) / max(sigma_rgb, 1e-6))

    if targets is not None and ignore_index is not None:
        valid_x = ((targets[:, :, 1:] != ignore_index) & (targets[:, :, :-1] != ignore_index)).unsqueeze(1)
        valid_y = ((targets[:, 1:, :] != ignore_index) & (targets[:, :-1, :] != ignore_index)).unsqueeze(1)
        dx = dx * valid_x
        dy = dy * valid_y
        denom_x = valid_x.sum().clamp_min(1)
        denom_y = valid_y.sum().clamp_min(1)
        return ((dx * wx).sum() / denom_x + (dy * wy).sum() / denom_y) * 0.5
    return (dx * wx).mean() + (dy * wy).mean()


def semantic_boundary_mask(targets: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    """Return a soft-ish binary mask around semantic label transitions."""
    if targets.ndim == 4 and targets.size(1) == 1:
        targets = targets[:, 0]
    valid = targets != ignore_index
    mask = torch.zeros_like(targets, dtype=torch.float32)
    dx = (targets[:, :, 1:] != targets[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
    dy = (targets[:, 1:, :] != targets[:, :-1, :]) & valid[:, 1:, :] & valid[:, :-1, :]
    mask[:, :, 1:] = torch.maximum(mask[:, :, 1:], dx.float())
    mask[:, :, :-1] = torch.maximum(mask[:, :, :-1], dx.float())
    mask[:, 1:, :] = torch.maximum(mask[:, 1:, :], dy.float())
    mask[:, :-1, :] = torch.maximum(mask[:, :-1, :], dy.float())
    return mask


def semantic_sensor_snr_loss(
    sensor_out: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = 255,
    nonboundary_weight: float = 0.25,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Task-aware sensor information objective for camera co-design.

    A useful perception camera should preserve signal where semantic labels
    change, but should not create arbitrary high-frequency texture everywhere.
    This loss therefore maximizes sensor-output contrast across semantic
    boundaries relative to same-class neighborhoods. It gives optics, CFA,
    exposure, noise, and ADC parameters a direct physical-AI learning signal
    even when the downstream mIoU curve is flat.
    """
    if targets.ndim == 4 and targets.size(1) == 1:
        targets = targets[:, 0]
    if targets.shape[-2:] != sensor_out.shape[-2:]:
        targets = F.interpolate(
            targets[:, None].float(),
            size=sensor_out.shape[-2:],
            mode="nearest",
        )[:, 0].long()

    x = sensor_out.float()
    if x.ndim != 4 or x.shape[-1] < 2 or x.shape[-2] < 2:
        zero = sensor_out.mean() * 0.0
        return zero, {"boundary_signal": zero.detach(), "nonboundary_signal": zero.detach(), "snr": zero.detach()}

    valid = targets != int(ignore_index)
    valid_x = valid[:, :, 1:] & valid[:, :, :-1]
    valid_y = valid[:, 1:, :] & valid[:, :-1, :]
    boundary_x = (targets[:, :, 1:] != targets[:, :, :-1]) & valid_x
    boundary_y = (targets[:, 1:, :] != targets[:, :-1, :]) & valid_y
    nonboundary_x = (targets[:, :, 1:] == targets[:, :, :-1]) & valid_x
    nonboundary_y = (targets[:, 1:, :] == targets[:, :-1, :]) & valid_y

    grad_x = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]), dim=1)
    grad_y = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]), dim=1)

    boundary_vals = []
    nonboundary_vals = []
    if boundary_x.any():
        boundary_vals.append(grad_x[boundary_x])
    if boundary_y.any():
        boundary_vals.append(grad_y[boundary_y])
    if nonboundary_x.any():
        nonboundary_vals.append(grad_x[nonboundary_x])
    if nonboundary_y.any():
        nonboundary_vals.append(grad_y[nonboundary_y])

    if not boundary_vals or not nonboundary_vals:
        zero = sensor_out.mean() * 0.0
        return zero, {"boundary_signal": zero.detach(), "nonboundary_signal": zero.detach(), "snr": zero.detach()}

    boundary_signal = torch.cat(boundary_vals).mean()
    nonboundary_signal = torch.cat(nonboundary_vals).mean()
    snr = (boundary_signal + eps) / (nonboundary_signal + eps)
    loss = -torch.log(snr.clamp_min(eps)) + float(nonboundary_weight) * nonboundary_signal
    return loss, {
        "boundary_signal": boundary_signal.detach(),
        "nonboundary_signal": nonboundary_signal.detach(),
        "snr": snr.detach(),
    }


def task_feedback_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weight: torch.Tensor | None = None,
    class_pressure: torch.Tensor | None = None,
    ignore_index: int = 255,
    rare_class_ids: Iterable[int] | None = None,
    entropy_weight: float = 0.0,
    error_weight: float = 0.0,
    boundary_weight: float = 0.0,
    rare_weight: float = 0.0,
    detach_feedback: bool = True,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Training-time task feedback for camera co-design.

    This is deliberately not an inference loop. It builds a pixel weighting map
    from semantic uncertainty, label boundaries, and rare/task-critical classes,
    then applies an extra weighted CE term so optics/sensor parameters receive
    stronger gradients where perception is brittle.
    """
    if targets.ndim == 4 and targets.size(1) == 1:
        targets = targets[:, 0]
    valid = targets != ignore_index
    if not valid.any():
        zero = logits.mean() * 0.0
        return zero, {"entropy": zero.detach(), "boundary": zero.detach(), "rare": zero.detach(), "weight_mean": zero.detach()}

    with torch.no_grad() if detach_feedback else contextlib.nullcontext():
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1)
        entropy = entropy / math.log(max(logits.shape[1], 2))
        entropy = entropy.clamp(0.0, 1.0)
        safe_targets_for_prob = targets.clamp(min=0, max=logits.shape[1] - 1)
        true_prob = probs.gather(1, safe_targets_for_prob.unsqueeze(1)).squeeze(1)
        pred = probs.argmax(dim=1)
        error = 0.5 * (pred != targets).to(dtype=entropy.dtype) + 0.5 * (1.0 - true_prob).clamp(0.0, 1.0)
        boundary = semantic_boundary_mask(targets, ignore_index=ignore_index).to(device=logits.device, dtype=logits.dtype)
        rare = torch.zeros_like(entropy)
        if rare_class_ids:
            for cls_id in rare_class_ids:
                rare = torch.maximum(rare, (targets == int(cls_id)).to(dtype=rare.dtype))
        pressure = torch.ones_like(entropy)
        if class_pressure is not None:
            safe_targets = targets.clamp(min=0, max=int(class_pressure.numel()) - 1)
            pressure = class_pressure.to(device=logits.device, dtype=logits.dtype)[safe_targets]
            pressure = torch.where(valid, pressure, torch.ones_like(pressure))
        feedback = torch.ones_like(entropy)
        feedback = feedback + float(entropy_weight) * entropy
        feedback = feedback + float(error_weight) * error
        feedback = feedback + float(boundary_weight) * boundary
        feedback = feedback + float(rare_weight) * rare
        feedback = feedback * pressure
        feedback = feedback * valid.to(dtype=feedback.dtype)
        feedback = feedback / feedback[valid].mean().clamp_min(1e-6)

    ce_map = F.cross_entropy(
        logits,
        targets,
        weight=class_weight,
        ignore_index=ignore_index,
        reduction="none",
    )
    loss = (ce_map * feedback)[valid].mean()
    stats = {
        "entropy": entropy[valid].mean().detach(),
        "error": error[valid].mean().detach(),
        "boundary": boundary[valid].mean().detach(),
        "rare": rare[valid].mean().detach(),
        "pressure": pressure[valid].mean().detach(),
        "weight_mean": feedback[valid].mean().detach(),
    }
    return loss, stats


def build_distillation_teacher(cfg: Dict[str, Any], num_classes: int, device: torch.device) -> nn.Module | None:
    distill_cfg = cfg.get("distillation", {}) or {}
    train_cfg = cfg.get("train", {}) or {}
    if not bool(distill_cfg.get("enabled", float(train_cfg.get("distill_weight", 0.0)) > 0.0)):
        return None

    teacher_model_cfg = copy.deepcopy(
        distill_cfg.get(
            "teacher_model",
            {
                "name": "segformer_b1",
                "pretrained": True,
                "hf_model_id": "nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
            },
        )
    )
    teacher_cfg = copy.deepcopy(cfg)
    teacher_cfg["pipeline"] = {"input": "rgb"}
    teacher_cfg["model"] = teacher_model_cfg
    teacher = build_seg_model(teacher_cfg, in_channels=3, num_classes=num_classes).to(device)
    ckpt_path = str(distill_cfg.get("checkpoint", "") or "")
    if ckpt_path:
        payload = _load_ckpt(os.path.expanduser(ckpt_path), device)
        state = payload.get("model", payload)
        teacher.load_state_dict(state, strict=bool(distill_cfg.get("strict", False)))
        logging.info(f"[Distill] loaded teacher checkpoint: {ckpt_path}")
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    logging.info(f"[Distill] enabled teacher={teacher_model_cfg.get('name', 'teacher')}")
    return teacher


def distillation_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = 255,
    temperature: float = 2.0,
    confidence_threshold: float = 0.0,
) -> torch.Tensor:
    if teacher_logits.shape[-2:] != student_logits.shape[-2:]:
        teacher_logits = F.interpolate(teacher_logits, size=student_logits.shape[-2:], mode="bilinear", align_corners=False)
    if targets.shape[-2:] != student_logits.shape[-2:]:
        targets = F.interpolate(targets[:, None].float(), size=student_logits.shape[-2:], mode="nearest")[:, 0].long()
    valid = targets != int(ignore_index)
    if confidence_threshold > 0.0:
        confidence = F.softmax(teacher_logits.detach(), dim=1).max(dim=1).values
        valid = valid & (confidence >= float(confidence_threshold))
    if not valid.any():
        return student_logits.mean() * 0.0
    temp = max(float(temperature), 1e-3)
    student_logp = F.log_softmax(student_logits / temp, dim=1)
    teacher_prob = F.softmax(teacher_logits.detach() / temp, dim=1)
    kl_map = F.kl_div(student_logp, teacher_prob, reduction="none").sum(dim=1) * (temp * temp)
    return kl_map[valid].mean()


def teacher_guided_pseudo_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = 255,
    confidence_threshold: float = 0.0,
    boundary_weight: float = 0.0,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Confidence-gated hard teacher loss with optional boundary emphasis.

    This is meant for foundation-model teachers such as AD-SAM/SAM-derived
    semantic masks, Mask2Former/OneFormer, or a stronger SegFormer. It does not
    replace ground-truth supervision; it supplies extra dense targets where the
    teacher is confident and especially where boundaries are fragile.
    """
    if teacher_logits.shape[-2:] != student_logits.shape[-2:]:
        teacher_logits = F.interpolate(teacher_logits, size=student_logits.shape[-2:], mode="bilinear", align_corners=False)
    if targets.shape[-2:] != student_logits.shape[-2:]:
        targets = F.interpolate(targets[:, None].float(), size=student_logits.shape[-2:], mode="nearest")[:, 0].long()
    with torch.no_grad():
        teacher_prob = F.softmax(teacher_logits, dim=1)
        teacher_conf, pseudo = teacher_prob.max(dim=1)
        valid = targets != int(ignore_index)
        if confidence_threshold > 0.0:
            valid = valid & (teacher_conf >= float(confidence_threshold))
        boundary = semantic_boundary_mask(pseudo, ignore_index=ignore_index).to(
            device=student_logits.device,
            dtype=student_logits.dtype,
        )
        weights = torch.ones_like(teacher_conf, dtype=student_logits.dtype)
        weights = weights + float(boundary_weight) * boundary
        weights = weights * valid.to(dtype=weights.dtype)
        if valid.any():
            weights = weights / weights[valid].mean().clamp_min(1e-6)
    if not valid.any():
        zero = student_logits.mean() * 0.0
        return zero, {"teacher_coverage": zero.detach(), "teacher_boundary": zero.detach()}
    ce_map = F.cross_entropy(student_logits, pseudo, ignore_index=ignore_index, reduction="none")
    loss = (ce_map * weights)[valid].mean()
    stats = {
        "teacher_coverage": valid.float().mean().detach(),
        "teacher_boundary": boundary[valid].mean().detach(),
    }
    return loss, stats


class TaskFeedbackController:
    """Closed-loop task controller for co-design training.

    The controller observes validation IoU and raises next-epoch feedback
    pressure on underperforming classes. It is intentionally training-only:
    the deployed camera remains a single fixed optics/sensor configuration.
    """

    def __init__(self, num_classes: int, class_names: List[str], cfg: Dict[str, Any], device: torch.device):
        train_cfg = cfg.get("train", {})
        self.enabled = bool(train_cfg.get("task_feedback_controller", True))
        self.num_classes = int(num_classes)
        self.class_names = list(class_names)[: self.num_classes]
        while len(self.class_names) < self.num_classes:
            self.class_names.append(f"class{len(self.class_names)}")
        self.ema = float(train_cfg.get("task_feedback_controller_ema", 0.70))
        self.gamma = float(train_cfg.get("task_feedback_iou_gamma", 1.50))
        self.min_pressure = float(train_cfg.get("task_feedback_min_pressure", 0.75))
        self.max_pressure = float(train_cfg.get("task_feedback_max_pressure", 2.50))
        self.min_support = float(train_cfg.get("task_feedback_min_support", 1000.0))
        self.pressure = torch.ones(self.num_classes, device=device)
        self.last_iou = torch.full((self.num_classes,), float("nan"), device=device)
        self.last_support = torch.zeros(self.num_classes, device=device)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "pressure": self.pressure.detach().cpu(),
            "last_iou": self.last_iou.detach().cpu(),
            "last_support": self.last_support.detach().cpu(),
        }

    def load_state_dict(self, state: Dict[str, Any] | None, device: torch.device) -> None:
        if not state:
            return
        if "pressure" in state:
            self.pressure = torch.as_tensor(state["pressure"], dtype=torch.float32, device=device).flatten()[: self.num_classes]
            if self.pressure.numel() < self.num_classes:
                pad = torch.ones(self.num_classes - self.pressure.numel(), device=device)
                self.pressure = torch.cat([self.pressure, pad], dim=0)
        if "last_iou" in state:
            self.last_iou = torch.as_tensor(state["last_iou"], dtype=torch.float32, device=device).flatten()[: self.num_classes]
            if self.last_iou.numel() < self.num_classes:
                pad = torch.full((self.num_classes - self.last_iou.numel(),), float("nan"), device=device)
                self.last_iou = torch.cat([self.last_iou, pad], dim=0)
        if "last_support" in state:
            self.last_support = torch.as_tensor(state["last_support"], dtype=torch.float32, device=device).flatten()[: self.num_classes]
            if self.last_support.numel() < self.num_classes:
                pad = torch.zeros(self.num_classes - self.last_support.numel(), device=device)
                self.last_support = torch.cat([self.last_support, pad], dim=0)

    def current_pressure(self) -> torch.Tensor | None:
        if not self.enabled:
            return None
        return self.pressure.detach()

    def update_from_validation(self, per_class_iou: Iterable[float], class_support: Iterable[float] | None = None) -> None:
        if not self.enabled:
            return
        vals = torch.as_tensor(list(per_class_iou), dtype=torch.float32, device=self.pressure.device)
        if vals.numel() < self.num_classes:
            pad = torch.full((self.num_classes - vals.numel(),), float("nan"), device=self.pressure.device)
            vals = torch.cat([vals, pad], dim=0)
        vals = vals[: self.num_classes]
        if class_support is None:
            support = torch.full((self.num_classes,), self.min_support, dtype=torch.float32, device=self.pressure.device)
        else:
            support = torch.as_tensor(list(class_support), dtype=torch.float32, device=self.pressure.device)
            if support.numel() < self.num_classes:
                pad = torch.zeros(self.num_classes - support.numel(), device=self.pressure.device)
                support = torch.cat([support, pad], dim=0)
            support = support[: self.num_classes]
        valid = torch.isfinite(vals) & (vals >= 0.0) & (support >= self.min_support)
        if not valid.any():
            self.pressure = (self.ema * self.pressure + (1.0 - self.ema) * torch.ones_like(self.pressure)).clamp(
                self.min_pressure, self.max_pressure
            )
            self.last_iou = vals
            self.last_support = support
            return
        difficulty = (1.0 - vals.clamp(0.0, 1.0)).pow(self.gamma)
        raw = torch.ones_like(vals)
        raw[valid] = 1.0 + difficulty[valid]
        raw_mean = raw[valid].mean().clamp_min(1e-6)
        raw[valid] = (raw[valid] / raw_mean).clamp(self.min_pressure, self.max_pressure)
        raw[~valid] = 1.0
        self.pressure = (self.ema * self.pressure + (1.0 - self.ema) * raw).clamp(self.min_pressure, self.max_pressure)
        self.last_iou = vals
        self.last_support = support

    def write_epoch_report(self, out_dir: str, epoch: int) -> None:
        if not self.enabled:
            return
        import csv
        import json

        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "task_feedback_controller.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["epoch", "class_id", "class_name", "last_iou", "support", "pressure"])
            for idx in range(self.num_classes):
                iou = float(self.last_iou[idx].detach().cpu())
                writer.writerow([
                    int(epoch),
                    int(idx),
                    self.class_names[idx],
                    "" if not math.isfinite(iou) else iou,
                    float(self.last_support[idx].detach().cpu()),
                    float(self.pressure[idx].detach().cpu()),
                ])
        with open(os.path.join(out_dir, "task_feedback_controller_latest.json"), "w") as f:
            json.dump(
                {
                    "epoch": int(epoch),
                    "enabled": bool(self.enabled),
                    "ema": float(self.ema),
                    "gamma": float(self.gamma),
                    "min_pressure": float(self.min_pressure),
                    "max_pressure": float(self.max_pressure),
                    "classes": [
                        {
                            "class_id": idx,
                            "class_name": self.class_names[idx],
                            "last_iou": None
                            if not math.isfinite(float(self.last_iou[idx].detach().cpu()))
                            else float(self.last_iou[idx].detach().cpu()),
                            "support": float(self.last_support[idx].detach().cpu()),
                            "pressure": float(self.pressure[idx].detach().cpu()),
                        }
                        for idx in range(self.num_classes)
                    ],
                },
                f,
                indent=2,
            )


class SensorTrustRegionController:
    """Monitor and control optics/sensor learning with same-batch evidence.

    Backprop still computes the gradients. This controller measures whether
    the sensor-side update was useful, logs per-component gradients/steps, and
    adjusts only the sensor optimizer LR within a trust region.
    """

    COMPONENTS = ("optics", "cfa", "exposure", "noise_adc", "readout", "other")

    def __init__(
        self,
        sensor: nn.Module,
        cfg: Dict[str, Any],
        sensor_group_indices: Dict[str, int] | None,
        base_sensor_lrs: Dict[str, float] | None,
        ckpt_dir: str,
    ) -> None:
        train_cfg = cfg.get("train", {}) or {}
        ctrl_cfg = train_cfg.get("sensor_trust_region", {}) or {}
        self.enabled = bool(ctrl_cfg.get("enabled", train_cfg.get("sensor_trust_region_enabled", False)))
        self.sensor_group_indices = dict(sensor_group_indices or {})
        self.base_sensor_lrs = {str(k): float(v) for k, v in (base_sensor_lrs or {}).items()}
        self.ckpt_dir = ckpt_dir
        self.counterfactual_interval = int(ctrl_cfg.get("counterfactual_interval", 25))
        self.harm_tolerance = float(ctrl_cfg.get("harm_tolerance", 0.003))
        self.improve_tolerance = float(ctrl_cfg.get("improve_tolerance", 0.001))
        self.shrink_factor = float(ctrl_cfg.get("shrink_factor", 0.5))
        self.expand_factor = float(ctrl_cfg.get("expand_factor", 1.08))
        self.min_lr_mult = float(ctrl_cfg.get("min_lr_mult", 0.05))
        self.max_lr_mult = float(ctrl_cfg.get("max_lr_mult", 1.0))
        self.good_patience = int(ctrl_cfg.get("good_patience", 3))
        self.revert_on_harm = bool(ctrl_cfg.get("revert_on_harm", False))
        self.revert_threshold = float(ctrl_cfg.get("revert_threshold", 0.02))
        initial_lr_mult = float(ctrl_cfg.get("initial_lr_mult", 1.0))
        self.lr_scale_by_component = {
            comp: initial_lr_mult
            for comp in self.COMPONENTS
            if comp in self.sensor_group_indices
        }
        self.good_count_by_component = {comp: 0 for comp in self.COMPONENTS}
        self.bad_count_by_component = {comp: 0 for comp in self.COMPONENTS}
        self.bad_count = 0
        self.good_count = 0
        self.opt_step = 0
        self.last_grad: Dict[str, float] = {}
        self.last_step: Dict[str, float] = {}
        self.last_counterfactual: Dict[str, Any] = {}
        self._csv_path = os.path.join(ckpt_dir, "monitoring", "sensor_trust_region_steps.csv")
        self._latest_path = os.path.join(ckpt_dir, "monitoring", "sensor_trust_region_latest.json")
        self.enabled = self.enabled and bool(self.sensor_group_indices) and any(p.requires_grad for p in sensor.parameters())
        if self.enabled:
            logging.info(
                "[SensorTrust] enabled: components=%s counterfactual_interval=%d harm_tol=%.4f improve_tol=%.4f lr_mult=[%.3f, %.3f]",
                ",".join(sorted(self.sensor_group_indices.keys())),
                self.counterfactual_interval,
                self.harm_tolerance,
                self.improve_tolerance,
                self.min_lr_mult,
                self.max_lr_mult,
            )

    @staticmethod
    def _component(name: str) -> str:
        n = name.lower()
        if "optics" in n or "lens" in n or "psf" in n:
            return "optics"
        if "cfa" in n:
            return "cfa"
        if "exposure" in n:
            return "exposure"
        if "noise" in n or "bit_depth" in n or "read_noise" in n or "shot_noise" in n:
            return "noise_adc"
        if "task_token" in n or "readout" in n:
            return "readout"
        return "other"

    def capture_state(self, sensor: nn.Module) -> Dict[str, torch.Tensor]:
        if not self.enabled:
            return {}
        return {
            name: p.detach().clone()
            for name, p in sensor.named_parameters()
            if p.requires_grad
        }

    @staticmethod
    def restore_state(sensor: nn.Module, state: Dict[str, torch.Tensor]) -> None:
        if not state:
            return
        params = dict(sensor.named_parameters())
        with torch.no_grad():
            for name, value in state.items():
                if name in params:
                    params[name].copy_(value.to(device=params[name].device, dtype=params[name].dtype))

    def grad_stats(self, sensor: nn.Module) -> Dict[str, float]:
        stats = {k: 0.0 for k in self.COMPONENTS}
        if not self.enabled:
            return stats
        for name, p in sensor.named_parameters():
            if p.grad is None:
                continue
            comp = self._component(name)
            stats[comp] += float(p.grad.detach().float().norm().cpu()) ** 2
        for key in stats:
            stats[key] = math.sqrt(stats[key])
        stats["total"] = math.sqrt(sum(v * v for v in stats.values()))
        self.last_grad = stats
        return stats

    def step_stats(self, sensor: nn.Module, before_state: Dict[str, torch.Tensor]) -> Dict[str, float]:
        stats = {k: 0.0 for k in self.COMPONENTS}
        rel_stats = {f"{k}_rel": 0.0 for k in self.COMPONENTS}
        if not self.enabled or not before_state:
            stats["total"] = 0.0
            stats["total_rel"] = 0.0
            self.last_step = {**stats, **rel_stats}
            return self.last_step
        params = dict(sensor.named_parameters())
        denom = {k: 0.0 for k in self.COMPONENTS}
        for name, old in before_state.items():
            p = params.get(name)
            if p is None:
                continue
            comp = self._component(name)
            new = p.detach()
            diff = (new - old.to(device=new.device, dtype=new.dtype)).float()
            stats[comp] += float(diff.norm().cpu()) ** 2
            denom[comp] += float(new.float().norm().cpu()) ** 2
        for key in self.COMPONENTS:
            stats[key] = math.sqrt(stats[key])
            rel_stats[f"{key}_rel"] = stats[key] / max(math.sqrt(denom[key]), 1e-12)
        stats["total"] = math.sqrt(sum(stats[k] * stats[k] for k in self.COMPONENTS))
        total_param = math.sqrt(sum(denom.values()))
        stats["total_rel"] = stats["total"] / max(total_param, 1e-12)
        self.last_step = {**stats, **rel_stats}
        return self.last_step

    def _task_loss(
        self,
        sensor: nn.Module,
        model: nn.Module,
        imgs: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int,
        class_weight: torch.Tensor | None,
        camera_alpha_override: float | None = None,
    ) -> float:
        sensor_was_training = sensor.training
        model_was_training = model.training
        sensor.eval()
        model.eval()
        try:
            with torch.no_grad():
                with _temporary_camera_alpha(sensor, camera_alpha_override):
                    raw = sensor(imgs)
                    logits = align_logits_to_labels(model(raw), labels)
                    loss = F.cross_entropy(
                        logits,
                        labels,
                        weight=class_weight,
                        ignore_index=int(ignore_index),
                    )
            return float(loss.detach().cpu())
        finally:
            sensor.train(sensor_was_training)
            model.train(model_was_training)

    def _dominant_component(self, step_stats: Dict[str, float]) -> str:
        candidates = [comp for comp in self.COMPONENTS if comp in self.sensor_group_indices]
        if not candidates:
            return "other"
        return max(
            candidates,
            key=lambda comp: (
                float(step_stats.get(f"{comp}_rel", 0.0)),
                float(step_stats.get(comp, 0.0)),
                float(self.last_grad.get(comp, 0.0)),
            ),
        )

    def _adjust_sensor_lr(self, optim: torch.optim.Optimizer, component: str, factor: float) -> float | None:
        group_idx = self.sensor_group_indices.get(component)
        if group_idx is None or group_idx >= len(optim.param_groups):
            return
        base_lr = float(self.base_sensor_lrs.get(component, 0.0))
        if base_lr <= 0.0:
            return None
        group = optim.param_groups[group_idx]
        current_lr = float(group.get("lr", base_lr))
        target_lr = current_lr * float(factor)
        lo = base_lr * self.min_lr_mult
        hi = base_lr * self.max_lr_mult
        target_lr = max(lo, min(hi, target_lr))
        group["lr"] = target_lr
        self.lr_scale_by_component[component] = target_lr / max(base_lr, 1e-12)
        return target_lr

    def _refresh_lr_scales(self, optim: torch.optim.Optimizer) -> Dict[str, float]:
        scales: Dict[str, float] = {}
        for comp, group_idx in self.sensor_group_indices.items():
            if group_idx >= len(optim.param_groups):
                continue
            base_lr = max(float(self.base_sensor_lrs.get(comp, 0.0)), 1e-12)
            lr = float(optim.param_groups[group_idx].get("lr", base_lr))
            scales[comp] = lr / base_lr
        self.lr_scale_by_component.update(scales)
        return scales

    def scale_sensor_lrs(
        self,
        optim: torch.optim.Optimizer,
        factor: float,
        *,
        components: Iterable[str] | None = None,
        min_lr_mult: float | None = None,
    ) -> Dict[str, float]:
        """Scale sensor optimizer groups together for epoch-level guardrails."""
        if not self.sensor_group_indices:
            return {}
        comps = list(components) if components is not None else list(self.sensor_group_indices.keys())
        old_min = self.min_lr_mult
        if min_lr_mult is not None:
            self.min_lr_mult = float(min_lr_mult)
        try:
            for comp in comps:
                self._adjust_sensor_lr(optim, str(comp), float(factor))
        finally:
            self.min_lr_mult = old_min
        return self._refresh_lr_scales(optim)

    def after_step(
        self,
        optim: torch.optim.Optimizer,
        sensor: nn.Module,
        model: nn.Module,
        imgs: torch.Tensor,
        labels: torch.Tensor,
        before_state: Dict[str, torch.Tensor],
        epoch: int,
        global_step: int,
        camera_alpha: float,
        ignore_index: int,
        class_weight: torch.Tensor | None,
    ) -> None:
        if not self.enabled:
            return
        self.opt_step += 1
        step_stats = self.step_stats(sensor, before_state)
        action = "hold"
        action_component = ""
        loss_old = ""
        loss_new = ""
        delta = ""
        should_probe = self.counterfactual_interval > 0 and (self.opt_step % self.counterfactual_interval == 0)
        if should_probe and before_state:
            dominant_component = self._dominant_component(step_stats)
            action_component = dominant_component
            after_state = self.capture_state(sensor)
            # The trust-region question is: did this sensor update improve the
            # fixed deployment camera?  During curriculum training, the main
            # forward may be RGB-blended, but the controller must judge the
            # real camera path.
            loss_new_f = self._task_loss(sensor, model, imgs, labels, ignore_index, class_weight, camera_alpha_override=1.0)
            self.restore_state(sensor, before_state)
            loss_old_f = self._task_loss(sensor, model, imgs, labels, ignore_index, class_weight, camera_alpha_override=1.0)
            self.restore_state(sensor, after_state)
            delta_f = loss_new_f - loss_old_f
            loss_old = f"{loss_old_f:.8f}"
            loss_new = f"{loss_new_f:.8f}"
            delta = f"{delta_f:.8f}"
            if delta_f > self.harm_tolerance:
                self.bad_count_by_component[dominant_component] = self.bad_count_by_component.get(dominant_component, 0) + 1
                self.good_count_by_component[dominant_component] = 0
                self._adjust_sensor_lr(optim, dominant_component, self.shrink_factor)
                action = f"shrink_{dominant_component}_lr"
                if self.revert_on_harm and delta_f > self.revert_threshold:
                    self.restore_state(sensor, before_state)
                    action = f"revert_and_shrink_{dominant_component}_lr"
            elif delta_f < -self.improve_tolerance:
                self.good_count_by_component[dominant_component] = self.good_count_by_component.get(dominant_component, 0) + 1
                self.bad_count_by_component[dominant_component] = 0
                if self.good_count_by_component[dominant_component] >= self.good_patience:
                    self._adjust_sensor_lr(optim, dominant_component, self.expand_factor)
                    self.good_count_by_component[dominant_component] = 0
                    action = f"expand_{dominant_component}_lr"
            else:
                action = "hold_neutral"
            self.last_counterfactual = {
                "loss_old_sensor": loss_old_f,
                "loss_new_sensor": loss_new_f,
                "delta_new_minus_old": delta_f,
                "action": action,
                "dominant_component": dominant_component,
            }
            scales = self._refresh_lr_scales(optim)
            logging.info(
                "[SensorTrust] epoch=%d step=%d opt_step=%d delta=%.6f action=%s component=%s lr_scale=%.3f step_rel=%.3e grad=%.3e",
                epoch,
                global_step,
                self.opt_step,
                delta_f,
                action,
                dominant_component,
                float(scales.get(dominant_component, 1.0)),
                float(step_stats.get("total_rel", 0.0)),
                float(self.last_grad.get("total", 0.0)),
            )

        self._write_step(epoch, global_step, camera_alpha, optim, action, action_component, loss_old, loss_new, delta, step_stats)

    def _write_step(
        self,
        epoch: int,
        global_step: int,
        camera_alpha: float,
        optim: torch.optim.Optimizer,
        action: str,
        action_component: str,
        loss_old: str,
        loss_new: str,
        delta: str,
        step_stats: Dict[str, float],
    ) -> None:
        import csv
        import json

        os.makedirs(os.path.dirname(self._csv_path), exist_ok=True)
        self._refresh_lr_scales(optim)
        fields = [
            "epoch",
            "global_step",
            "optimizer_step",
            "camera_alpha",
            "action",
            "action_component",
            "counterfactual_loss_old_sensor",
            "counterfactual_loss_new_sensor",
            "counterfactual_delta_new_minus_old",
            "grad_total",
            "step_total",
            "step_total_rel",
        ]
        for comp in self.COMPONENTS:
            fields.extend([f"lr_{comp}", f"lr_scale_{comp}", f"grad_{comp}", f"step_{comp}", f"step_{comp}_rel"])
        row = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "optimizer_step": int(self.opt_step),
            "camera_alpha": float(camera_alpha),
            "action": action,
            "action_component": action_component,
            "counterfactual_loss_old_sensor": loss_old,
            "counterfactual_loss_new_sensor": loss_new,
            "counterfactual_delta_new_minus_old": delta,
            "grad_total": self.last_grad.get("total", 0.0),
            "step_total": step_stats.get("total", 0.0),
            "step_total_rel": step_stats.get("total_rel", 0.0),
        }
        for comp in self.COMPONENTS:
            group_idx = self.sensor_group_indices.get(comp)
            row[f"lr_{comp}"] = optim.param_groups[group_idx].get("lr", "") if group_idx is not None and group_idx < len(optim.param_groups) else ""
            row[f"lr_scale_{comp}"] = self.lr_scale_by_component.get(comp, "")
            row[f"grad_{comp}"] = self.last_grad.get(comp, 0.0)
            row[f"step_{comp}"] = step_stats.get(comp, 0.0)
            row[f"step_{comp}_rel"] = step_stats.get(f"{comp}_rel", 0.0)
        write_header = not os.path.exists(self._csv_path)
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        with open(self._latest_path, "w") as f:
            json.dump(row, f, indent=2)


def align_logits_to_labels(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Resize logits to label resolution when the sensor packs RAW at half-res."""
    target_size = labels.shape[-2:]
    if logits.shape[-2:] != target_size:
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
    return logits

# -------------------------
# Optional DenseCRF (eval only)
# -------------------------
def refine_with_crf(rgb_np, prob_np, sxy=3, srgb=5, compat=3, iters=5):
    """
    rgb_np: HxWx3 uint8
    prob_np: CxHxW float
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except Exception as e:
        logging.warning(f"[CRF] pydensecrf not available ({e}); skipping refinement.")
        return prob_np

    prob_np = np.ascontiguousarray(prob_np.astype(np.float32))
    if rgb_np.dtype != np.uint8:
        rgb_np = (np.clip(rgb_np * 255.0, 0, 255)).astype(np.uint8)
    rgb_np = np.ascontiguousarray(rgb_np)

    C, H, W = prob_np.shape
    d = dcrf.DenseCRF2D(W, H, C)

    U = unary_from_softmax(prob_np)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=sxy, compat=compat)
    d.addPairwiseBilateral(sxy=sxy, srgb=srgb, rgbim=rgb_np, compat=compat)

    Q = d.inference(iters)
    return np.asarray(Q, dtype=np.float32).reshape(C, H, W)

# -------------------------
# Sensor module
# -------------------------
class TaskTokenReadout(nn.Module):
    """Compact machine-native readout from a physical RAW/CFA stream.

    The readout is intentionally small and constrained: it can mix packed RGGB,
    optional soft-CFA RGB, and ISO metadata into perception tokens, but it does
    not replace the optics/CFA/noise/ADC simulation with a large learned ISP.
    """

    def __init__(
        self,
        token_channels: int = 8,
        hidden_channels: int = 16,
        use_soft_rgb: bool = True,
        use_iso: bool = True,
    ):
        super().__init__()
        self.token_channels = int(max(1, token_channels))
        self.hidden_channels = int(max(4, hidden_channels))
        self.use_soft_rgb = bool(use_soft_rgb)
        self.use_iso = bool(use_iso)
        basis_channels = 4 + (3 if self.use_soft_rgb else 0) + (1 if self.use_iso else 0)
        self.basis_channels = basis_channels

        self.identity_mix = nn.Conv2d(basis_channels, self.token_channels, kernel_size=1, bias=False)
        self.residual_mix = nn.Sequential(
            nn.Conv2d(basis_channels, self.hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, self.hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                self.hidden_channels,
                self.hidden_channels,
                kernel_size=3,
                padding=1,
                groups=self.hidden_channels,
                bias=False,
            ),
            nn.Conv2d(self.hidden_channels, self.token_channels, kernel_size=1, bias=True),
        )
        self.residual_scale_raw = nn.Parameter(torch.tensor(-2.1972246))  # sigmoid ~= 0.10
        self._init_identity()

    def _init_identity(self) -> None:
        with torch.no_grad():
            self.identity_mix.weight.zero_()
            for c in range(min(self.token_channels, self.basis_channels)):
                self.identity_mix.weight[c, c, 0, 0] = 1.0
            for module in self.residual_mix.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(
        self,
        raw: torch.Tensor,
        cfa_weights: torch.Tensor,
        iso_norm: float | torch.Tensor,
        demosaic_kernel_size: int = 5,
    ) -> torch.Tensor:
        packed = superpixel_demux(raw)
        basis = [packed]
        if self.use_soft_rgb:
            rgb = soft_cfa_demosaic(raw, cfa_weights, kernel_size=demosaic_kernel_size)
            rgb = F.avg_pool2d(rgb, kernel_size=2, stride=2)
            basis.append(rgb)
        if self.use_iso:
            iso_value = torch.as_tensor(iso_norm, device=raw.device, dtype=raw.dtype)
            iso_channel = torch.full(
                (packed.shape[0], 1, packed.shape[2], packed.shape[3]),
                float(iso_value.detach().cpu()),
                device=packed.device,
                dtype=packed.dtype,
            )
            basis.append(iso_channel)
        x = torch.cat(basis, dim=1)
        residual_scale = torch.sigmoid(self.residual_scale_raw)
        return self.identity_mix(x) + residual_scale * self.residual_mix(x)

    def regularization(self) -> Dict[str, torch.Tensor]:
        w = self.identity_mix.weight.flatten(1)
        gram = F.normalize(w, dim=1) @ F.normalize(w, dim=1).t()
        eye = torch.eye(gram.shape[0], device=gram.device, dtype=torch.bool)
        token_collapse = gram[~eye].abs().mean() if gram.numel() > gram.shape[0] else gram.new_tensor(0.0)
        residual_energy = torch.sigmoid(self.residual_scale_raw).square()
        weight_l2 = self.identity_mix.weight.square().mean()
        for module in self.residual_mix.modules():
            if isinstance(module, nn.Conv2d):
                weight_l2 = weight_l2 + module.weight.square().mean()
        return {
            "task_token_collapse": token_collapse,
            "task_token_residual_energy": residual_energy,
            "task_token_weight_l2": weight_l2,
        }

    def export_design(self) -> Dict[str, Any]:
        with torch.no_grad():
            mix = self.identity_mix.weight.detach().cpu().flatten(1)
            return {
                "type": "task_token_readout",
                "token_channels": self.token_channels,
                "basis_channels": self.basis_channels,
                "hidden_channels": self.hidden_channels,
                "use_soft_rgb_basis": self.use_soft_rgb,
                "use_iso_basis": self.use_iso,
                "residual_scale": float(torch.sigmoid(self.residual_scale_raw).detach().cpu()),
                "identity_mix_l1_per_token": mix.abs().sum(dim=1).tolist(),
            }


class CoDesignSensor(nn.Module):
    """
    Optics -> exposure -> learnable 2x2 CFA -> Poisson+Gaussian noise + N-bit quant.

    Optics can be a trainable constrained field-PSF, a fixed DeepLens GeoLens
    renderer, or identity. The trainable PSF path is the reviewer-facing
    co-design setting because its parameters are optimized with the task and
    exported as a fixed camera design.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.device = cfg["device"]
        self.camera_blend_alpha = 1.0

        dbg = cfg.get("debug", {})
        self.dbg_bypass_optics = bool(dbg.get("bypass_optics", False))
        self.dbg_bypass_cfa    = bool(dbg.get("bypass_cfa", False))
        self.dbg_bypass_noise  = bool(dbg.get("bypass_noise", False))
        self.dbg_dump_stats    = bool(dbg.get("dump_stats", False))

        s = cfg["sensor"]
        self.tolerance_cfg = (s.get("tolerance") or {})
        self.sensor_model = str(s.get("sensor_model", "normalized")).lower()
        self.raw_output = str(s.get("raw_output", "mosaic")).lower()
        self.eval_stochastic_noise = bool(s.get("eval_stochastic_noise", False))
        self.force_stochastic_noise = False
        self.task_token_aliases = ("task_tokens", "tokens", "machine_tokens", "perception_tokens")
        self.black_level = float(s.get("black_level", 0.0))
        self.iso_base = float(s.get("iso_base", 100.0))
        self.iso_value = float(s.get("iso", s.get("iso_value", self.iso_base)))
        self.iso_scale = float(s.get("iso_scale", 1000.0))
        self.cfa_temperature_start = float(s.get("cfa_temperature_start", s.get("cfa_temperature", 1.0)))
        self.cfa_temperature_end = float(s.get("cfa_temperature_end", 1.0))
        self.cfa_temperature_epochs = int(s.get("cfa_temperature_epochs", 0))
        self.cfa = LearnableCFA(
            init=s.get("cfa_init", "bayer_rggb"),
            init_floor=float(s.get("cfa_init_floor", 1e-4)),
            temperature=self.cfa_temperature_start,
            min_transmittance=float(s.get("cfa_min_transmittance", 0.0)),
            tile_size=int(s.get("cfa_tile_size", 2)),
        )
        if not bool(s.get("learn_cfa", True)):
            for p in self.cfa.parameters():
                p.requires_grad_(False)
        self.noise = PoissonGaussianNoiseQuant(
            bit_depth=int(s.get("bit_depth", 6)),
            read_noise_std=float(s.get("read_noise_std", 0.005)),
            shot_noise_scale=float(s.get("shot_noise_scale", 0.02)),
            learn_noise=bool(s.get("learn_noise", True)),
            learn_bit_depth=bool(s.get("learn_bit_depth", True)),
            min_read_noise_std=float(s.get("min_read_noise_std", 1e-5)),
            max_read_noise_std=float(s.get("max_read_noise_std", 0.05)),
            min_shot_noise_scale=float(s.get("min_shot_noise_scale", 1e-5)),
            max_shot_noise_scale=float(s.get("max_shot_noise_scale", 0.12)),
            min_bit_depth=float(s.get("min_bit_depth", 4.0)),
            max_bit_depth=float(s.get("max_bit_depth", 12.0)),
        )
        self.exposure_logit = nn.Parameter(
            torch.tensor(float(s.get("exposure_init", 0.0))),
            requires_grad=bool(s.get("learn_exposure", True)),
        )
        self.task_token_readout = None
        if self.raw_output in self.task_token_aliases:
            self.task_token_readout = TaskTokenReadout(
                token_channels=int(s.get("task_token_channels", 8)),
                hidden_channels=int(s.get("task_token_hidden_channels", 16)),
                use_soft_rgb=bool(s.get("task_token_use_soft_rgb", True)),
                use_iso=bool(s.get("task_token_use_iso", True)),
            )
            if not bool(s.get("learn_task_token_readout", True)):
                for p in self.task_token_readout.parameters():
                    p.requires_grad_(False)
        self.pixel_pitch_m = float(s.get("sensor_pitch", 1.4e-6))
        self.rgb_sensor = None
        if self.sensor_model in ("deeplens", "deeplens_nbit", "unprocess_nbit", "raw_nbit"):
            if RGBSensor is None:
                raise ImportError("deeplens.sensor.rgb_sensor.RGBSensor is required for sensor.sensor_model='deeplens_nbit'.")
            bit_for_isp = int(round(float(s.get("isp_bit_depth", s.get("bit_depth", 10)))))
            self.rgb_sensor = RGBSensor(
                bit=bit_for_isp,
                black_level=float(s.get("black_level", 64.0)),
                res=tuple(s.get("res", (4000, 3000))),
                size=tuple(s.get("size", (8.0, 6.0))),
                bayer_pattern=str(s.get("bayer_pattern", "rggb")),
                iso_base=float(s.get("iso_base", 100.0)),
                read_noise_std=float(s.get("read_noise_std", 0.5)),
                shot_noise_std_alpha=float(s.get("shot_noise_scale", 0.4)),
            )
            # We use RGBSensor's differentiable inverse ISP, but the optimized
            # CFA/noise/ADC below remain our explicit co-design parameters.
            for p in self.rgb_sensor.parameters():
                p.requires_grad_(False)

        self.focus_distance_m = float(cfg["train"].get("depth", 2.0))

        lens_cfg = cfg.get("lens") or {}
        lens_json = str(lens_cfg.get("path", "") or "").strip()
        lens_mode = str(lens_cfg.get("mode", "") or "").strip().lower()
        if not lens_mode:
            lens_mode = "geolens" if lens_json else "identity"

        self._use_optics = False
        self._optics_mode = lens_mode
        self.lens = None
        self.optics = None

        if lens_mode in ("trainable_psf", "constrained_psf", "field_psf"):
            self.optics = ConstrainedFieldPSF.from_config(lens_cfg)
            if not bool(lens_cfg.get("learn_optics", True)):
                for p in self.optics.parameters():
                    p.requires_grad_(False)
            self._use_optics = True
            logging.info(
                "[CoDesignSensor] Using constrained trainable field PSF "
                f"({self.optics.num_zones} zones, {self.optics.kernel_size}x{self.optics.kernel_size})."
            )
        elif lens_mode in ("geolens", "deeplens") and lens_json:
            try:
                from deeplens import GeoLens
                self.lens = GeoLens(filename=lens_json).to(self.device)
                surfs = getattr(self.lens, "surfaces", None)
                if not surfs:
                    logging.warning("[CoDesignSensor] Lens JSON has no surfaces; disabling optics.")
                    self.lens = None
                else:
                    self._use_optics = True
                    logging.info(f"[CoDesignSensor] Using GeoLens from JSON: {lens_json}")
            except Exception as e:
                logging.warning(f"[CoDesignSensor] Failed to load GeoLens ({lens_json}); using identity optics. Error: {e}")
        elif lens_mode not in ("identity", "none", "off", ""):
            logging.warning(f"[CoDesignSensor] Unknown lens.mode={lens_mode!r}; using identity optics.")

    def _prepare_lens(self, H: int, W: int):
        if not (self._use_optics and self.lens is not None):
            return
        if hasattr(self.lens, "set_sensor_res"):
            try:
                self.lens.set_sensor_res((H, W))
            except Exception:
                pass
        for attr in ("pixel", "pixsize", "pixel_size"):
            if hasattr(self.lens, attr):
                try:
                    setattr(self.lens, attr, self.pixel_pitch_m)
                except Exception:
                    pass

    def _log_stats(self, tag: str, x: torch.Tensor):
        if not self.dbg_dump_stats:
            return
        with torch.no_grad():
            m = x.mean().item(); s = x.std().item()
            mn = x.min().item(); mx = x.max().item()
        logging.info(f"[stats] {tag}: mean={m:.4f} std={s:.4f} min={mn:.4f} max={mx:.4f}")

    def set_camera_blend_alpha(self, alpha: float) -> None:
        """Blend simulated camera output with RGB input during curriculum warmup.

        This keeps pretrained RGB backbones from being shocked by a camera
        distribution shift before the task head has stabilized. The deployed
        setting is alpha=1.0, i.e. pure learned/fixed camera output.
        """
        self.camera_blend_alpha = float(max(0.0, min(1.0, alpha)))

    def set_cfa_temperature_for_epoch(self, epoch: int) -> None:
        if not hasattr(self.cfa, "set_temperature"):
            return
        total = int(max(0, self.cfa_temperature_epochs))
        if total <= 0:
            self.cfa.set_temperature(self.cfa_temperature_end)
            return
        progress = max(0.0, min(1.0, float(max(1, epoch) - 1) / float(total)))
        temperature = self.cfa_temperature_start + progress * (self.cfa_temperature_end - self.cfa_temperature_start)
        self.cfa.set_temperature(temperature)

    @staticmethod
    def _round_ste(x: torch.Tensor) -> torch.Tensor:
        rounded = torch.round(x)
        return x + (rounded - x).detach()

    def _unprocess_to_raw_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        """Approximate DeepLens' RGBSensor.unprocess contract for RGB datasets."""
        if self.rgb_sensor is None:
            return rgb
        dtype = rgb.dtype
        # InvertibleISP buffers are float32; run this path in fp32 for stable
        # gamma/CCM/AWB inverse math, then cast back for the main pipeline.
        raw_rgb = self.rgb_sensor.unprocess(rgb.float(), in_type="rgb")
        return torch.clamp(raw_rgb.to(dtype=dtype), 0.0, 1.0)

    def _tolerance_enabled(self) -> bool:
        return bool(self.tolerance_cfg.get("enabled", False))

    def _tol_std(self, name: str, default: float = 0.0) -> float:
        return float(self.tolerance_cfg.get(name, default)) if self._tolerance_enabled() else 0.0

    def _noise_quant_deeplens_nbit(self, raw: torch.Tensor, perturb: bool = False) -> torch.Tensor:
        """DeepLens-style noise/ADC in digital sensor units with black level.

        The normalized path treats noise as a [0,1] perturbation. DeepLens works
        in N-bit RAW space: black level is explicit, shot noise scales with
        photo-electrons/digital value, read noise is in digital units, and ISO
        controls gain. This makes low-light/low-bit experiments materially
        different from clean RGB augmentation.
        """
        raw = torch.clamp(raw, 0.0, 1.0)
        bit_depth = self.noise.bit_depth_value().to(device=raw.device, dtype=raw.dtype)
        levels_total = torch.pow(torch.tensor(2.0, device=raw.device, dtype=raw.dtype), bit_depth) - 1.0
        black = torch.as_tensor(self.black_level, device=raw.device, dtype=raw.dtype).clamp(0.0, float(max(0.0, 2 ** 16 - 1)))
        signal_levels = (levels_total - black).clamp_min(1.0)
        raw_nbit = raw * signal_levels + black

        iso = torch.as_tensor(self.iso_value, device=raw.device, dtype=raw.dtype)
        gain = (iso / max(self.iso_base, 1e-6)).clamp_min(1e-6)
        shot_alpha = self.noise.shot_noise_scale_value().to(device=raw.device, dtype=raw.dtype)
        read_std = self.noise.read_noise_std_value().to(device=raw.device, dtype=raw.dtype)
        if perturb and self.training:
            read_rel = self._tol_std("read_noise_rel_std", 0.0)
            shot_rel = self._tol_std("shot_noise_rel_std", 0.0)
            bit_std = self._tol_std("bit_depth_std", 0.0)
            if read_rel > 0.0:
                read_std = read_std * torch.exp(torch.randn((), device=raw.device, dtype=raw.dtype) * read_rel)
            if shot_rel > 0.0:
                shot_alpha = shot_alpha * torch.exp(torch.randn((), device=raw.device, dtype=raw.dtype) * shot_rel)
            if bit_std > 0.0:
                bit_depth = (bit_depth + torch.randn((), device=raw.device, dtype=raw.dtype) * bit_std).clamp(
                    float(self.noise.min_bit_depth), float(self.noise.max_bit_depth)
                )
        signal = (raw_nbit - black).clamp_min(0.0)
        shot_std = shot_alpha * torch.sqrt(signal + 1e-6)
        noise_std = torch.sqrt(shot_std.square() * gain + read_std.square() * gain.square())
        stochastic = bool(self.training or self.eval_stochastic_noise or self.force_stochastic_noise or perturb)
        noisy_nbit = raw_nbit + torch.randn_like(raw_nbit) * noise_std if stochastic else raw_nbit
        noisy_nbit = noisy_nbit.clamp(black, levels_total)
        quant_nbit = self._round_ste(noisy_nbit)
        return ((quant_nbit - black) / signal_levels).clamp(0.0, 1.0)

    def _noise_quant_normalized(self, raw: torch.Tensor, perturb: bool = False) -> torch.Tensor:
        if not (perturb and self.training):
            stochastic = bool(self.training or self.eval_stochastic_noise or self.force_stochastic_noise)
            return self.noise(raw, stochastic=stochastic)
        read_std_org = self.noise.read_noise_std
        shot_scale_org = self.noise.shot_noise_scale
        bit_depth_org = float(self.noise.bit_depth_value().detach().cpu())
        try:
            read_rel = self._tol_std("read_noise_rel_std", 0.0)
            shot_rel = self._tol_std("shot_noise_rel_std", 0.0)
            bit_std = self._tol_std("bit_depth_std", 0.0)
            if read_rel > 0.0:
                self.noise.read_noise_std = read_std_org * float(torch.exp(torch.randn(()) * read_rel))
            if shot_rel > 0.0:
                self.noise.shot_noise_scale = shot_scale_org * float(torch.exp(torch.randn(()) * shot_rel))
            if bit_std > 0.0:
                self.noise.bit_depth = bit_depth_org + float(torch.randn(()) * bit_std)
            return self.noise(raw)
        finally:
            self.noise.read_noise_std = read_std_org
            self.noise.shot_noise_scale = shot_scale_org
            self.noise.bit_depth = bit_depth_org

    # ------------------------------------------------------------------
    # Physics-correct noise/quant helpers — split around CFA mosaicking.
    # Noise is per-channel (pre-CFA); quantization is post-CFA.
    # ------------------------------------------------------------------

    def _precfa_noise_nbit(self, rgb: torch.Tensor, perturb: bool = False) -> torch.Tensor:
        """Shot + read noise in N-bit space applied per channel *before* CFA.

        Each RGB channel is treated as an independent photodiode with its
        own Poisson shot noise and Gaussian read noise.  No quantization
        is performed here — that happens post-CFA in ``_postcfa_quant_nbit``.
        """
        rgb = torch.clamp(rgb, 0.0, 1.0)
        bit_depth = self.noise.bit_depth_value().to(device=rgb.device, dtype=rgb.dtype)
        levels_total = torch.pow(torch.tensor(2.0, device=rgb.device, dtype=rgb.dtype), bit_depth) - 1.0
        black = torch.as_tensor(self.black_level, device=rgb.device, dtype=rgb.dtype).clamp(0.0, 2 ** 16 - 1.0)
        signal_levels = (levels_total - black).clamp_min(1.0)
        rgb_nbit = rgb * signal_levels + black  # (B, 3, H, W) in N-bit space

        iso = torch.as_tensor(self.iso_value, device=rgb.device, dtype=rgb.dtype)
        gain = (iso / max(self.iso_base, 1e-6)).clamp_min(1e-6)
        shot_alpha = self.noise.shot_noise_scale_value().to(device=rgb.device, dtype=rgb.dtype)
        read_std = self.noise.read_noise_std_value().to(device=rgb.device, dtype=rgb.dtype)
        if perturb and self.training:
            read_rel = self._tol_std("read_noise_rel_std", 0.0)
            shot_rel = self._tol_std("shot_noise_rel_std", 0.0)
            if read_rel > 0.0:
                read_std = read_std * torch.exp(torch.randn((), device=rgb.device, dtype=rgb.dtype) * read_rel)
            if shot_rel > 0.0:
                shot_alpha = shot_alpha * torch.exp(torch.randn((), device=rgb.device, dtype=rgb.dtype) * shot_rel)

        signal = (rgb_nbit - black).clamp_min(0.0)
        shot_std = shot_alpha * torch.sqrt(signal + 1e-6)
        noise_std = torch.sqrt(shot_std.square() * gain + read_std.square() * gain.square())
        stochastic = bool(self.training or self.eval_stochastic_noise or self.force_stochastic_noise or perturb)
        noisy_nbit = rgb_nbit + torch.randn_like(rgb_nbit) * noise_std if stochastic else rgb_nbit
        noisy_nbit = noisy_nbit.clamp(black, levels_total)
        return ((noisy_nbit - black) / signal_levels).clamp(0.0, 1.0)

    def _postcfa_quant_nbit(self, raw: torch.Tensor, perturb: bool = False) -> torch.Tensor:
        """ADC quantization only for the N-bit path — noise was applied pre-CFA."""
        raw = torch.clamp(raw, 0.0, 1.0)
        bit_depth = self.noise.bit_depth_value().to(device=raw.device, dtype=raw.dtype)
        if perturb and self.training:
            bit_std = self._tol_std("bit_depth_std", 0.0)
            if bit_std > 0.0:
                bit_depth = (bit_depth + torch.randn((), device=raw.device, dtype=raw.dtype) * bit_std).clamp(
                    float(self.noise.min_bit_depth), float(self.noise.max_bit_depth)
                )
        levels_total = torch.pow(torch.tensor(2.0, device=raw.device, dtype=raw.dtype), bit_depth) - 1.0
        black = torch.as_tensor(self.black_level, device=raw.device, dtype=raw.dtype).clamp(0.0, 2 ** 16 - 1.0)
        signal_levels = (levels_total - black).clamp_min(1.0)
        raw_nbit = (raw * signal_levels + black).clamp(black, levels_total)
        quant_nbit = self._round_ste(raw_nbit)
        return ((quant_nbit - black) / signal_levels).clamp(0.0, 1.0)

    def _apply_precfa_noise(self, rgb: torch.Tensor, perturb: bool = False) -> torch.Tensor:
        """Dispatch pre-CFA per-channel noise to the correct sensor-model path."""
        if self.sensor_model in ("deeplens", "deeplens_nbit", "unprocess_nbit", "raw_nbit"):
            return self._precfa_noise_nbit(rgb, perturb=perturb)
        # Normalized path: temporarily perturb read/shot params if requested.
        if perturb and self.training:
            read_std_org = self.noise.read_noise_std
            shot_scale_org = self.noise.shot_noise_scale
            try:
                read_rel = self._tol_std("read_noise_rel_std", 0.0)
                shot_rel = self._tol_std("shot_noise_rel_std", 0.0)
                if read_rel > 0.0:
                    self.noise.read_noise_std = read_std_org * float(torch.exp(torch.randn(()) * read_rel))
                if shot_rel > 0.0:
                    self.noise.shot_noise_scale = shot_scale_org * float(torch.exp(torch.randn(()) * shot_rel))
                stochastic = True
                return self.noise.forward_noise_only(rgb, stochastic=stochastic)
            finally:
                self.noise.read_noise_std = read_std_org
                self.noise.shot_noise_scale = shot_scale_org
        stochastic = bool(self.training or self.eval_stochastic_noise or self.force_stochastic_noise)
        return self.noise.forward_noise_only(rgb, stochastic=stochastic)

    def _apply_postcfa_quant(self, raw: torch.Tensor, perturb: bool = False) -> torch.Tensor:
        """Dispatch post-CFA ADC quantization to the correct sensor-model path."""
        if self.sensor_model in ("deeplens", "deeplens_nbit", "unprocess_nbit", "raw_nbit"):
            return self._postcfa_quant_nbit(raw, perturb=perturb)
        # Normalized path: temporarily perturb bit_depth if requested.
        if perturb and self.training:
            bit_depth_org = float(self.noise.bit_depth_value().detach().cpu())
            try:
                bit_std = self._tol_std("bit_depth_std", 0.0)
                if bit_std > 0.0:
                    self.noise.bit_depth = bit_depth_org + float(torch.randn(()) * bit_std)
                return self.noise.forward_quant_only(raw)
            finally:
                self.noise.bit_depth = bit_depth_org
        return self.noise.forward_quant_only(raw)

    def forward(self, rgb: torch.Tensor, return_stages: bool = False, perturb: bool = False):
        """
        rgb: (B,3,H,W) in [0,1]
        return: raw OR (raw, stages)
        """
        B, C, H, W = rgb.shape
        rgb_scene_input = rgb
        self._log_stats("in_rgb", rgb)

        stages = {}

        # 0) Approximate RAW scene signal from ISP-processed RGB when requested.
        # This mirrors DeepLens Camera.render(): unprocess first, then perform
        # optics/sensor simulation in RAW-linear space.
        if self.sensor_model in ("deeplens", "deeplens_nbit", "unprocess_nbit", "raw_nbit"):
            rgb = self._unprocess_to_raw_rgb(rgb)
            self._log_stats("after_unprocess", rgb)
            stages["after_unprocess"] = rgb.detach()

        # 1) Optics
        if self._use_optics and self.optics is not None and not self.dbg_bypass_optics:
            rgb = self.optics(rgb, coeff_perturb_std=self._tol_std("optics_coeff_std", 0.0) if perturb else 0.0)
            self._log_stats("after_trainable_optics", rgb)

        elif self._use_optics and self.lens is not None and not self.dbg_bypass_optics:
            self._prepare_lens(H, W)
            lens_cfg = self.cfg.get("lens", {})
            spp     = int(lens_cfg.get("spp", 1))
            rscale  = float(lens_cfg.get("render_scale", 0.75))
            use_amp = bool(lens_cfg.get("use_amp", True))
            no_grad = bool(lens_cfg.get("no_grad", True))

            if 0.0 < rscale < 1.0:
                newH = max(16, int(H * rscale))
                newW = max(16, int(W * rscale))
                rgb_in = torch.nn.functional.interpolate(
                    rgb, size=(newH, newW), mode="bilinear", align_corners=False
                )
            else:
                rgb_in = rgb

            ctx_no_grad = torch.no_grad() if no_grad else torch.enable_grad()
            with ctx_no_grad:
                with _amp_context(enabled=use_amp):
                    try:
                        rgb_out = self.lens.render(rgb_in, depth=self.focus_distance_m, spp=spp)
                    except TypeError:
                        rgb_out = self.lens.render(rgb_in)

            if (rgb_out.shape[-2] != H) or (rgb_out.shape[-1] != W):
                rgb_out = torch.nn.functional.interpolate(
                    rgb_out, size=(H, W), mode="bilinear", align_corners=False
                )

            rgb = torch.clamp(rgb_out, 0.0, 1.0)
            self._log_stats("after_optics", rgb)

            if lens_cfg.get("normalize_output", True):
                target_mean = float(lens_cfg.get("target_mean", 0.5))
                m = rgb.mean().detach()
                eps = 1e-6
                raw_scale = target_mean / max(float(m.item()), eps)
                scale = float(torch.clamp(torch.tensor(raw_scale), 0.1, 1000.0))
                if m.item() <= 1e-5:
                    logging.warning(f"[Optics] near-zero output (mean={m.item():.2e}); force-normalizing with scale {scale:.1f}.")
                rgb = torch.clamp(rgb * scale, 0.0, 1.0)

        stages["after_optics"] = rgb.detach()

        # 2) Exposure (sigmoid -> [0.25, 4.0])
        exposure_logit = self.exposure_logit
        if perturb and self.training:
            exposure_std = self._tol_std("exposure_logit_std", 0.0)
            if exposure_std > 0.0:
                exposure_logit = exposure_logit + torch.randn_like(exposure_logit) * exposure_std
        exposure = 0.25 + 3.75 * torch.sigmoid(exposure_logit)
        rgb = torch.clamp(rgb * exposure, 0.0, 1.0)
        self._log_stats("after_exposure", rgb)
        stages["after_exposure"] = rgb.detach()
        stages["exposure_gain"] = exposure.detach()

        # 3) Pre-CFA per-channel noise — each photodiode accumulates independently.
        # Shot noise and read noise are applied to the 3-channel scene signal before
        # the CFA weighted sum so that noise statistics match a real sensor array.
        if not self.dbg_bypass_noise:
            rgb = self._apply_precfa_noise(rgb, perturb=perturb)
        self._log_stats("after_precfa_noise", rgb)
        stages["after_precfa_noise"] = rgb.detach()

        # 4) CFA mosaic
        if self.dbg_bypass_cfa:
            raw = rgb.mean(dim=1, keepdim=True)
        else:
            cfa_std = self._tol_std("cfa_logit_std", 0.0) if perturb else 0.0
            raw = self.cfa(rgb, weights=self.cfa.effective_weights(perturb_std=cfa_std))
        self._log_stats("after_cfa", raw)
        stages["after_cfa"] = raw.detach()

        # 5) Post-CFA ADC quantization — the ADC sees the single-channel RAW signal.
        if not self.dbg_bypass_noise:
            raw = self._apply_postcfa_quant(raw, perturb=perturb)
        self._log_stats("after_noise", raw)   # key name preserved for backward compat
        stages["after_noise"] = raw.detach()
        if self.raw_output in ("soft_rgb", "demosaic", "demosaiced", "cfa_rgb", "rgb_readout"):
            _dk = (self.cfg.get("sensor") or {}).get("demosaic_kernel_size")
            raw_out = soft_cfa_demosaic(
                raw,
                self.cfa.effective_weights(),
                kernel_size=int(_dk) if _dk is not None else None,
            )
            alpha = float(getattr(self, "camera_blend_alpha", 1.0))
            if alpha < 0.999:
                stages["model_input_pre_blend"] = raw_out.detach()
                raw_out = (1.0 - alpha) * rgb_scene_input + alpha * raw_out
                stages["camera_blend_alpha"] = torch.as_tensor(alpha, device=raw_out.device, dtype=raw_out.dtype)
            stages["model_input"] = raw_out.detach()
        elif self.raw_output in ("packed", "superpixel", "packed_raw", "raw4"):
            raw_out = superpixel_demux(raw)
            stages["model_input"] = raw_out.detach()
        elif self.raw_output in ("rggbi", "packed_iso", "raw4_iso", "rggb_iso"):
            packed = superpixel_demux(raw)
            iso_norm = self.iso_value / max(self.iso_scale, 1e-6)
            iso_channel = torch.full(
                (packed.shape[0], 1, packed.shape[2], packed.shape[3]),
                float(iso_norm),
                device=packed.device,
                dtype=packed.dtype,
            )
            raw_out = torch.cat([packed, iso_channel], dim=1)
            stages["model_input"] = raw_out.detach()
        elif self.raw_output in self.task_token_aliases:
            if self.task_token_readout is None:
                raise RuntimeError("sensor.raw_output='task_tokens' requires TaskTokenReadout to be initialized.")
            raw_out = self.task_token_readout(
                raw,
                self.cfa.effective_weights(),
                iso_norm=self.iso_value / max(self.iso_scale, 1e-6),
                demosaic_kernel_size=int(self.cfg.get("sensor", {}).get("demosaic_kernel_size", 5)),
            )
            stages["task_tokens"] = raw_out.detach()
            stages["model_input"] = raw_out.detach()
        elif self.raw_output in ("mosaic", "raw1", "single"):
            raw_out = raw
            stages["model_input"] = raw.detach()
        else:
            raise ValueError(f"Unsupported sensor.raw_output={self.raw_output!r}")

        return (raw_out, stages) if return_stages else raw_out

    @property
    def output_channels(self) -> int:
        if self.raw_output in self.task_token_aliases:
            if self.task_token_readout is not None:
                return int(self.task_token_readout.token_channels)
            return int((self.cfg.get("sensor", {}) or {}).get("task_token_channels", 8))
        if self.raw_output in ("rggbi", "packed_iso", "raw4_iso", "rggb_iso"):
            return 5
        if self.raw_output in ("packed", "superpixel", "packed_raw", "raw4"):
            return 4
        if self.raw_output in ("soft_rgb", "demosaic", "demosaiced", "cfa_rgb", "rgb_readout"):
            return 3
        return 1

    def exposure_value(self) -> torch.Tensor:
        return 0.25 + 3.75 * torch.sigmoid(self.exposure_logit)

    def regularization(self) -> Dict[str, torch.Tensor]:
        """Differentiable sensor regularizers for manufacturable co-design."""
        device = self.exposure_logit.device
        dtype = self.exposure_logit.dtype
        total = torch.zeros((), device=device, dtype=dtype)
        out: Dict[str, torch.Tensor] = {}

        if self.optics is not None and hasattr(self.optics, "regularization"):
            optics_reg = self.optics.regularization()
            for name, value in optics_reg.items():
                out[f"optics_{name}"] = value
            total = total + optics_reg.get("total", torch.zeros_like(total))

        # Physics-constrained CFA spectral regularization.
        # peak:      -mean(max channel weight) — rewards spectrally peaked filters.
        # diversity: mean pairwise cosine similarity — penalises spectral collapse.
        # smooth:    L2 distance between adjacent tile sites — rewards smooth patterns.
        # entropy:   Shannon entropy — alternative peak constraint (not used by default).
        cfa_spec = self.cfa.spectral_regularization_loss()
        cfa_peak      = cfa_spec["peak"]       # negative value; minimising makes filters more peaked
        cfa_diversity = cfa_spec["diversity"]  # positive value; minimising pushes filters apart
        cfa_smooth    = cfa_spec["smooth"]     # positive value; minimising smooths tile transitions
        w = self.cfa.effective_weights()
        cfa_entropy = -(w * torch.log(w.clamp_min(1e-8))).sum(dim=-1).mean()
        # cfa_collapse kept for backward compat with existing configs
        cfa_collapse = cfa_diversity

        exposure_prior = torch.log(self.exposure_value().clamp_min(1e-6)).square()
        noise = self.noise
        read_prior = torch.log(noise.read_noise_std_value().clamp_min(1e-8) / max(float(self.cfg["sensor"].get("read_noise_std", 0.005)), 1e-8)).square()
        shot_prior = torch.log(noise.shot_noise_scale_value().clamp_min(1e-8) / max(float(self.cfg["sensor"].get("shot_noise_scale", 0.02)), 1e-8)).square()
        bit_prior = (noise.bit_depth_value() - float(self.cfg["sensor"].get("bit_depth", 8))).square() / 64.0

        reg_weights = (self.cfg.get("sensor", {}) or {}).get("regularization", {}) or {}
        out["cfa_peak"]     = cfa_peak
        out["cfa_diversity"] = cfa_diversity
        out["cfa_smooth"]   = cfa_smooth
        out["cfa_entropy"]  = cfa_entropy
        out["cfa_collapse"] = cfa_collapse   # alias for cfa_diversity
        out["exposure_prior"] = exposure_prior
        out["read_noise_prior"] = read_prior
        out["shot_noise_prior"] = shot_prior
        out["bit_depth_prior"] = bit_prior
        # Defaults: peak=0.05 encourages real-filter-like responses without being too aggressive.
        # diversity=0.05 keeps filters spectrally separated.  smooth=0.01 is a light nudge.
        total = total + float(reg_weights.get("cfa_peak",      0.05)) * cfa_peak
        total = total + float(reg_weights.get("cfa_diversity", 0.05)) * cfa_diversity
        total = total + float(reg_weights.get("cfa_smooth",    0.01)) * cfa_smooth
        total = total + float(reg_weights.get("cfa_entropy",   0.0))  * cfa_entropy
        total = total + float(reg_weights.get("cfa_collapse",  0.0))  * cfa_collapse   # off by default (duplicate)
        total = total + float(reg_weights.get("exposure_prior", 0.10)) * exposure_prior
        total = total + float(reg_weights.get("read_noise_prior", 0.05)) * read_prior
        total = total + float(reg_weights.get("shot_noise_prior", 0.05)) * shot_prior
        total = total + float(reg_weights.get("bit_depth_prior", 0.02)) * bit_prior
        if self.task_token_readout is not None:
            token_reg = self.task_token_readout.regularization()
            for name, value in token_reg.items():
                out[name] = value
            total = total + float(reg_weights.get("task_token_collapse", 0.02)) * token_reg["task_token_collapse"]
            total = total + float(reg_weights.get("task_token_residual_energy", 0.01)) * token_reg["task_token_residual_energy"]
            total = total + float(reg_weights.get("task_token_weight_l2", 0.0001)) * token_reg["task_token_weight_l2"]
        out["total"] = total
        return out

    def export_design(self, include_kernels: bool = True) -> Dict[str, Any]:
        """Serializable fixed-at-inference camera configuration."""
        with torch.no_grad():
            cfa_w = self.cfa.effective_weights().detach().cpu()
            cfa_row_sums = cfa_w.sum(dim=-1)
            noise_design = self.noise.export_design()
            design: Dict[str, Any] = {
                "pipeline": "optics_exposure_cfa_noise_quant",
                "sensor_model": self.sensor_model,
                "optics_mode": self._optics_mode,
                "pixel_pitch_m": self.pixel_pitch_m,
                "focus_distance_m": self.focus_distance_m,
                "raw_output": self.raw_output,
                "black_level": self.black_level,
                "iso": self.iso_value,
                "iso_base": self.iso_base,
                "exposure_gain": float(self.exposure_value().detach().cpu()),
                "cfa_weights_rgb": cfa_w.tolist(),
                "trainable_flags": {
                    "optics": any(p.requires_grad for p in self.optics.parameters()) if self.optics is not None else False,
                    "cfa": any(p.requires_grad for p in self.cfa.parameters()),
                    "exposure": bool(self.exposure_logit.requires_grad),
                    "noise": any(p.requires_grad for p in [self.noise.read_noise_raw, self.noise.shot_noise_raw]),
                    "bit_depth": bool(self.noise.bit_depth_raw.requires_grad),
                    "task_token_readout": any(p.requires_grad for p in self.task_token_readout.parameters())
                    if self.task_token_readout is not None
                    else False,
                },
                "noise_quantization": noise_design,
                "deployment_constraints": {
                    "fixed_at_inference": True,
                    "single_global_exposure_gain": True,
                    "nonnegative_unit_sum_cfa_rows": bool(cfa_w.min() >= -1e-7 and torch.allclose(cfa_row_sums, torch.ones_like(cfa_row_sums), atol=1e-5)),
                    "bounded_noise_parameters": True,
                    "bounded_bit_depth": True,
                    "integer_deploy_bit_depth": int(noise_design.get("bit_depth_deploy", self.noise.bit_depth)),
                    "raw_pipeline_before_task_head": self.raw_output != "soft_rgb" or self.sensor_model in ("deeplens", "deeplens_nbit", "unprocess_nbit", "raw_nbit"),
                    "camera_design_exported": True,
                    "simulated_not_hardware_validated": True,
                },
                "constraint_margins": {
                    "min_cfa_weight": float(cfa_w.min()),
                    "max_cfa_weight": float(cfa_w.max()),
                    "max_cfa_row_sum_error": float((cfa_row_sums - 1.0).abs().max()),
                    "cfa_mean_entropy": float((-(cfa_w * torch.log(cfa_w.clamp_min(1e-8))).sum(dim=-1)).mean()),
                    "cfa_temperature": float(getattr(self.cfa, "temperature", 1.0)),
                    "exposure_gain": float(self.exposure_value().detach().cpu()),
                    "bit_depth_continuous": float(self.noise.bit_depth_value().detach().cpu()),
                    "bit_depth_deploy": int(noise_design.get("bit_depth_deploy", self.noise.bit_depth)),
                },
                "tolerance": {
                    "enabled": self._tolerance_enabled(),
                    **{k: v for k, v in self.tolerance_cfg.items() if k != "enabled"},
                },
            }
            if self.task_token_readout is not None:
                design["task_token_readout"] = self.task_token_readout.export_design()
            if self.optics is not None and hasattr(self.optics, "export_design"):
                design["optics"] = self.optics.export_design(include_kernels=include_kernels)
            elif self.lens is not None:
                design["optics"] = {
                    "type": "deeplens_geolens",
                    "path": str((self.cfg.get("lens") or {}).get("path", "")),
                    "trainable": any(p.requires_grad for p in self.lens.parameters())
                    if hasattr(self.lens, "parameters")
                    else False,
                }
            else:
                design["optics"] = {"type": "identity"}
            return design


class PassThroughSensor(nn.Module):
    """RGB baseline frontend with the same call signature as CoDesignSensor."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

    def forward(self, rgb: torch.Tensor, return_stages: bool = False):
        stages = {
            "after_optics": rgb.detach(),
            "after_exposure": rgb.detach(),
            "after_cfa": rgb.detach(),
            "after_noise": rgb.detach(),
            "exposure_gain": torch.ones((), device=rgb.device, dtype=rgb.dtype),
            "rgb_passthrough": True,
        }
        return (rgb, stages) if return_stages else rgb

    def regularization(self) -> Dict[str, torch.Tensor]:
        device = self.cfg.get("device", torch.device("cpu"))
        return {"total": torch.zeros((), device=device)}

    def export_design(self, include_kernels: bool = True) -> Dict[str, Any]:
        return {"pipeline": "processed_rgb_passthrough", "optics_mode": "none"}

    @property
    def output_channels(self) -> int:
        return 3

# -------------------------
# Models
# -------------------------
def build_cls_model(cfg: Dict[str, Any], in_channels: int, num_classes: int) -> nn.Module:
    width = int(cfg["model"].get("width", 64))
    return SmallCNN(in_channels=in_channels, width=width, num_classes=num_classes)

def build_seg_model(cfg: Dict[str, Any], in_channels: int, num_classes: int) -> nn.Module:
    return build_segmentation_model(cfg, in_channels=in_channels, num_classes=num_classes)

# -------------------------
# Dataset
# -------------------------
def get_dataset(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, int]:
    name = cfg["data"]["name"]
    bs = cfg["data"]["batch_size"]
    nw = cfg["data"]["num_workers"]
    img_res = int(cfg["train"].get("img_res", 32))

    def _remap_label_np(mask_np: np.ndarray, label_encoding: str, ignore_index: int) -> np.ndarray:
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]
        enc = (label_encoding or "auto").lower()
        if enc in ("original", "original_id", "id", "cityscapes_id", "cityscapes_labelids", "labelids"):
            return _remap_original_id_to_trainid(mask_np, ignore_index=ignore_index)
        if enc in ("train", "train_id", "trainid", "cityscapes_trainids", "labeltrainids"):
            return _sanitize_trainid(mask_np, ignore_index=ignore_index)
        return _maybe_remap_to_trainid(mask_np, ignore_index=ignore_index)

    def _dataset_mask_paths(ds: Any) -> List[str]:
        if hasattr(ds, "lbl_paths"):
            return list(getattr(ds, "lbl_paths"))
        if hasattr(ds, "rows"):
            return [r["mask_path"] for r in getattr(ds, "rows") if r.get("mask_path")]
        return []

    def _build_rare_sampler(ds: Any, data_cfg: Dict[str, Any], num_classes: int, ignore_index: int):
        rare_ids = [int(x) for x in data_cfg.get("rare_class_ids", [])]
        factor = float(data_cfg.get("rare_oversample_factor", 1.0))
        if not rare_ids or factor <= 1.0:
            return None
        mask_paths = _dataset_mask_paths(ds)
        if len(mask_paths) != len(ds):
            logging.warning("[Sampler] rare-class sampler unavailable for this dataset object.")
            return None
        label_encoding = data_cfg.get("label_encoding", "auto")
        cache_path = data_cfg.get("rare_sampler_cache", "")
        if not cache_path:
            cache_path = os.path.join(
                data_cfg.get("root_seg", data_cfg.get("root", ".")),
                ".raw2task_rare_sampler_cache",
                f"{data_cfg.get('name','seg')}_{len(mask_paths)}_{'_'.join(map(str, rare_ids))}.npy",
            )
        cache_path = os.path.expanduser(cache_path)
        weights_np = None
        if os.path.isfile(cache_path):
            try:
                weights_np = np.load(cache_path)
                if len(weights_np) != len(mask_paths):
                    weights_np = None
            except Exception:
                weights_np = None
        if weights_np is None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            weights = np.ones(len(mask_paths), dtype=np.float64)
            rare_set = set(rare_ids)
            for idx, mp in enumerate(mask_paths):
                try:
                    mask = np.array(Image.open(mp))
                    trainid = _remap_label_np(mask, label_encoding=label_encoding, ignore_index=ignore_index)
                    if "occupancy" in str(data_cfg.get("name", "")).lower():
                        trainid = trainid_to_occupancy_proxy_np(trainid, ignore_index=ignore_index)
                    vals = set(np.unique(trainid).tolist())
                    hits = len(rare_set.intersection(vals))
                    if hits:
                        weights[idx] = factor * (1.0 + 0.15 * (hits - 1))
                except Exception as e:
                    logging.debug(f"[Sampler] failed to scan {mp}: {e}")
            np.save(cache_path, weights)
            weights_np = weights
        names = dense_task_class_names(cfg, num_classes)
        logging.info(
            "[Sampler] rare-class oversampling enabled: "
            f"classes={rare_ids} names={[names[i] for i in rare_ids if 0 <= i < len(names)]} "
            f"factor={factor:.2f} boosted_images={(weights_np > 1.0).sum()}/{len(weights_np)}"
        )
        return WeightedRandomSampler(
            weights=torch.as_tensor(weights_np, dtype=torch.double),
            num_samples=len(weights_np),
            replacement=True,
        )

    if name in ("kitti360_seg", "kitti360_occupancy_proxy", "kitti360_occupancy"):
        root = cfg["data"]["root_seg"]
        img_size = tuple(cfg["data"].get("img_size", [384, 1280]))
        camera = cfg["data"].get("camera", "image_00")
        train_seqs = cfg["data"].get("train_seqs", None)
        val_seqs   = cfg["data"].get("val_seqs", None)
        label_encoding = cfg["data"].get("label_encoding", "original_id")
        train_split_file = cfg["data"].get("train_split_file", None)
        val_split_file = cfg["data"].get("val_split_file", None)
        max_train_samples = cfg["data"].get("max_train_samples", None)
        max_val_samples = cfg["data"].get("max_val_samples", None)
        train_stride = int(cfg["data"].get("train_stride", 1))
        val_stride = int(cfg["data"].get("val_stride", 1))

        tr = KITTI360Seg(
            root=root,
            split="train",
            img_size=img_size,
            camera=camera,
            sequences=train_seqs,
            split_file=train_split_file,
            label_encoding=label_encoding,
            max_samples=max_train_samples,
            stride=train_stride,
            photometric_jitter=float(cfg["data"].get("photometric_jitter", 0.0)),
            viewpoint_jitter=cfg["data"].get("viewpoint_jitter", {}),
        )
        va = KITTI360Seg(
            root=root,
            split="val",
            img_size=img_size,
            camera=camera,
            sequences=val_seqs,
            split_file=val_split_file,
            label_encoding=label_encoding,
            max_samples=max_val_samples,
            stride=val_stride,
            photometric_jitter=float(cfg["data"].get("photometric_jitter", 0.0)),
            viewpoint_jitter={},
        )
        if "occupancy" in name:
            ignore_index = int(cfg.get("model", {}).get("ignore_index", 255))
            tr = OccupancyProxyDataset(tr, ignore_index=ignore_index)
            va = OccupancyProxyDataset(va, ignore_index=ignore_index)
            num_classes = 3
            cfg.setdefault("model", {})["num_classes"] = 3
            cfg["occupancy_proxy"] = occupancy_proxy_metadata()
        else:
            num_classes = int(cfg["model"].get("num_classes", 19))
    elif name == "acdc_seg":
        from raw2task.data.acdc_seg import ACDCSeg
        root       = cfg["data"]["root_seg"]
        img_size   = tuple(cfg["data"].get("img_size", [512, 1024]))
        conditions = cfg["data"].get("conditions", None)
        train_stride = int(cfg["data"].get("train_stride", 1))
        val_stride   = int(cfg["data"].get("val_stride", 1))
        tr = ACDCSeg(
            root=root, split="train", img_size=img_size,
            conditions=conditions, stride=train_stride,
            photometric_jitter=float(cfg["data"].get("photometric_jitter", 0.15)),
        )
        va = ACDCSeg(
            root=root, split="val", img_size=img_size,
            conditions=conditions, stride=val_stride,
        )
        num_classes = int(cfg["model"].get("num_classes", 19))
    elif name in (
        "seg_jsonl",
        "cityscapes_jsonl",
        "acdc_jsonl",
        "worldfactor_jsonl",
        "cityscapes_occupancy_proxy",
        "acdc_occupancy_proxy",
        "worldfactor_occupancy_proxy",
    ):
        root = cfg["data"].get("root", cfg["data"].get("root_seg", ""))
        img_size = tuple(cfg["data"].get("img_size", [512, 1024]))
        label_encoding = cfg["data"].get("label_encoding", "auto")
        train_split_file = cfg["data"].get("train_split_file")
        val_split_file = cfg["data"].get("val_split_file")
        if not train_split_file or not val_split_file:
            raise ValueError(
                "JSONL segmentation datasets require data.train_split_file and data.val_split_file."
            )
        max_train_samples = cfg["data"].get("max_train_samples", None)
        max_val_samples = cfg["data"].get("max_val_samples", None)
        train_stride = int(cfg["data"].get("train_stride", 1))
        val_stride = int(cfg["data"].get("val_stride", 1))
        ignore_index = int(cfg.get("model", {}).get("ignore_index", 255))

        tr = SegmentationJSONLDataset(
            root=root,
            split_file=train_split_file,
            split="train",
            img_size=img_size,
            label_encoding=label_encoding,
            ignore_index=ignore_index,
            max_samples=max_train_samples,
            stride=train_stride,
            photometric_jitter=float(cfg["data"].get("photometric_jitter", 0.0)),
            viewpoint_jitter=cfg["data"].get("viewpoint_jitter", {}),
        )
        va = SegmentationJSONLDataset(
            root=root,
            split_file=val_split_file,
            split="val",
            img_size=img_size,
            label_encoding=label_encoding,
            ignore_index=ignore_index,
            max_samples=max_val_samples,
            stride=val_stride,
            photometric_jitter=float(cfg["data"].get("photometric_jitter", 0.0)),
            viewpoint_jitter={},
        )
        if "occupancy" in name:
            tr = OccupancyProxyDataset(tr, ignore_index=ignore_index)
            va = OccupancyProxyDataset(va, ignore_index=ignore_index)
            num_classes = 3
            cfg.setdefault("model", {})["num_classes"] = 3
            cfg["occupancy_proxy"] = occupancy_proxy_metadata()
        else:
            num_classes = int(cfg["model"].get("num_classes", 19))
    else:
        import torchvision.transforms as T
        from torchvision import datasets

        if img_res != 32:
            resize = T.Resize(img_res)
        else:
            resize = T.Lambda(lambda x: x)
        train_tf = T.Compose([resize, T.RandomHorizontalFlip(), T.ToTensor()])
        val_tf   = T.Compose([resize, T.ToTensor()])

        if name == "cifar10":
            root = cfg["data"]["root"]
            tr = datasets.CIFAR10(root=root, train=True,  download=True, transform=train_tf)
            va = datasets.CIFAR10(root=root, train=False, download=True, transform=val_tf)
            num_classes = 10
        elif name in ("imagenet", "imagenet_local"):
            if name == "imagenet":
                train_root = cfg.get("imagenet_train_dir")
                val_root   = cfg.get("imagenet_val_dir")
            else:
                train_root = cfg.get("imagenet_train_dir_local")
                val_root   = cfg.get("imagenet_val_dir_local")
            if not (train_root and val_root):
                raise ValueError("ImageNet paths missing in config.")
            tr = datasets.ImageFolder(root=train_root, transform=train_tf)
            va = datasets.ImageFolder(root=val_root,   transform=val_tf)
            num_classes = len(tr.classes)
        else:
            raise NotImplementedError(f"Unsupported dataset: {name}")

    ignore_index = int(cfg.get("model", {}).get("ignore_index", 255))
    train_sampler = _build_rare_sampler(tr, cfg["data"], num_classes=num_classes, ignore_index=ignore_index)
    train_loader = DataLoader(
        tr,
        batch_size=bs,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=nw,
        pin_memory=True,
    )
    val_loader   = DataLoader(va, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    print(f"[Data] train N={len(tr)}  val N={len(va)}  "
          f"batch_size={bs}  steps/epoch={len(train_loader)}")

    return train_loader, val_loader, num_classes

# -------------------------
# Evaluation
# -------------------------
@torch.no_grad()
def evaluate_cls(sensor: nn.Module, model: nn.Module, loader: DataLoader, device) -> float:
    sensor.eval(); model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device); labels = labels.to(device)
        raw = sensor(imgs)
        logits = model(raw)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def _forward_with_tta(sensor: nn.Module, model: nn.Module, imgs: torch.Tensor, scales: List[float], use_flip: bool):
    base_size = imgs.shape[-2:]
    logits_sum = None
    count = 0
    stages_for_viz = None

    for scale in scales:
        if abs(scale - 1.0) < 1e-6:
            imgs_s = imgs
        else:
            imgs_s = F.interpolate(imgs, scale_factor=scale, mode="bilinear", align_corners=False)

        raw_s, stages_s = sensor(imgs_s, return_stages=True)
        if stages_for_viz is None:
            stages_for_viz = stages_s
        logits_s = model(raw_s)
        if logits_s.shape[-2:] != base_size:
            logits_s = F.interpolate(logits_s, size=base_size, mode="bilinear", align_corners=False)
        logits_sum = logits_s if logits_sum is None else (logits_sum + logits_s)
        count += 1

        if use_flip:
            imgs_sf = torch.flip(imgs_s, dims=[3])
            raw_sf = sensor(imgs_sf)
            logits_sf = model(raw_sf)
            logits_sf = torch.flip(logits_sf, dims=[3])
            if logits_sf.shape[-2:] != base_size:
                logits_sf = F.interpolate(logits_sf, size=base_size, mode="bilinear", align_corners=False)
            logits_sum = logits_sf if logits_sum is None else (logits_sum + logits_sf)
            count += 1

    if count <= 0:
        raw, stages = sensor(imgs, return_stages=True)
        return model(raw), stages

    return logits_sum / float(count), stages_for_viz

@torch.no_grad()
def evaluate_seg(sensor: nn.Module, model: nn.Module, loader: DataLoader, device,
                 num_classes: int, out_dir: str, epoch: int, max_batches: int | None = None,
                 write_outputs: bool = True, split_name: str = "val") -> Dict[str, Any]:
    """
    Returns dict with pixel_acc, mIoU, per_class_IoU, and writes:
      - PNG grids in {out_dir}/viz/epoch{epoch}_stepXXXX.png
      - confusion matrix .npy in {out_dir}/confusion_epoch{epoch}.npy
    """
    sensor.eval(); model.eval()
    ignore_index = int(getattr(model, "ignore_index", 255))

    self_cfg = getattr(sensor, "cfg", None)
    use_crf   = bool(self_cfg.get("eval", {}).get("use_crf", False)) if self_cfg else False
    crf_iters = int(self_cfg.get("eval", {}).get("crf_iters", 5))     if self_cfg else 5
    crf_sxy   = int(self_cfg.get("eval", {}).get("crf_sxy", 3))       if self_cfg else 3
    crf_srgb  = int(self_cfg.get("eval", {}).get("crf_srgb", 5))      if self_cfg else 5
    crf_comp  = int(self_cfg.get("eval", {}).get("crf_compat", 3))    if self_cfg else 3
    use_tta   = bool(self_cfg.get("eval", {}).get("tta", False))      if self_cfg else False
    tta_scales = self_cfg.get("eval", {}).get("tta_scales", [1.0])     if self_cfg else [1.0]
    tta_flip  = bool(self_cfg.get("eval", {}).get("tta_flip", True))   if self_cfg else True
    if isinstance(tta_scales, (int, float)):
        tta_scales = [float(tta_scales)]
    else:
        tta_scales = [float(s) for s in tta_scales]

    total_correct, total_labeled = 0, 0
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    viz_every = int(self_cfg.get("eval", {}).get("viz_every", 50)) if self_cfg else 50
    eval_log_interval = int(self_cfg.get("eval", {}).get("log_interval", 0)) if self_cfg else 0
    if eval_log_interval <= 0:
        eval_log_interval = max(1, len(loader) // 10) if hasattr(loader, "__len__") else 25
    palette = dense_task_palette(self_cfg or {}, num_classes)
    class_names = dense_task_class_names(self_cfg or {}, num_classes)
    viz_dir = os.path.join(out_dir, "viz")
    if write_outputs:
        os.makedirs(viz_dir, exist_ok=True)

    eval_start_time = time.time()
    total_batches = len(loader) if hasattr(loader, "__len__") else 0
    if max_batches is not None and max_batches > 0 and total_batches:
        total_batches = min(total_batches, int(max_batches))
    for it, (imgs, labels) in enumerate(loader, 1):
        if max_batches is not None and max_batches > 0 and it > int(max_batches):
            break
        imgs = imgs.to(device); labels = labels.to(device)
        if use_tta:
            logits, stages = _forward_with_tta(sensor, model, imgs, tta_scales, tta_flip)
        else:
            raw, stages = sensor(imgs, return_stages=True)
            logits = model(raw)
            logits = align_logits_to_labels(logits, labels)

        if use_crf:
            imgs_u8 = (imgs.detach().cpu().permute(0,2,3,1).numpy() * 255).astype(np.uint8)
            probs   = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32)
            pred_list = []
            for b in range(probs.shape[0]):
                Q = refine_with_crf(
                    imgs_u8[b], probs[b],
                    sxy=crf_sxy, srgb=crf_srgb, compat=crf_comp, iters=crf_iters
                )
                pred_list.append(Q.argmax(0))
            pred = torch.from_numpy(np.stack(pred_list, 0)).to(labels.device)
        else:
            pred = logits.argmax(dim=1)

        if ignore_index is not None:
            mask = (labels != ignore_index)
            correct = (pred[mask] == labels[mask]).sum().item()
            labeled = mask.sum().item()
        else:
            correct = (pred == labels).sum().item()
            labeled = labels.numel()
        total_correct += correct
        total_labeled += max(labeled, 1)

        update_confusion(conf, pred, labels, num_classes=num_classes, ignore_index=ignore_index)

        is_last = bool(total_batches and it == total_batches)
        if it == 1 or it % eval_log_interval == 0 or is_last:
            elapsed = time.time() - eval_start_time
            steps_left = max(0, total_batches - it) if total_batches else 0
            eta_eval = elapsed / max(1, it) * steps_left
            running_miou, running_iou = iou_from_confusion(conf)
            running_acc = total_correct / max(total_labeled, 1)
            low_iou = float(np.nanmin(running_iou)) if len(running_iou) else 0.0
            print(
                f"EVAL[{split_name}] progress {_progress_bar(it, total_batches or it)} "
                f"batch {it}/{total_batches or '?'} ETA {_format_seconds(eta_eval)} "
                f"pixel_acc {running_acc:.4f} running_mIoU {running_miou:.4f} "
                f"min_class_IoU {low_iou:.4f}",
                flush=True,
            )

        if write_outputs and (it == 1 or (viz_every > 0 and it % viz_every == 0)):
            prefix = "" if split_name in ("", "val") else f"{split_name}_"
            out_png = os.path.join(viz_dir, f"{prefix}epoch{epoch}_step{it:05d}.png")
            vis_batch_with_stages(
                rgb=imgs, pred=pred, target=labels, stages=stages,
                out_png=out_png, palette=palette,
                ignore_index=ignore_index, max_items=4,
                panel_caption=(
                    f"{split_name.title()} epoch {epoch}, step {it}: representative frames with "
                    "learned-camera diagnostics, ground truth, and prediction."
                ),
            )

    pixel_acc = total_correct / max(total_labeled, 1)
    miou, per_class_iou = iou_from_confusion(conf)
    class_support = conf.sum(axis=1).astype(np.int64)
    class_pred = conf.sum(axis=0).astype(np.int64)
    class_union = (conf.sum(axis=1) + conf.sum(axis=0) - np.diag(conf)).astype(np.int64)

    if write_outputs:
        prefix = "" if split_name in ("", "val") else f"{split_name}_"
        np.save(os.path.join(out_dir, f"{prefix}confusion_epoch{epoch}.npy"), conf)

    return {
        "pixel_acc": float(pixel_acc),
        "mIoU": float(miou),
        "per_class_IoU": [float(x) for x in per_class_iou],
        "per_class_support": [int(x) for x in class_support.tolist()],
        "per_class_pred": [int(x) for x in class_pred.tolist()],
        "per_class_union": [int(x) for x in class_union.tolist()],
        "class_names": class_names,
    }

# -------------------------
# Config loader
# -------------------------
def load_config() -> Dict[str, Any]:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()

    with open(os.path.expanduser(args.config), "r") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("seed", random.randint(0, 100))
    set_seed(cfg["seed"])
    cfg.setdefault("task", "cls")

    if torch.cuda.is_available():
        cfg["device"] = torch.device("cuda")
        cfg["num_gpus"] = torch.cuda.device_count()
        logging.info(f"Using {cfg['num_gpus']} GPU(s): {torch.cuda.get_device_name(0)}")
    else:
        cfg["device"] = torch.device("cpu")
        cfg["num_gpus"] = 0
        logging.info("Using CPU")

    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    except Exception:
        pass

    t = cfg.setdefault("train", {})
    t["epochs"] = int(t.get("epochs", 10))
    t["lr"] = float(t.get("lr", 3e-4))
    t["weight_decay"] = float(t.get("weight_decay", 1e-4))
    t["sensor_lr_mult"] = float(t.get("sensor_lr_mult", 1.0))
    t["log_interval"] = int(t.get("log_interval", 100))
    t["depth"] = float(t.get("depth", 2.0))
    t["ckpt_dir"] = t.get("ckpt_dir", "./checkpoints")
    t.setdefault("resume", False)
    t.setdefault("resume_path", "")
    t.setdefault("load_strict", True)
    t.setdefault("keep_last_n", 3)
    t.setdefault("amp", True)
    t.setdefault("accum_steps", 1)
    t.setdefault("grad_clip_norm", 0.0)
    t.setdefault("early_stop_patience", 0)
    t.setdefault("early_stop_min_delta", 0.0005)
    t.setdefault("lr_plateau_patience", 3)
    t.setdefault("lr_reduce_factor", 0.4)
    t.setdefault("lr_plateau_cooldown", 0)
    t.setdefault("min_lr", 3e-6)
    t.setdefault("ce_weight", 1.0)
    t.setdefault("overlap_weight", 0.5)
    t.setdefault("overlap_loss", "dice")
    t.setdefault("use_lovasz", False)
    t.setdefault("ohem_pct", 1.0)
    t.setdefault("smooth_lambda", 0.0)
    t.setdefault("smooth_sigma_rgb", 0.10)
    t.setdefault("task_feedback_weight", 0.0)
    t.setdefault("task_feedback_entropy_weight", 0.5)
    t.setdefault("task_feedback_error_weight", 1.0)
    t.setdefault("task_feedback_boundary_weight", 1.0)
    t.setdefault("task_feedback_rare_weight", 1.0)
    t.setdefault("task_feedback_detach", True)
    t.setdefault("task_feedback_controller", True)
    t.setdefault("task_feedback_controller_ema", 0.70)
    t.setdefault("task_feedback_iou_gamma", 1.50)
    t.setdefault("task_feedback_min_pressure", 0.75)
    t.setdefault("task_feedback_max_pressure", 2.50)
    t.setdefault("task_feedback_min_support", 1000)
    t.setdefault("distill_weight", 0.0)
    t.setdefault("distill_temperature", 2.0)
    t.setdefault("distill_confidence_threshold", 0.0)
    t.setdefault("distill_hard_weight", 0.0)
    t.setdefault("distill_boundary_weight", 0.0)
    t.setdefault("semantic_snr_weight", 0.0)
    t.setdefault("semantic_snr_nonboundary_weight", 0.25)
    t.setdefault("semantic_snr_start_epoch", 1)
    t.setdefault("deployable_task_weight", 0.0)
    t.setdefault("deployable_task_start_epoch", 0)
    t.setdefault("deployable_task_apply_to_model", False)
    t.setdefault("fixed_sensor_advantage_weight", 0.0)
    t.setdefault("fixed_sensor_advantage_margin", 0.0)
    t.setdefault("fixed_sensor_advantage_start_epoch", 0)
    t.setdefault("fixed_sensor_advantage_apply_to_model", False)
    t.setdefault("sensor_rollback_on_val_drop", False)
    t.setdefault("sensor_rollback_val_tolerance", 0.006)
    t.setdefault("sensor_rollback_shrink", 0.5)
    t.setdefault("sensor_freeze_after_best_patience", 0)
    t.setdefault("sensor_freeze_restore_best", True)
    t.setdefault("freeze_backbone_during_sensor_stage", False)
    t.setdefault("aux_losses_update_adapter", True)
    t.setdefault("sensor_reg_weight", 0.0)
    t.setdefault("sensor_lr_component_scales", {})
    t.setdefault("sensor_trust_region", {})
    t.setdefault("deploy_guard", {})
    t.setdefault("train_deploy_probe_batches", 0)
    t.setdefault("tolerance_consistency_weight", 0.0)
    t.setdefault("tolerance_task_weight", 0.0)
    t.setdefault("export_design_every", 1)
    t.setdefault("dice_weight", 0.5)   # backward-compatible alias
    t.setdefault("img_res", 32)
    t.setdefault("torch_compile", False)
    t.setdefault("channels_last", False)
    t.setdefault("progress_bar", True)
    t.setdefault("ema", False)
    t.setdefault("ema_decay", 0.999)
    t.setdefault("ema_start_epoch", 1)
    t.setdefault("checkpoint_average_k", 0)
    t.setdefault("checkpoint_average_source", "best")
    t.setdefault("checkpoint_average_use_ema", True)

    d = cfg.setdefault("data", {})
    d["name"] = (d.get("name", "cifar10") or "").lower()
    d["root"] = d.get("root", "./data")
    d.setdefault("root_seg", d.get("root", "./data"))
    d["batch_size"] = int(d.get("batch_size", 256))
    d["num_workers"] = int(d.get("num_workers", 4))
    d.setdefault("label_encoding", "original_id")
    d.setdefault("train_stride", 1)
    d.setdefault("val_stride", 1)
    d.setdefault("rare_class_ids", [])
    d.setdefault("rare_oversample_factor", 1.0)
    d.setdefault("photometric_jitter", 0.0)
    d.setdefault("viewpoint_jitter", {})

    s = cfg.setdefault("sensor", {})
    s.setdefault("sensor_model", "normalized")
    s["cfa_init"] = s.get("cfa_init", "bayer_rggb")
    s.setdefault("cfa_init_floor", 1e-4)
    s.setdefault("cfa_temperature_start", s.get("cfa_temperature", 1.0))
    s.setdefault("cfa_temperature_end", 1.0)
    s.setdefault("cfa_temperature_epochs", 0)
    s.setdefault("cfa_min_transmittance", 0.0)
    s["exposure_init"] = float(s.get("exposure_init", 0.0))
    s.setdefault("learn_cfa", True)
    s.setdefault("learn_exposure", True)
    s["bit_depth"] = int(s.get("bit_depth", 6))
    s["read_noise_std"] = float(s.get("read_noise_std", 0.005))
    s["shot_noise_scale"] = float(s.get("shot_noise_scale", 0.02))
    s.setdefault("learn_noise", True)
    s.setdefault("learn_bit_depth", True)
    s.setdefault("min_read_noise_std", 1e-5)
    s.setdefault("max_read_noise_std", 0.05)
    s.setdefault("min_shot_noise_scale", 1e-5)
    s.setdefault("max_shot_noise_scale", 0.12)
    s.setdefault("min_bit_depth", 4.0)
    s.setdefault("max_bit_depth", 12.0)
    s["sensor_pitch"] = float(s.get("sensor_pitch", 1.4e-6))
    s.setdefault("black_level", 0.0)
    s.setdefault("eval_stochastic_noise", False)
    s.setdefault("iso_base", 100.0)
    s.setdefault("iso", s.get("iso_value", s.get("iso_base", 100.0)))
    s.setdefault("iso_scale", 1000.0)
    s.setdefault("bayer_pattern", "rggb")
    s.setdefault("tolerance", {})
    s["tolerance"].setdefault("enabled", False)

    lens_block = cfg.get("lens")
    if not isinstance(lens_block, dict):
        cfg["lens"] = {}
    cfg["lens"].setdefault("path", "")
    cfg["lens"].setdefault("mode", "geolens" if str(cfg["lens"].get("path", "") or "").strip() else "identity")
    cfg["lens"].setdefault("learn_optics", True)
    cfg["lens"].setdefault("spp", 1)
    cfg["lens"].setdefault("render_scale", 0.75)
    cfg["lens"].setdefault("no_grad", True)
    cfg["lens"].setdefault("use_amp", True)
    cfg["lens"].setdefault("normalize_output", True)
    cfg["lens"].setdefault("target_mean", 0.5)
    cfg["lens"].setdefault("trainable_psf", {})
    cfg["lens"]["trainable_psf"].setdefault("num_zones", 5)
    cfg["lens"]["trainable_psf"].setdefault("kernel_size", 15)
    cfg["lens"]["trainable_psf"].setdefault("base_sigma_px", 1.05)
    cfg["lens"]["trainable_psf"].setdefault("min_sigma_px", 0.35)
    cfg["lens"]["trainable_psf"].setdefault("max_sigma_px", 3.75)
    cfg["lens"]["trainable_psf"].setdefault("max_shift_px", 1.25)
    cfg["lens"]["trainable_psf"].setdefault("field_sigma", 0.65)

    m = cfg.setdefault("model", {})
    m.setdefault("name", "unet_tiny")
    m["width"] = int(m.get("width", 64))
    m.setdefault("ignore_index", 255)
    m.setdefault("num_classes", 19)

    distill = cfg.setdefault("distillation", {})
    distill.setdefault("enabled", float(t.get("distill_weight", 0.0)) > 0.0)
    distill.setdefault("checkpoint", "")
    distill.setdefault("strict", False)
    distill.setdefault(
        "teacher_model",
        {
            "name": "segformer_b1",
            "pretrained": True,
            "hf_model_id": "nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
        },
    )

    p = cfg.setdefault("pipeline", {})
    p.setdefault("input", "sensor")
    p["input"] = str(p.get("input", "sensor")).lower()

    isp = cfg.setdefault("task_isp", {})
    isp.setdefault("enabled", False)
    isp.setdefault("type", "task_aware_modulation")
    isp.setdefault("hidden_channels", 24)
    isp.setdefault("residual_scale", 0.20)
    isp.setdefault("regional_grid", 8)
    isp.setdefault("clamp_output", True)

    cfg.setdefault("debug", {})
    cfg["debug"].setdefault("bypass_optics", False)
    cfg["debug"].setdefault("bypass_cfa", False)
    cfg["debug"].setdefault("bypass_noise", False)
    cfg["debug"].setdefault("dump_stats", False)

    e = cfg.setdefault("eval", {})
    e.setdefault("viz_every", 50)
    e.setdefault("train_viz_steps", [1])
    e.setdefault("use_crf", False)
    e.setdefault("crf_iters", 5)
    e.setdefault("crf_sxy", 3)
    e.setdefault("crf_srgb", 5)
    e.setdefault("crf_compat", 3)
    e.setdefault("tta", False)
    e.setdefault("tta_scales", [1.0])
    e.setdefault("tta_flip", True)

    return cfg

# -------------------------
# Training loop
# -------------------------
def train(cfg: Dict[str, Any]):
    import json, csv, glob, re
    train_wall_start = time.time()

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if "seed" in cfg:
        set_seed(int(cfg["seed"]))
    if not isinstance(cfg.get("device", None), torch.device):
        cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg["num_gpus"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = cfg["device"]
    task = cfg.get("task", "cls")
    tcfg = cfg["train"]

    train_loader, val_loader, num_classes = get_dataset(cfg)

    input_mode = str(cfg.get("pipeline", {}).get("input", "sensor")).lower()
    if input_mode in ("rgb", "rgb_baseline", "processed_rgb"):
        sensor = PassThroughSensor(cfg).to(device)
        in_ch = sensor.output_channels
        logging.info("[Pipeline] RGB baseline: processed RGB goes directly to the segmentation model.")
    elif input_mode in ("sensor", "camera_sim", "simulated_raw", "codesign"):
        sensor = CoDesignSensor(cfg).to(device)
        in_ch = sensor.output_channels
        logging.info(
            "[Pipeline] Camera-simulation frontend: "
            f"optics/exposure/CFA/noise/quantization -> model ({in_ch} input channels)."
        )
    else:
        raise ValueError(f"Unsupported pipeline.input={input_mode!r}")

    use_amp        = bool(tcfg.get("amp", True))
    accum_steps    = int(tcfg.get("accum_steps", 1))
    grad_clip_norm = float(tcfg.get("grad_clip_norm", 0.0))
    early_stop_pat = int(tcfg.get("early_stop_patience", 0))
    early_stop_min_delta = float(tcfg.get("early_stop_min_delta", 0.0005))
    ce_loss_weight = float(tcfg.get("ce_weight", 1.0))
    overlap_weight = float(tcfg.get("overlap_weight", tcfg.get("dice_weight", 0.5)))
    use_lovasz = bool(tcfg.get("use_lovasz", False)) or str(tcfg.get("overlap_loss", "")).lower() == "lovasz"
    ohem_pct = float(tcfg.get("ohem_pct", 1.0))
    smooth_lambda = float(tcfg.get("smooth_lambda", 0.0))
    smooth_sigma_rgb = float(tcfg.get("smooth_sigma_rgb", 0.10))
    task_feedback_weight = float(tcfg.get("task_feedback_weight", 0.0))
    task_feedback_entropy_weight = float(tcfg.get("task_feedback_entropy_weight", 0.5))
    task_feedback_error_weight = float(tcfg.get("task_feedback_error_weight", 1.0))
    task_feedback_boundary_weight = float(tcfg.get("task_feedback_boundary_weight", 1.0))
    task_feedback_rare_weight = float(tcfg.get("task_feedback_rare_weight", 1.0))
    task_feedback_detach = bool(tcfg.get("task_feedback_detach", True))
    distill_weight = float(tcfg.get("distill_weight", 0.0))
    distill_temperature = float(tcfg.get("distill_temperature", 2.0))
    distill_confidence_threshold = float(tcfg.get("distill_confidence_threshold", 0.0))
    distill_hard_weight = float(tcfg.get("distill_hard_weight", 0.0))
    distill_boundary_weight = float(tcfg.get("distill_boundary_weight", 0.0))
    semantic_snr_weight = float(tcfg.get("semantic_snr_weight", 0.0))
    semantic_snr_nonboundary_weight = float(tcfg.get("semantic_snr_nonboundary_weight", 0.25))
    semantic_snr_start_epoch = int(tcfg.get("semantic_snr_start_epoch", 1))
    deployable_task_weight = float(tcfg.get("deployable_task_weight", 0.0))
    deployable_task_start_epoch = int(tcfg.get("deployable_task_start_epoch", 0))
    deployable_task_apply_to_model = bool(tcfg.get("deployable_task_apply_to_model", False))
    fixed_sensor_advantage_weight = float(tcfg.get("fixed_sensor_advantage_weight", 0.0))
    fixed_sensor_advantage_margin = float(tcfg.get("fixed_sensor_advantage_margin", 0.0))
    fixed_sensor_advantage_start_epoch = int(tcfg.get("fixed_sensor_advantage_start_epoch", deployable_task_start_epoch))
    fixed_sensor_advantage_apply_to_model = bool(tcfg.get("fixed_sensor_advantage_apply_to_model", False))
    aux_losses_update_adapter = bool(tcfg.get("aux_losses_update_adapter", True))
    freeze_backbone_during_sensor_stage = bool(tcfg.get("freeze_backbone_during_sensor_stage", False))
    backbone_freeze_epochs = int(tcfg.get("backbone_freeze_epochs", 0))
    sensor_freeze_after_best_patience = int(tcfg.get("sensor_freeze_after_best_patience", 0))
    sensor_freeze_restore_best = bool(tcfg.get("sensor_freeze_restore_best", True))
    sensor_reg_weight = float(tcfg.get("sensor_reg_weight", 0.0))
    sensor_warmup_epochs = int(tcfg.get("sensor_warmup_epochs", 0))
    # Alternating optimization (Successive Optimization, arXiv 2412.14603):
    # alternate network-only and sensor-only optimizer steps within the joint phase.
    alternating_opt_enabled = bool(tcfg.get("alternating_opt_enabled", False))
    alternating_net_steps = int(tcfg.get("alternating_net_steps", 10))
    alternating_sensor_steps = int(tcfg.get("alternating_sensor_steps", 1))
    # PSF proximity regularization (Tolerance-Aware Deep Optics, arXiv 2502.04719):
    # penalize large PSF deviations from epoch-start snapshot to prevent overshoot.
    psf_proximity_weight = float(tcfg.get("psf_proximity_weight", 0.0))
    tolerance_consistency_weight = float(tcfg.get("tolerance_consistency_weight", 0.0))
    tolerance_task_weight = float(tcfg.get("tolerance_task_weight", 0.0))
    export_design_every = int(tcfg.get("export_design_every", 1))
    use_ema = bool(tcfg.get("ema", False))
    ema_decay = float(tcfg.get("ema_decay", 0.999))
    ema_start_epoch = int(tcfg.get("ema_start_epoch", 1))

    is_dense_task = is_dense_prediction_task(task)
    task_palette = dense_task_palette(cfg, num_classes)
    task_class_names = dense_task_class_names(cfg, num_classes)
    feedback_controller = None
    teacher_model = None

    if is_dense_task:
        backbone_in_ch = in_ch
        task_isp_adapter = build_task_isp_adapter(cfg, channels=in_ch)
        if task_isp_adapter is not None:
            backbone_in_ch = in_ch
            logging.info(
                "[TaskISP] enabled: type=%s hidden=%s residual_scale=%.3f regional_grid=%s",
                str(cfg.get("task_isp", {}).get("type", "task_aware_modulation")),
                str(cfg.get("task_isp", {}).get("hidden_channels", 24)),
                float(cfg.get("task_isp", {}).get("residual_scale", 0.20)),
                str(cfg.get("task_isp", {}).get("regional_grid", 8)),
            )
        model = build_seg_model(cfg, in_channels=backbone_in_ch, num_classes=num_classes).to(device)
        if task_isp_adapter is not None:
            model = PerceptionWithTaskISP(task_isp_adapter.to(device), model).to(device)
        ignore_index = int(cfg.get("model", {}).get("ignore_index", 255))
        model.ignore_index = ignore_index
        loss_cfg = cfg.get("loss", {}) or {}
        feedback_rare_class_ids = cfg.get("data", {}).get("rare_class_ids")
        if not feedback_rare_class_ids:
            feedback_rare_class_ids = list(_parse_class_boosts(loss_cfg.get("rare_class_boost", {})).keys())
        class_weights = compute_class_weights(
            train_loader,
            num_classes=num_classes,
            ignore_index=ignore_index,
            max_batches=int(loss_cfg.get("class_weight_batches", 300)),
            max_weight=float(loss_cfg.get("class_weight_max", 25.0)),
            min_weight=float(loss_cfg.get("class_weight_min", 0.05)),
            rare_class_boost=_parse_class_boosts(loss_cfg.get("rare_class_boost", {})),
        )
        logging.info(f"[Loss] class weights : {class_weights.tolist()}")
        ce_weight = class_weights.to(device)
        feedback_controller = TaskFeedbackController(num_classes, task_class_names, cfg, device)
        teacher_model = build_distillation_teacher(cfg, num_classes, device) if distill_weight > 0.0 else None
    else:
        task_isp_adapter = build_task_isp_adapter(cfg, channels=in_ch)
        model = build_cls_model(cfg, in_channels=in_ch, num_classes=num_classes).to(device)
        if task_isp_adapter is not None:
            model = PerceptionWithTaskISP(task_isp_adapter.to(device), model).to(device)
            logging.info("[TaskISP] enabled for classification pipeline.")
        loss_fn = nn.CrossEntropyLoss()

    base_lr   = float(tcfg["lr"])
    wd        = float(tcfg["weight_decay"])
    sensor_lr = base_lr * float(tcfg["sensor_lr_mult"])
    sensor_lr_component_scales = {
        str(k): float(v)
        for k, v in (tcfg.get("sensor_lr_component_scales", {}) or {}).items()
    }
    camera_curriculum_epochs = int(tcfg.get("camera_curriculum_epochs", 0))

    if bool(tcfg.get("channels_last", False)):
        sensor = sensor.to(memory_format=torch.channels_last)
        model = model.to(memory_format=torch.channels_last)

    if bool(tcfg.get("torch_compile", False)) and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            logging.info("[Compile] torch.compile enabled for segmentation model.")
        except Exception as e:
            logging.warning(f"[Compile] torch.compile failed; continuing eager. Error: {e}")

    model_param_requires_grad = {id(p): bool(p.requires_grad) for p in model.parameters()}

    ema_model = None
    if use_ema:
        if bool(tcfg.get("torch_compile", False)):
            logging.warning("[EMA] Disabled because torch_compile is enabled (state dict mismatch risk).")
            use_ema = False
        else:
            ema_model = _build_ema_model(model)
            logging.info(f"[EMA] enabled: decay={ema_decay}, start_epoch={ema_start_epoch}")

    param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": base_lr, "name": "model"}]
    sensor_param_requires_grad = {id(p): bool(p.requires_grad) for p in sensor.parameters()}
    sensor_group_indices: Dict[str, int] = {}
    sensor_base_lrs: Dict[str, float] = {}
    sensor_component_params: Dict[str, List[nn.Parameter]] = {comp: [] for comp in SensorTrustRegionController.COMPONENTS}
    for name, p in sensor.named_parameters():
        if not p.requires_grad:
            continue
        comp = SensorTrustRegionController._component(name)
        sensor_component_params.setdefault(comp, []).append(p)
    for comp in SensorTrustRegionController.COMPONENTS:
        params = sensor_component_params.get(comp, [])
        if not params:
            continue
        comp_lr = sensor_lr * float(sensor_lr_component_scales.get(comp, 1.0))
        sensor_group_indices[comp] = len(param_groups)
        sensor_base_lrs[comp] = comp_lr
        param_groups.append(
            {
                "params": params,
                "lr": comp_lr,
                "name": f"sensor_{comp}",
                "component": comp,
            }
        )
    if sensor_group_indices:
        logging.info(
            "[Optim] sensor parameter groups: %s",
            ", ".join(
                f"{comp}={len(sensor_component_params[comp])}@{sensor_base_lrs[comp]:.2e}"
                for comp in sensor_group_indices
            ),
        )
    optim = torch.optim.AdamW(param_groups, weight_decay=wd)
    sensor_trust = SensorTrustRegionController(
        sensor=sensor,
        cfg=cfg,
        sensor_group_indices=sensor_group_indices,
        base_sensor_lrs=sensor_base_lrs,
        ckpt_dir=tcfg["ckpt_dir"],
    )
    deploy_guard_cfg = tcfg.get("deploy_guard", {}) or {}
    deploy_guard_enabled = bool(deploy_guard_cfg.get("enabled", False)) and bool(sensor_group_indices)
    deploy_guard_start_alpha = float(deploy_guard_cfg.get("start_alpha", 0.999))
    deploy_guard_batch_ratio = float(deploy_guard_cfg.get("batch_miou_ratio", 0.45))
    deploy_guard_val_drop = float(deploy_guard_cfg.get("val_drop_tolerance", 0.008))
    deploy_guard_probe_drop = float(deploy_guard_cfg.get("probe_drop_tolerance", 0.08))
    deploy_guard_patience = int(deploy_guard_cfg.get("patience", 2))
    deploy_guard_shrink = float(deploy_guard_cfg.get("shrink_factor", 0.35))
    deploy_guard_min_lr_mult = float(deploy_guard_cfg.get("min_lr_mult", 0.01))
    deploy_guard_freeze_after = int(deploy_guard_cfg.get("freeze_after_actions", 3))
    deploy_guard_collapse_count = 0
    deploy_guard_actions = 0
    deploy_guard_best_probe = -float("inf")
    sensor_rollback_enabled = bool(tcfg.get("sensor_rollback_on_val_drop", False)) and bool(sensor_group_indices)
    sensor_rollback_val_tolerance = float(tcfg.get("sensor_rollback_val_tolerance", 0.006))
    sensor_rollback_shrink = float(tcfg.get("sensor_rollback_shrink", 0.5))
    best_sensor_state: Dict[str, torch.Tensor] | None = None
    sensor_frozen_by_plateau = False
    fixed_reference_sensor = None
    if fixed_sensor_advantage_weight > 0.0 and input_mode in ("sensor", "camera_sim", "simulated_raw", "codesign") and bool(sensor_group_indices):
        fixed_reference_sensor = copy.deepcopy(sensor).to(device)
        fixed_reference_sensor.eval()
        if hasattr(fixed_reference_sensor, "set_camera_blend_alpha"):
            fixed_reference_sensor.set_camera_blend_alpha(1.0)
        for p in fixed_reference_sensor.parameters():
            p.requires_grad_(False)
        logging.info(
            "[FixedSensorAdvantage] enabled: weight=%.4f margin=%.4f start_epoch=%d apply_to_model=%s",
            fixed_sensor_advantage_weight,
            fixed_sensor_advantage_margin,
            fixed_sensor_advantage_start_epoch,
            fixed_sensor_advantage_apply_to_model,
        )
    if deploy_guard_enabled:
        logging.info(
            "[DeployGuard] enabled: start_alpha=%.3f batch_ratio=%.3f val_drop=%.4f probe_drop=%.4f "
            "patience=%d shrink=%.3f min_lr_mult=%.3f freeze_after=%d",
            deploy_guard_start_alpha,
            deploy_guard_batch_ratio,
            deploy_guard_val_drop,
            deploy_guard_probe_drop,
            deploy_guard_patience,
            deploy_guard_shrink,
            deploy_guard_min_lr_mult,
            deploy_guard_freeze_after,
        )
    if sensor_rollback_enabled:
        logging.info(
            "[SensorRollback] enabled: val_tolerance=%.4f shrink=%.3f",
            sensor_rollback_val_tolerance,
            sensor_rollback_shrink,
        )
    if sensor_freeze_after_best_patience > 0 and sensor_group_indices:
        logging.info(
            "[SensorPlateauFreeze] enabled: patience=%d restore_best=%s",
            sensor_freeze_after_best_patience,
            sensor_freeze_restore_best,
        )
    if backbone_freeze_epochs > 0 and hasattr(model, "backbone_parameters"):
        logging.info(
            "[StageSchedule] backbone_freeze_epochs=%d: backbone frozen ep%d–%d (Task-Driven Lens Design), joint fine-tune after.",
            backbone_freeze_epochs, sensor_warmup_epochs + 1, backbone_freeze_epochs,
        )
    elif freeze_backbone_during_sensor_stage and hasattr(model, "backbone_parameters"):
        logging.info("[StageSchedule] backbone frozen during active sensor co-design stage; task ISP remains trainable.")
    if alternating_opt_enabled:
        logging.info(
            "[AltOpt] alternating optimization enabled: %d net-steps then %d sensor-steps per period (joint phase only).",
            alternating_net_steps, alternating_sensor_steps,
        )
    if psf_proximity_weight > 0.0:
        logging.info("[PSFProximity] PSF proximity regularization enabled: weight=%.4f", psf_proximity_weight)

    scaler = _grad_scaler(enabled=use_amp)

    _lr_schedule = str(tcfg.get("lr_schedule", "plateau")).lower()
    if _lr_schedule == "poly":
        _total_epochs = int(tcfg.get("epochs", 40))
        _power        = float(tcfg.get("poly_power", 0.9))
        _min_lr_ratio = float(tcfg.get("min_lr", 1e-6)) / max(
            group["lr"] for group in optim.param_groups
        )
        def _poly_lambda(ep, _T=_total_epochs, _p=_power, _min=_min_lr_ratio):
            return max(_min, (1.0 - ep / _T) ** _p)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=_poly_lambda)
        logging.info(f"[LR] Polynomial decay: power={_power}, total={_total_epochs} epochs, min_lr_ratio={_min_lr_ratio:.2e}")
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            mode="max",
            factor=float(tcfg.get("lr_reduce_factor", 0.4)),
            patience=int(tcfg.get("lr_plateau_patience", 3)),
            threshold=1e-4,
            threshold_mode="rel",
            cooldown=int(tcfg.get("lr_plateau_cooldown", 0)),
            min_lr=float(tcfg.get("min_lr", 3e-6)),
        )

    log_interval = int(tcfg["log_interval"])
    ckpt_dir = tcfg["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    def dense_task_loss_from_logits(logits_x: torch.Tensor, labels_x: torch.Tensor) -> torch.Tensor:
        ce_x = ohem_cross_entropy(
            logits_x,
            labels_x,
            weight=ce_weight,
            ignore_index=ignore_index,
            keep_frac=ohem_pct,
        )
        if use_lovasz:
            overlap_x = lovasz_softmax_loss(logits_x, labels_x, ignore_index=ignore_index)
        else:
            overlap_x = dice_loss(logits_x, labels_x, ignore_index=ignore_index)
        return ce_loss_weight * ce_x + overlap_weight * overlap_x

    # Efficiency measurement (once)
    try:
        if measure_model_efficiency is not None:
            if is_dense_task:
                img_size = tuple(cfg["data"].get("img_size", [384, 1280]))
                C_in, H, W = 3, img_size[0], img_size[1]
            else:
                C_in, H, W = 3, int(tcfg.get("img_res", 32)), int(tcfg.get("img_res", 32))

            eff_sensor = measure_model_efficiency(sensor, (C_in, H, W), device, batch_size=1, notes="sensor_only")
            save_efficiency(eff_sensor, ckpt_dir, stem="efficiency_sensor")

            with torch.no_grad():
                dummy_rgb = torch.randn(1, 3, H, W, device=device)
                dummy_raw = sensor(dummy_rgb)
                _, _, Hr, Wr = dummy_raw.shape
            # Use the actual number of channels produced by the sensor when measuring the model
            C_model_in = int(dummy_raw.shape[1])
            eff_model = measure_model_efficiency(model, (C_model_in, Hr, Wr), device, batch_size=1, notes="model_only")
            save_efficiency(eff_model, ckpt_dir, stem="efficiency_model")

            if Chain is not None:
                chain = Chain(sensor, model)
                eff_chain = measure_model_efficiency(chain, (C_in, H, W), device, batch_size=1, notes="sensor_plus_model")
                save_efficiency(eff_chain, ckpt_dir, stem="efficiency_chain")
                logging.info(
                    f"[Eff] chain latency: {eff_chain.latency_ms:.3f} ms; "
                    f"params(sensor)={eff_sensor.params_m:.3f}M, params(model)={eff_model.params_m:.3f}M"
                )
        else:
            logging.info("[Eff] efficiency utils not available; skipping measurement.")
    except Exception as e:
        logging.warning(f"[Eff] Measurement failed (continuing): {e}")

    logging.info("==> Start training (DeepLens optics + learnable sensor + task).")

    start_epoch = 1
    step = 0
    best = -float("inf")
    last_val_score = None
    last_improve_epoch = 0

    if bool(tcfg.get("resume", False)):
        resume_path = tcfg.get("resume_path") or _find_default_resume_path(ckpt_dir)
        if resume_path and os.path.isfile(resume_path):
            ckpt = _load_ckpt(resume_path, device)
            strict = bool(tcfg.get("load_strict", True))
            try:
                sensor.load_state_dict(ckpt["sensor"], strict=strict)
                model.load_state_dict(ckpt["model"], strict=strict)
                if use_ema and ema_model is not None and ckpt.get("ema_model") is not None:
                    try:
                        ema_model.load_state_dict(ckpt["ema_model"], strict=False)
                    except Exception as e:
                        logging.warning(f"[Resume] Failed to load EMA state; reinitializing EMA from model. Error: {e}")
                        ema_model = _build_ema_model(model)
                elif use_ema and ema_model is not None:
                    # Backward-compatible resume from checkpoints without explicit EMA state.
                    ema_model = _build_ema_model(model)
                if ckpt.get("optim"):     optim.load_state_dict(ckpt["optim"])
                if ckpt.get("scaler"):
                    try: scaler.load_state_dict(ckpt["scaler"])
                    except Exception: logging.warning("[Resume] AMP scaler state incompatible; continuing.")
                if ckpt.get("scheduler"):
                    try: scheduler.load_state_dict(ckpt["scheduler"])
                    except Exception: logging.warning("[Resume] Scheduler state incompatible; continuing.")
                if feedback_controller is not None and ckpt.get("feedback_controller") is not None:
                    feedback_controller.load_state_dict(ckpt.get("feedback_controller"), device)
                start_epoch = int(ckpt.get("epoch", 0)) + 1
                step = int(ckpt.get("step", 0))
                best = float(ckpt.get("best", -float("inf")))
                last_improve_epoch = start_epoch - 1
                logging.info(f"[Resume] Loaded {resume_path} (start_epoch={start_epoch}, best={best:.3f})")
            except Exception as e:
                logging.warning(f"[Resume] Failed to load checkpoint '{resume_path}': {e}")
        else:
            logging.info("[Resume] No valid resume checkpoint found; starting fresh.")
    elif is_dense_task:
        try:
            _export_sensor_design(
                sensor,
                os.path.join(ckpt_dir, "camera_design_initial.json"),
                include_kernels=True,
            )
            _save_sensor_preview(
                sensor,
                os.path.join(ckpt_dir, "design_preview"),
                stem="initial",
            )
        except Exception as e:
            logging.warning(f"[DesignExport] initial export failed: {e}")

    keep_last_n = int(tcfg.get("keep_last_n", 3))

    optim.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, int(tcfg["epochs"]) + 1):
        epoch_start_time = time.time()
        sensor.train(); model.train()
        camera_alpha = 1.0
        if camera_curriculum_epochs > 0:
            ramp_start = max(0, sensor_warmup_epochs)
            camera_alpha = max(0.0, min(1.0, (epoch - ramp_start) / float(max(1, camera_curriculum_epochs))))
        if hasattr(sensor, "set_camera_blend_alpha"):
            sensor.set_camera_blend_alpha(camera_alpha)
            if epoch == start_epoch or (camera_curriculum_epochs > 0 and epoch <= sensor_warmup_epochs + camera_curriculum_epochs + 1):
                logging.info(f"[CameraCurriculum] camera blend alpha={camera_alpha:.3f} at epoch {epoch}.")
        if hasattr(sensor, "set_cfa_temperature_for_epoch"):
            sensor.set_cfa_temperature_for_epoch(epoch)
            if hasattr(sensor, "cfa") and (epoch == start_epoch or epoch <= max(1, int(getattr(sensor, "cfa_temperature_epochs", 0))) + 1):
                logging.info(f"[CFA] temperature={float(getattr(sensor.cfa, 'temperature', 1.0)):.3f} at epoch {epoch}.")
        sensor_active = bool(sensor_group_indices) and (epoch > sensor_warmup_epochs) and not sensor_frozen_by_plateau
        if sensor_warmup_epochs > 0 or sensor_frozen_by_plateau:
            for p in sensor.parameters():
                p.requires_grad_(sensor_active and sensor_param_requires_grad.get(id(p), False))
            if epoch == start_epoch or epoch == sensor_warmup_epochs + 1 or sensor_frozen_by_plateau:
                state = "unfrozen" if sensor_active else "frozen"
                logging.info(f"[SensorWarmup] sensor parameters are {state} at epoch {epoch}.")

        # backbone_freeze_epochs > 0: Task-Driven Lens Design approach — freeze backbone for
        # the full first phase (sensor_warmup_epochs+1 .. backbone_freeze_epochs), then joint.
        # backbone_freeze_epochs == 0: fall back to old freeze_backbone_during_sensor_stage
        # behaviour (backbone frozen for the entire sensor-active stage).
        if backbone_freeze_epochs > 0:
            backbone_frozen_for_stage = (
                hasattr(model, "backbone_parameters")
                and hasattr(model, "adapter_parameters")
                and sensor_active
                and epoch <= backbone_freeze_epochs
            )
        else:
            backbone_frozen_for_stage = (
                freeze_backbone_during_sensor_stage
                and hasattr(model, "backbone_parameters")
                and hasattr(model, "adapter_parameters")
                and sensor_active
            )
        if hasattr(model, "backbone_parameters") and hasattr(model, "adapter_parameters"):
            _set_requires_grad(model.adapter_parameters(), True, model_param_requires_grad)
            _set_requires_grad(model.backbone_parameters(), not backbone_frozen_for_stage, model_param_requires_grad)
            if epoch == start_epoch or epoch == sensor_warmup_epochs + 1 or epoch == backbone_freeze_epochs + 1 or (sensor_frozen_by_plateau and epoch == last_improve_epoch + sensor_freeze_after_best_patience):
                logging.info(
                    "[StageSchedule] epoch=%d sensor_active=%s backbone=%s task_isp=trainable",
                    epoch,
                    sensor_active,
                    "frozen" if backbone_frozen_for_stage else "trainable",
                )
        epoch_seen = 0
        epoch_loss_sum = 0.0
        epoch_ce_sum = 0.0
        epoch_overlap_sum = 0.0
        epoch_smooth_sum = 0.0
        epoch_feedback_sum = 0.0
        epoch_distill_sum = 0.0
        epoch_semantic_snr_sum = 0.0
        epoch_deploy_sum = 0.0
        epoch_fixed_advantage_sum = 0.0
        epoch_sensor_reg_sum = 0.0
        # PSF proximity: snapshot optics params at epoch start (Tolerance-Aware Deep Optics).
        psf_ref_snapshot: list = []
        if psf_proximity_weight > 0.0 and sensor_active:
            _optics_params = sensor_component_params.get("optics", [])
            psf_ref_snapshot = [p.detach().clone() for p in _optics_params]

        # Whether alternating opt applies this epoch (joint phase: sensor on, backbone not epoch-frozen).
        _altopt_joint = (
            alternating_opt_enabled
            and sensor_active
            and not backbone_frozen_for_stage
            and hasattr(model, "backbone_parameters")
            and hasattr(model, "adapter_parameters")
        )
        _altopt_period = alternating_net_steps + alternating_sensor_steps

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            disable=(not bool(tcfg.get("progress_bar", True)) or not sys.stderr.isatty()),
        )
        for it, (imgs, labels) in enumerate(pbar, 1):
            imgs = imgs.to(device); labels = labels.to(device)
            if bool(tcfg.get("channels_last", False)):
                imgs = imgs.contiguous(memory_format=torch.channels_last)

            # Alternating optimization toggle (Successive Optimization, arXiv 2412.14603).
            # During joint phase: alternate net-only and sensor-only optimizer steps so neither
            # component can silently absorb the other's gradient signal.
            if _altopt_joint:
                _eff_step = (it - 1) // accum_steps   # optimizer-step index within epoch
                _sensor_turn = (_eff_step % _altopt_period) >= alternating_net_steps
                # sensor turn: backbone frozen; network turn: sensor frozen
                _set_requires_grad(model.backbone_parameters(), not _sensor_turn, model_param_requires_grad)
                for _p in sensor.parameters():
                    _p.requires_grad_(sensor_param_requires_grad.get(id(_p), False) and _sensor_turn)

            with _amp_context(enabled=use_amp):
                raw = sensor(imgs)
                logits = model(raw)
                logits = align_logits_to_labels(logits, labels)
                ce_for_log = logits.new_tensor(0.0)
                overlap_for_log = logits.new_tensor(0.0)
                smooth_for_log = logits.new_tensor(0.0)
                feedback_for_log = logits.new_tensor(0.0)
                distill_for_log = logits.new_tensor(0.0)
                semantic_snr_for_log = logits.new_tensor(0.0)
                deploy_for_log = logits.new_tensor(0.0)
                fixed_advantage_for_log = logits.new_tensor(0.0)
                sensor_reg_for_log = logits.new_tensor(0.0)

                if is_dense_task:
                    ce = ohem_cross_entropy(
                        logits,
                        labels,
                        weight=ce_weight,
                        ignore_index=ignore_index,
                        keep_frac=ohem_pct,
                    )
                    ce_for_log = ce
                    if use_lovasz:
                        overlap = lovasz_softmax_loss(logits, labels, ignore_index=ignore_index)
                    else:
                        overlap = dice_loss(logits, labels, ignore_index=ignore_index)
                    overlap_for_log = overlap
                    smooth = logits.new_tensor(0.0)
                    if smooth_lambda > 0.0:
                        smooth = edge_aware_smoothness_loss(
                            logits,
                            imgs,
                            ignore_index=ignore_index,
                            targets=labels,
                            sigma_rgb=smooth_sigma_rgb,
                        )
                    smooth_for_log = smooth
                    total_loss = ce_loss_weight * ce + overlap_weight * overlap + smooth_lambda * smooth
                    if task_feedback_weight > 0.0:
                        feedback_loss, feedback_stats = task_feedback_loss(
                            logits,
                            labels,
                            class_weight=ce_weight,
                            class_pressure=feedback_controller.current_pressure() if feedback_controller is not None else None,
                            ignore_index=ignore_index,
                            rare_class_ids=feedback_rare_class_ids,
                            entropy_weight=task_feedback_entropy_weight,
                            error_weight=task_feedback_error_weight,
                            boundary_weight=task_feedback_boundary_weight,
                            rare_weight=task_feedback_rare_weight,
                            detach_feedback=task_feedback_detach,
                        )
                        feedback_for_log = feedback_loss
                        total_loss = total_loss + task_feedback_weight * feedback_loss
                    if teacher_model is not None and distill_weight > 0.0:
                        with torch.no_grad():
                            teacher_logits = teacher_model(imgs)
                            teacher_logits = align_logits_to_labels(teacher_logits, labels)
                        distill_loss = distillation_kl_loss(
                            logits,
                            teacher_logits,
                            labels,
                            ignore_index=ignore_index,
                            temperature=distill_temperature,
                            confidence_threshold=distill_confidence_threshold,
                        )
                        distill_for_log = distill_loss
                        total_loss = total_loss + distill_weight * distill_loss
                        if distill_hard_weight > 0.0:
                            hard_teacher_loss, _hard_teacher_stats = teacher_guided_pseudo_loss(
                                logits,
                                teacher_logits,
                                labels,
                                ignore_index=ignore_index,
                                confidence_threshold=distill_confidence_threshold,
                                boundary_weight=distill_boundary_weight,
                            )
                            total_loss = total_loss + distill_hard_weight * hard_teacher_loss
                            distill_for_log = distill_for_log + hard_teacher_loss.detach()
                    if (
                        semantic_snr_weight > 0.0
                        and epoch >= max(1, semantic_snr_start_epoch)
                        and any(p.requires_grad for p in sensor.parameters())
                    ):
                        semantic_snr_loss_value, _semantic_snr_stats = semantic_sensor_snr_loss(
                            raw,
                            labels,
                            ignore_index=ignore_index,
                            nonboundary_weight=semantic_snr_nonboundary_weight,
                        )
                        semantic_snr_for_log = semantic_snr_loss_value
                        total_loss = total_loss + semantic_snr_weight * semantic_snr_loss_value
                    if (
                        getattr(sensor, "_tolerance_enabled", lambda: False)()
                        and (tolerance_consistency_weight > 0.0 or tolerance_task_weight > 0.0)
                    ):
                        raw_pert = sensor(imgs, perturb=True)
                        logits_pert = model(raw_pert)
                        logits_pert = align_logits_to_labels(logits_pert, labels)
                        if tolerance_task_weight > 0.0:
                            ce_pert = ohem_cross_entropy(
                                logits_pert,
                                labels,
                                weight=ce_weight,
                                ignore_index=ignore_index,
                                keep_frac=ohem_pct,
                            )
                            total_loss = total_loss + tolerance_task_weight * ce_pert
                        if tolerance_consistency_weight > 0.0:
                            log_p = F.log_softmax(logits_pert, dim=1)
                            q = F.softmax(logits.detach(), dim=1)
                            if ignore_index is not None:
                                valid = (labels != ignore_index).unsqueeze(1)
                                kl_map = F.kl_div(log_p, q, reduction="none").sum(dim=1, keepdim=True)
                                consistency = (kl_map * valid).sum() / valid.sum().clamp_min(1)
                            else:
                                consistency = F.kl_div(log_p, q, reduction="batchmean")
                            total_loss = total_loss + tolerance_consistency_weight * consistency
                    if (
                        deployable_task_weight > 0.0
                        and hasattr(sensor, "set_camera_blend_alpha")
                        and epoch >= max(1, deployable_task_start_epoch)
                        and any(p.requires_grad for p in sensor.parameters())
                    ):
                        # Optimize the camera against the same fixed path used
                        # at inference.  By default this auxiliary branch
                        # updates only optics/sensor parameters; the main
                        # curriculum branch still trains the task model.
                        with _temporary_camera_alpha(sensor, 1.0):
                            raw_deploy = sensor(imgs)
                        if deployable_task_apply_to_model:
                            logits_deploy = model(raw_deploy)
                        else:
                            with _temporary_model_trainability(model, allow_adapter=aux_losses_update_adapter):
                                logits_deploy = model(raw_deploy)
                        logits_deploy = align_logits_to_labels(logits_deploy, labels)
                        deploy_ce = ohem_cross_entropy(
                            logits_deploy,
                            labels,
                            weight=ce_weight,
                            ignore_index=ignore_index,
                            keep_frac=ohem_pct,
                        )
                        if use_lovasz:
                            deploy_overlap = lovasz_softmax_loss(logits_deploy, labels, ignore_index=ignore_index)
                        else:
                            deploy_overlap = dice_loss(logits_deploy, labels, ignore_index=ignore_index)
                        deploy_loss = ce_loss_weight * deploy_ce + overlap_weight * deploy_overlap
                        deploy_for_log = deploy_loss
                        total_loss = total_loss + deployable_task_weight * deploy_loss
                    if (
                        fixed_reference_sensor is not None
                        and fixed_sensor_advantage_weight > 0.0
                        and epoch >= max(1, fixed_sensor_advantage_start_epoch)
                        and any(p.requires_grad for p in sensor.parameters())
                    ):
                        # Same-batch comparison against the frozen deployable
                        # reference camera.  The learned camera is penalized
                        # only when its task loss is worse than the fixed
                        # design by more than the configured margin.
                        with _temporary_camera_alpha(sensor, 1.0):
                            raw_adv = sensor(imgs)
                        model_was_training = model.training
                        model.eval()
                        try:
                            if fixed_sensor_advantage_apply_to_model:
                                logits_adv = model(raw_adv)
                            else:
                                with _temporary_model_trainability(model, allow_adapter=aux_losses_update_adapter):
                                    logits_adv = model(raw_adv)
                            logits_adv = align_logits_to_labels(logits_adv, labels)
                            current_sensor_loss = dense_task_loss_from_logits(logits_adv, labels)
                            with torch.no_grad():
                                if hasattr(fixed_reference_sensor, "set_camera_blend_alpha"):
                                    fixed_reference_sensor.set_camera_blend_alpha(1.0)
                                raw_fixed = fixed_reference_sensor(imgs)
                                logits_fixed = model(raw_fixed)
                                logits_fixed = align_logits_to_labels(logits_fixed, labels)
                                fixed_sensor_loss = dense_task_loss_from_logits(logits_fixed, labels)
                        finally:
                            model.train(model_was_training)
                        fixed_advantage_loss = F.relu(
                            current_sensor_loss - fixed_sensor_loss.detach() + fixed_sensor_advantage_margin
                        )
                        fixed_advantage_for_log = fixed_advantage_loss
                        total_loss = total_loss + fixed_sensor_advantage_weight * fixed_advantage_loss
                    sensor_reg = logits.new_tensor(0.0)
                    if sensor_reg_weight > 0.0 and hasattr(sensor, "regularization") and (epoch > sensor_warmup_epochs):
                        sensor_reg = sensor.regularization().get("total", sensor_reg)
                        sensor_reg_for_log = sensor_reg
                        total_loss = total_loss + sensor_reg_weight * sensor_reg
                    # PSF proximity: penalise deviation from epoch-start PSF snapshot.
                    # Prevents large per-step PSF jumps (Tolerance-Aware Deep Optics, arXiv 2502.04719).
                    if psf_proximity_weight > 0.0 and psf_ref_snapshot and sensor_active:
                        _optics_now = sensor_component_params.get("optics", [])
                        if len(_optics_now) == len(psf_ref_snapshot):
                            _prox_loss = sum(
                                (p - r.to(p.device)).pow(2).sum()
                                for p, r in zip(_optics_now, psf_ref_snapshot)
                            )
                            total_loss = total_loss + psf_proximity_weight * _prox_loss
                    loss_for_log = total_loss.detach()
                else:
                    total_loss = loss_fn(logits, labels)
                    if sensor_reg_weight > 0.0 and hasattr(sensor, "regularization") and (epoch > sensor_warmup_epochs):
                        sensor_reg_for_log = sensor.regularization().get("total", total_loss.new_tensor(0.0))
                        total_loss = total_loss + sensor_reg_weight * sensor_reg_for_log
                    loss_for_log = total_loss.detach()

                total_loss = total_loss * (1.0 / max(1, accum_steps))

            batch_n = int(imgs.size(0))
            epoch_seen += batch_n
            epoch_loss_sum += float(loss_for_log.item()) * batch_n
            epoch_ce_sum += float(ce_for_log.detach().item()) * batch_n
            epoch_overlap_sum += float(overlap_for_log.detach().item()) * batch_n
            epoch_smooth_sum += float(smooth_for_log.detach().item()) * batch_n
            epoch_feedback_sum += float(feedback_for_log.detach().item()) * batch_n
            epoch_distill_sum += float(distill_for_log.detach().item()) * batch_n
            epoch_semantic_snr_sum += float(semantic_snr_for_log.detach().item()) * batch_n
            epoch_deploy_sum += float(deploy_for_log.detach().item()) * batch_n
            epoch_fixed_advantage_sum += float(fixed_advantage_for_log.detach().item()) * batch_n
            epoch_sensor_reg_sum += float(sensor_reg_for_log.detach().item()) * batch_n

            scaler.scale(total_loss).backward()
            step += 1

            if (step % accum_steps) == 0:
                if not torch.isfinite(loss_for_log).item():
                    logging.warning(f"[Train] non-finite loss at epoch {epoch}, step {step}; skipping step.")
                    optim.zero_grad(set_to_none=True)
                    continue

                sensor_state_before_step = sensor_trust.capture_state(sensor)
                did_unscale = False
                if grad_clip_norm > 0.0 or sensor_trust.enabled:
                    scaler.unscale_(optim)
                    did_unscale = True
                if sensor_trust.enabled:
                    sensor_trust.grad_stats(sensor)
                if grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(sensor.parameters()),
                        grad_clip_norm
                    )

                scaler.step(optim)
                scaler.update()
                if sensor_trust.enabled:
                    sensor_trust.after_step(
                        optim=optim,
                        sensor=sensor,
                        model=model,
                        imgs=imgs.detach(),
                        labels=labels.detach(),
                        before_state=sensor_state_before_step,
                        epoch=epoch,
                        global_step=step,
                        camera_alpha=camera_alpha,
                        ignore_index=ignore_index if is_dense_task else -100,
                        class_weight=ce_weight if is_dense_task else None,
                    )
                optim.zero_grad(set_to_none=True)
                # Restore epoch-level requires_grad after alternating opt step.
                if _altopt_joint:
                    _set_requires_grad(model.adapter_parameters(), True, model_param_requires_grad)
                    _set_requires_grad(model.backbone_parameters(), True, model_param_requires_grad)
                    for _p in sensor.parameters():
                        _p.requires_grad_(sensor_param_requires_grad.get(id(_p), False))
                if use_ema and ema_model is not None and epoch >= ema_start_epoch:
                    _ema_update(ema_model, model, ema_decay)

            if is_dense_task:
                train_viz_steps = cfg.get("eval", {}).get("train_viz_steps", [1])
                if it in train_viz_steps:
                    with torch.no_grad():
                        pred_train = logits.argmax(1)
                        _raw_dbg, stages_dbg = sensor(imgs, return_stages=True)
                        out_png = os.path.join(ckpt_dir, "viz", f"train_epoch{epoch}_step{it:04d}.png")
                        os.makedirs(os.path.dirname(out_png), exist_ok=True)
                        vis_batch_with_stages(
                            rgb=imgs, pred=pred_train, target=labels, stages=stages_dbg,
                            out_png=out_png, palette=task_palette,
                            ignore_index=ignore_index, max_items=4,
                            panel_caption=(
                                f"Training epoch {epoch}, step {it}: "
                                + (
                                    "RGB baseline pass-through is shown beside labels and predictions."
                                    if input_mode in ("rgb", "rgb_baseline", "processed_rgb")
                                    else "camera-stage diagnostics are shown beside labels and predictions."
                                )
                            ),
                        )

            if it % log_interval == 0:
                with torch.no_grad():
                    if is_dense_task:
                        pred = logits.argmax(1)
                        if ignore_index is not None:
                            mask = (labels != ignore_index)
                            acc = (pred[mask] == labels[mask]).float().mean().item() if mask.any() else 0.0
                        else:
                            acc = (pred == labels).float().mean().item()
                        batch_miou = imagewise_miou(
                            pred.detach(),
                            labels.detach(),
                            num_classes=num_classes,
                            ignore_index=ignore_index,
                        )
                        if (
                            deploy_guard_enabled
                            and camera_alpha >= deploy_guard_start_alpha
                            and last_val_score is not None
                            and float(last_val_score) > 0.20
                        ):
                            collapse_threshold = float(last_val_score) * deploy_guard_batch_ratio
                            if batch_miou < collapse_threshold:
                                deploy_guard_collapse_count += 1
                            else:
                                deploy_guard_collapse_count = max(0, deploy_guard_collapse_count - 1)
                            if deploy_guard_collapse_count >= deploy_guard_patience:
                                deploy_guard_actions += 1
                                factor = 0.0 if (deploy_guard_freeze_after > 0 and deploy_guard_actions >= deploy_guard_freeze_after) else deploy_guard_shrink
                                scales = sensor_trust.scale_sensor_lrs(
                                    optim,
                                    factor,
                                    min_lr_mult=0.0 if factor == 0.0 else deploy_guard_min_lr_mult,
                                )
                                action = "freeze_sensor_lrs" if factor == 0.0 else "shrink_sensor_lrs"
                                logging.warning(
                                    "[DeployGuard] %s at epoch=%d step=%d: batch_mIoU=%.4f below %.4f "
                                    "(prev_val=%.4f, count=%d, actions=%d, scales=%s)",
                                    action,
                                    epoch,
                                    step,
                                    batch_miou,
                                    collapse_threshold,
                                    float(last_val_score),
                                    deploy_guard_collapse_count,
                                    deploy_guard_actions,
                                    {k: round(float(v), 4) for k, v in scales.items()},
                                )
                                print(
                                    f"[DeployGuard] {action}: batch_mIoU {batch_miou:.4f} < "
                                    f"{collapse_threshold:.4f}; sensor_lr_scales="
                                    f"{ {k: round(float(v), 4) for k, v in scales.items()} }",
                                    flush=True,
                                )
                                deploy_guard_collapse_count = 0
                    else:
                        acc = (logits.argmax(1) == labels).float().mean().item()
                        batch_miou = 0.0
                curr_lr = [group['lr'] for group in optim.param_groups][0]
                elapsed = time.time() - epoch_start_time
                steps_left = max(0, len(train_loader) - it)
                eta_epoch = elapsed / max(1, it) * steps_left
                epochs_left = max(0, int(tcfg["epochs"]) - epoch)
                eta_total = eta_epoch + epochs_left * (elapsed / max(1, it) * max(1, len(train_loader)))
                postfix_dict = {
                    "loss": f"{loss_for_log.item():.4f}",
                    "acc": f"{acc:.3f}",
                    "bIoU": f"{batch_miou:.3f}",
                    "val_mIoU": "n/a" if last_val_score is None else f"{last_val_score:.4f}",
                    "best": "n/a" if not np.isfinite(best) else f"{best:.4f}",
                    "lr": f"{curr_lr:.2e}"
                }
                if is_dense_task:
                    postfix_dict["ce"] = f"{ce_for_log.item():.4f}"
                    postfix_dict["ovlp"] = f"{overlap_for_log.item():.4f}"
                    if task_feedback_weight > 0.0:
                        postfix_dict["fb"] = f"{feedback_for_log.item():.4f}"
                    if teacher_model is not None and distill_weight > 0.0:
                        postfix_dict["kd"] = f"{distill_for_log.item():.4f}"
                    if semantic_snr_weight > 0.0:
                        postfix_dict["semSNR"] = f"{semantic_snr_for_log.item():.4f}"
                    if deployable_task_weight > 0.0:
                        postfix_dict["deploy"] = f"{deploy_for_log.item():.4f}"
                    if fixed_sensor_advantage_weight > 0.0:
                        postfix_dict["fixAdv"] = f"{fixed_advantage_for_log.item():.4f}"
                pbar.set_postfix(postfix_dict)
                progress_msg = (
                    f"TRAIN epoch {epoch}/{int(tcfg['epochs'])} "
                    f"{_progress_bar(it, len(train_loader))} "
                    f"step {it}/{len(train_loader)} "
                    f"ETA_epoch {_format_seconds(eta_epoch)} "
                    f"ETA_total {_format_seconds(eta_total)} "
                    f"loss {loss_for_log.item():.4f} "
                    f"acc {acc:.3f} "
                )
                if is_dense_task:
                    progress_msg += (
                        f"batch_mIoU {batch_miou:.4f} "
                        f"prev_val_mIoU {'n/a' if last_val_score is None else f'{last_val_score:.4f}'} "
                        f"best_mIoU {'n/a' if not np.isfinite(best) else f'{best:.4f}'} "
                    )
                    if task_feedback_weight > 0.0:
                        progress_msg += f"feedback {feedback_for_log.item():.4f} "
                    if teacher_model is not None and distill_weight > 0.0:
                        progress_msg += f"distill {distill_for_log.item():.4f} "
                    if semantic_snr_weight > 0.0:
                        progress_msg += f"semSNR {semantic_snr_for_log.item():.4f} "
                    if deployable_task_weight > 0.0:
                        progress_msg += f"deploy {deploy_for_log.item():.4f} "
                    if fixed_sensor_advantage_weight > 0.0:
                        progress_msg += f"fixedAdv {fixed_advantage_for_log.item():.4f} "
                progress_msg += f"lr {curr_lr:.2e}"
                print(progress_msg, flush=True)
                logging.info(
                    f"Epoch {epoch} | {it}/{len(train_loader)} | "
                    f"loss {loss_for_log.item():.4f} | acc {acc:.3f} | "
                    f"batch_mIoU {batch_miou:.4f} | "
                    f"lr {curr_lr:.2e}"
                )

        # ---- Eval ----
        eval_model = model
        if use_ema and ema_model is not None and epoch >= ema_start_epoch:
            eval_model = ema_model

        if is_dense_task:
            if hasattr(sensor, "set_camera_blend_alpha"):
                # Validation is always the deployable camera, not the RGB-blended
                # curriculum view used to ease optimization.
                sensor.set_camera_blend_alpha(1.0)
            try:
                metrics = evaluate_seg(sensor, eval_model, val_loader, device,
                                       num_classes=num_classes, out_dir=ckpt_dir, epoch=epoch, split_name="val")
            finally:
                if hasattr(sensor, "set_camera_blend_alpha"):
                    sensor.set_camera_blend_alpha(camera_alpha)
            pixel_acc = metrics["pixel_acc"]
            miou      = metrics["mIoU"]
            last_val_score = float(miou)

            train_deploy_metrics = None
            train_deploy_probe_batches = int(tcfg.get("train_deploy_probe_batches", 0))
            if train_deploy_probe_batches > 0:
                if hasattr(sensor, "set_camera_blend_alpha"):
                    sensor.set_camera_blend_alpha(1.0)
                try:
                    train_deploy_metrics = evaluate_seg(
                        sensor,
                        eval_model,
                        train_loader,
                        device,
                        num_classes=num_classes,
                        out_dir=ckpt_dir,
                        epoch=epoch,
                        max_batches=train_deploy_probe_batches,
                        write_outputs=False,
                        split_name="train_deploy_probe",
                    )
                finally:
                    if hasattr(sensor, "set_camera_blend_alpha"):
                        sensor.set_camera_blend_alpha(camera_alpha)
                logging.info(
                    "[EvalTrainProbe] Epoch %s | batches %d | pixel_acc %.4f | mIoU %.4f",
                    epoch,
                    train_deploy_probe_batches,
                    float(train_deploy_metrics["pixel_acc"]),
                    float(train_deploy_metrics["mIoU"]),
                )
                deploy_guard_best_probe = max(deploy_guard_best_probe, float(train_deploy_metrics["mIoU"]))

            if deploy_guard_enabled and camera_alpha >= deploy_guard_start_alpha:
                reasons: List[str] = []
                if np.isfinite(best) and float(miou) < float(best) - deploy_guard_val_drop:
                    reasons.append(f"val_mIoU_drop {float(miou):.4f} < best {float(best):.4f}-{deploy_guard_val_drop:.4f}")
                if (
                    train_deploy_metrics is not None
                    and np.isfinite(deploy_guard_best_probe)
                    and float(train_deploy_metrics["mIoU"]) < deploy_guard_best_probe - deploy_guard_probe_drop
                ):
                    reasons.append(
                        f"train_probe_drop {float(train_deploy_metrics['mIoU']):.4f} "
                        f"< best_probe {deploy_guard_best_probe:.4f}-{deploy_guard_probe_drop:.4f}"
                    )
                if reasons:
                    deploy_guard_actions += 1
                    factor = 0.0 if (deploy_guard_freeze_after > 0 and deploy_guard_actions >= deploy_guard_freeze_after) else deploy_guard_shrink
                    scales = sensor_trust.scale_sensor_lrs(
                        optim,
                        factor,
                        min_lr_mult=0.0 if factor == 0.0 else deploy_guard_min_lr_mult,
                    )
                    action = "freeze_sensor_lrs" if factor == 0.0 else "shrink_sensor_lrs"
                    msg = (
                        f"[DeployGuard] {action} after epoch {epoch}: "
                        + "; ".join(reasons)
                        + f"; sensor_lr_scales={ {k: round(float(v), 4) for k, v in scales.items()} }"
                    )
                    logging.warning(msg)
                    print(msg, flush=True)

            logging.info(f"[Eval] Epoch {epoch} | pixel_acc {pixel_acc:.4f} | mIoU {miou:.4f}")
            per_class = metrics.get("per_class_IoU", [])
            class_names = metrics.get("class_names", [])
            if per_class:
                order = sorted(range(len(per_class)), key=lambda idx: float(per_class[idx]))
                low = ", ".join(
                    f"{class_names[i] if i < len(class_names) else f'class{i}'}={float(per_class[i]):.3f}"
                    for i in order[: min(5, len(order))]
                )
                high = ", ".join(
                    f"{class_names[i] if i < len(class_names) else f'class{i}'}={float(per_class[i]):.3f}"
                    for i in order[-min(5, len(order)) :][::-1]
                )
                print(
                    f"EVAL epoch {epoch}/{int(tcfg['epochs'])} "
                    f"pixel_acc {pixel_acc:.4f} mIoU {miou:.4f} "
                    + (
                        ""
                        if train_deploy_metrics is None
                        else f"train_deploy_probe_mIoU {float(train_deploy_metrics['mIoU']):.4f} "
                    )
                    + f"lowest_IoU [{low}] highest_IoU [{high}]",
                    flush=True,
                )

            with open(os.path.join(ckpt_dir, f"metrics_epoch{epoch}.json"), "w") as f:
                metrics_to_write = dict(metrics)
                if train_deploy_metrics is not None:
                    metrics_to_write["train_deploy_probe"] = train_deploy_metrics
                json.dump(metrics_to_write, f, indent=2)

            csv_path = os.path.join(ckpt_dir, "metrics_log.csv")
            header = ["epoch", "pixel_acc", "mIoU", "train_deploy_probe_pixel_acc", "train_deploy_probe_mIoU"] + [f"IoU_{i}" for i in range(num_classes)]
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(header)
                w.writerow([
                    epoch,
                    pixel_acc,
                    miou,
                    "" if train_deploy_metrics is None else train_deploy_metrics["pixel_acc"],
                    "" if train_deploy_metrics is None else train_deploy_metrics["mIoU"],
                    *metrics["per_class_IoU"],
                ])
            if feedback_controller is not None:
                feedback_controller.update_from_validation(
                    metrics.get("per_class_IoU", []),
                    metrics.get("per_class_support", []),
                )
                feedback_controller.write_epoch_report(os.path.join(ckpt_dir, "monitoring"), epoch)

            if export_design_every > 0 and (epoch % export_design_every) == 0:
                try:
                    _export_sensor_design(
                        sensor,
                        os.path.join(ckpt_dir, f"camera_design_epoch{epoch}.json"),
                        include_kernels=True,
                    )
                    _save_sensor_preview(
                        sensor,
                        os.path.join(ckpt_dir, "design_preview"),
                        stem=f"epoch{epoch}",
                    )
                except Exception as e:
                    logging.warning(f"[DesignExport] epoch export failed: {e}")

            val_score = miou
        else:
            val_score = evaluate_cls(sensor, eval_model, val_loader, device)
            logging.info(f"[Eval] Epoch {epoch} | acc {val_score:.4f}")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_score)
        else:
            scheduler.step()
        lrs = [group["lr"] for group in optim.param_groups]
        lr_by_component = {
            str(group.get("component")): float(group["lr"])
            for group in optim.param_groups
            if group.get("component") is not None
        }
        sensor_lr_summary = "|".join(
            f"{comp}:{lr_by_component[comp]:.8e}"
            for comp in SensorTrustRegionController.COMPONENTS
            if comp in lr_by_component
        )
        logging.info(f"[LR] current LRs: {lrs}")

        denom = max(epoch_seen, 1)
        # Print epoch summary
        print("\n" + "="*70)
        print(f"EPOCH {epoch} SUMMARY")
        print("="*70)
        print(f"  Train Loss   : {epoch_loss_sum / denom:.6f}")
        if is_dense_task:
            print(f"  Train CE Loss: {epoch_ce_sum / denom:.6f}")
            print(f"  Train Overlap: {epoch_overlap_sum / denom:.6f}")
            if epoch_smooth_sum > 0:
                print(f"  Train Smooth : {epoch_smooth_sum / denom:.6f}")
            if epoch_feedback_sum > 0:
                print(f"  Task Feedback: {epoch_feedback_sum / denom:.6f}")
            if epoch_distill_sum > 0:
                print(f"  Distillation : {epoch_distill_sum / denom:.6f}")
            if abs(epoch_semantic_snr_sum) > 0:
                print(f"  Semantic SNR : {epoch_semantic_snr_sum / denom:.6f}")
            if epoch_deploy_sum > 0:
                print(f"  Deploy Loss  : {epoch_deploy_sum / denom:.6f}")
            if epoch_fixed_advantage_sum > 0:
                print(f"  Fixed Adv Hng: {epoch_fixed_advantage_sum / denom:.6f}")
            if epoch_sensor_reg_sum > 0:
                print(f"  Sensor Reg   : {epoch_sensor_reg_sum / denom:.6f}")
        print(f"  Val Score    : {val_score:.6f}  {'[BEST]' if val_score > best + 1e-6 else ''}")
        best_display = max(best, val_score) if np.isfinite(best) else val_score
        print(f"  Best So Far  : {best_display:.6f}")
        print(f"  Learning Rates: {[f'{lr:.2e}' for lr in lrs]}")
        print("="*70 + "\n")

        epoch_summary_path = os.path.join(ckpt_dir, "epoch_summary.csv")
        epoch_header = [
            "epoch",
            "train_loss",
            "train_ce",
            "train_overlap",
            "train_smooth",
            "train_feedback",
            "train_distill",
            "train_semantic_snr",
            "train_deploy",
            "train_fixed_advantage",
            "train_sensor_reg",
            "val_score",
            "best_so_far",
            "lr_model",
            "lr_sensor",
            "lr_sensor_optics",
            "lr_sensor_cfa",
            "lr_sensor_exposure",
            "lr_sensor_noise_adc",
            "lr_sensor_readout",
            "lr_sensor_other",
        ]
        write_epoch_header = not os.path.exists(epoch_summary_path)
        with open(epoch_summary_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_epoch_header:
                w.writerow(epoch_header)
            w.writerow([
                epoch,
                epoch_loss_sum / denom,
                epoch_ce_sum / denom,
                epoch_overlap_sum / denom,
                epoch_smooth_sum / denom,
                epoch_feedback_sum / denom,
                epoch_distill_sum / denom,
                epoch_semantic_snr_sum / denom,
                epoch_deploy_sum / denom,
                epoch_fixed_advantage_sum / denom,
                epoch_sensor_reg_sum / denom,
                val_score,
                best_display,
                lrs[0] if len(lrs) > 0 else "",
                sensor_lr_summary,
                lr_by_component.get("optics", ""),
                lr_by_component.get("cfa", ""),
                lr_by_component.get("exposure", ""),
                lr_by_component.get("noise_adc", ""),
                lr_by_component.get("readout", ""),
                lr_by_component.get("other", ""),
            ])
        _plot_training_curves(ckpt_dir)

        improved = val_score > best + early_stop_min_delta
        if improved:
            best = val_score
            last_improve_epoch = epoch
            if sensor_rollback_enabled or sensor_freeze_after_best_patience > 0:
                best_sensor_state = {
                    name: p.detach().clone()
                    for name, p in sensor.named_parameters()
                    if p.requires_grad
                }
            metric_tag = "miou" if is_dense_task else "acc"
            save_path = os.path.join(ckpt_dir, f"best_ep{epoch}_{metric_tag}{best:.4f}.pt")
            best_payload = {
                "sensor": sensor.state_dict(),
                "model": eval_model.state_dict(),
                "cfg": cfg,
            }
            if feedback_controller is not None:
                best_payload["feedback_controller"] = feedback_controller.state_dict()
            torch.save(best_payload, save_path)
            logging.info(f"Saved best checkpoint to {save_path}")
            try:
                _export_sensor_design(
                    sensor,
                    os.path.join(ckpt_dir, "camera_design_best.json"),
                    include_kernels=True,
                )
                _save_sensor_preview(
                    sensor,
                    os.path.join(ckpt_dir, "design_preview"),
                    stem="best",
                )
            except Exception as e:
                logging.warning(f"[DesignExport] best export failed: {e}")
        elif (
            sensor_rollback_enabled
            and best_sensor_state
            and np.isfinite(best)
            and float(val_score) < float(best) - sensor_rollback_val_tolerance
        ):
            SensorTrustRegionController.restore_state(sensor, best_sensor_state)
            scales = sensor_trust.scale_sensor_lrs(
                optim,
                sensor_rollback_shrink,
                min_lr_mult=deploy_guard_min_lr_mult,
            )
            msg = (
                f"[SensorRollback] restored best sensor after epoch {epoch}: "
                f"val_score {float(val_score):.4f} < best {float(best):.4f}-"
                f"{sensor_rollback_val_tolerance:.4f}; "
                f"sensor_lr_scales={ {k: round(float(v), 4) for k, v in scales.items()} }"
            )
            logging.warning(msg)
            print(msg, flush=True)

        if (
            sensor_freeze_after_best_patience > 0
            and not sensor_frozen_by_plateau
            and best_sensor_state
            and np.isfinite(best)
            and (epoch - last_improve_epoch) >= sensor_freeze_after_best_patience
        ):
            if sensor_freeze_restore_best:
                SensorTrustRegionController.restore_state(sensor, best_sensor_state)
            for p in sensor.parameters():
                p.requires_grad_(False)
            scales = sensor_trust.scale_sensor_lrs(optim, 0.0, min_lr_mult=0.0)
            sensor_frozen_by_plateau = True
            msg = (
                f"[SensorPlateauFreeze] froze sensor after epoch {epoch}: "
                f"no val improvement for {sensor_freeze_after_best_patience} epoch(s); "
                f"restored_best={sensor_freeze_restore_best}; "
                f"sensor_lr_scales={ {k: round(float(v), 4) for k, v in scales.items()} }"
            )
            logging.warning(msg)
            print(msg, flush=True)

        # ---- Save rolling last + N recent ----
        last_payload = _make_ckpt(
            sensor,
            model,
            optim,
            scaler,
            scheduler,
            cfg,
            epoch,
            step,
            best,
            ema_model=ema_model,
            feedback_controller=feedback_controller,
        )
        last_path = os.path.join(ckpt_dir, "last.pt")
        torch.save(last_payload, last_path)

        ep_path = os.path.join(ckpt_dir, f"last_ep{epoch}.pt")
        torch.save(last_payload, ep_path)
        try:
            if keep_last_n > 0:
                pts = sorted(
                    glob.glob(os.path.join(ckpt_dir, "last_ep*.pt")),
                    key=lambda p: int(re.findall(r"last_ep(\d+)\.pt", p)[0])
                )
                for p in pts[:-keep_last_n]:
                    os.remove(p)
        except Exception as e:
            logging.debug(f"[Checkpoint] Prune skipped: {e}")

        if early_stop_pat > 0 and (epoch - last_improve_epoch) >= early_stop_pat:
            msg = (
                f"[EarlyStop] No validation improvement > {early_stop_min_delta:.6f} "
                f"for {early_stop_pat} epochs; stopping at epoch {epoch}."
            )
            logging.info(msg)
            print(msg, flush=True)
            break

        if hasattr(sensor, "lens") and (sensor.lens is not None):
            out_json = os.path.join(ckpt_dir, f"lens_epoch{epoch}.json")
            try:
                sensor.lens.write_lens_json(out_json)
                logging.info(f"Saved lens JSON to {out_json}")
            except Exception as e:
                logging.warning(f"Lens export failed: {e}")

    # Final training summary
    best = _maybe_eval_checkpoint_average(
        sensor=sensor,
        model=model,
        cfg=cfg,
        val_loader=val_loader,
        device=device,
        num_classes=num_classes,
        ckpt_dir=ckpt_dir,
        best=best,
        task=task,
    )

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"  Experiment  : {cfg.get('name', 'unnamed')}")
    print(f"  Total Epochs: {epoch}")
    print(f"  Best Score  : {best:.6f}")
    print(f"  Time Took   : {_format_seconds(time.time() - train_wall_start)}")
    print(f"  Checkpoint  : {os.path.join(ckpt_dir, 'best_*.pt')}")
    print(f"  Metrics Log : {os.path.join(ckpt_dir, 'metrics_log.csv')}")
    print(f"  Plots       : {os.path.join(ckpt_dir, 'plots/')}")
    print("="*70 + "\n")

    logging.info(f"Best score: {best:.4f}")
    try:
        with open(os.path.join(ckpt_dir, "train_complete.json"), "w") as f:
            json.dump(
                {
                    "epoch": int(epoch),
                    "target_epochs": int(tcfg["epochs"]),
                    "best": float(best),
                    "wall_time_seconds": float(time.time() - train_wall_start),
                    "completed_at_unix": float(time.time()),
                    "early_stop_patience": int(early_stop_pat),
                    "early_stop_min_delta": float(early_stop_min_delta),
                },
                f,
                indent=2,
            )
    except Exception as e:
        logging.warning(f"[TrainComplete] Failed to write completion marker: {e}")

# -------------------------
# Entrypoint
# -------------------------
def main():
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    logging.info(f"Set global seed: {seed}")
    train(cfg)

if __name__ == "__main__":
    main()
