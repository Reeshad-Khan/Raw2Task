"""Train real voxel occupancy with the optics-sensor co-design frontend."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import time
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
import yaml

from raw2task.train_extended import (
    CoDesignSensor,
    PALETTE_19,
    PassThroughSensor,
    TaskFeedbackController,
    _export_sensor_design,
    _format_seconds,
    _progress_bar,
    set_seed,
)
from raw2task.occupancy.dataset import OccupancyManifestDataset, occupancy_collate
from raw2task.occupancy.metrics import summarize_voxel_confusion, update_voxel_confusion
from raw2task.occupancy.model import MultiCameraSensorFrontend, OccupancyReferenceNet


def _occ_palette(num_classes: int) -> np.ndarray:
    base = list(PALETTE_19)
    if len(base) < num_classes:
        rng = np.random.default_rng(7)
        for _ in range(num_classes - len(base)):
            base.append(tuple(int(x) for x in rng.integers(20, 235, size=3)))
    return np.asarray(base[:num_classes], dtype=np.uint8)


def _bev_projection(volume: torch.Tensor, valid: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    """Project D,H,W semantic occupancy to H,W using nearest visible occupied voxel."""
    vol = volume.detach().cpu().long()
    val = valid.detach().cpu().bool() & (vol != int(ignore_index))
    out = torch.full(vol.shape[-2:], int(ignore_index), dtype=torch.long)
    for z in range(vol.shape[0]):
        write = val[z] & (out == int(ignore_index))
        out[write] = vol[z][write]
    fallback = val.any(dim=0) & (out == int(ignore_index))
    if fallback.any():
        out[fallback] = 0
    return out


def _label_to_rgb(label: torch.Tensor, palette: np.ndarray, ignore_index: int = 255) -> Image.Image:
    arr = label.detach().cpu().numpy().astype(np.int64)
    h, w = arr.shape
    rgb = np.full((h, w, 3), 245, dtype=np.uint8)
    valid = (arr != int(ignore_index)) & (arr >= 0) & (arr < len(palette))
    rgb[valid] = palette[arr[valid]]
    return Image.fromarray(rgb, mode="RGB")


def _resize_for_panel(img: Image.Image, height: int = 260) -> Image.Image:
    scale = height / max(1, img.height)
    return img.resize((max(1, int(round(img.width * scale))), height), Image.Resampling.NEAREST)


def _draw_caption(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.text(xy, text, fill=(20, 20, 20), font=font)


def save_occupancy_viz(
    images: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    out_path: str,
    num_classes: int,
    ignore_index: int = 255,
    title: str = "",
) -> None:
    """Save a paper-facing camera + BEV occupancy panel."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    palette = _occ_palette(num_classes)
    rgb = images[0, 0].detach().cpu().float().clamp(0, 1)
    rgb_img = Image.fromarray((rgb.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8), mode="RGB")
    rgb_img = rgb_img.resize((420, 236), Image.Resampling.BILINEAR)

    gt_bev = _bev_projection(target[0], valid[0], ignore_index=ignore_index)
    pred_bev = _bev_projection(pred[0], valid[0], ignore_index=ignore_index)
    valid_bev = valid[0].detach().cpu().bool().any(dim=0)
    err = (gt_bev != pred_bev) & valid_bev & (gt_bev != int(ignore_index))

    gt_img = _resize_for_panel(_label_to_rgb(gt_bev, palette, ignore_index=ignore_index))
    pred_img = _resize_for_panel(_label_to_rgb(pred_bev, palette, ignore_index=ignore_index))
    err_rgb = np.full((gt_bev.shape[0], gt_bev.shape[1], 3), 245, dtype=np.uint8)
    err_rgb[valid_bev.numpy()] = np.array([210, 232, 246], dtype=np.uint8)
    err_rgb[err.numpy()] = np.array([210, 40, 40], dtype=np.uint8)
    err_img = _resize_for_panel(Image.fromarray(err_rgb, mode="RGB"))

    pad = 18
    cap_h = 34
    header_h = 36 if title else 0
    widths = [rgb_img.width, gt_img.width, pred_img.width, err_img.width]
    heights = [rgb_img.height, gt_img.height, pred_img.height, err_img.height]
    canvas_w = sum(widths) + pad * (len(widths) + 1)
    canvas_h = max(heights) + cap_h + header_h + pad * 2
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    if title:
        _draw_caption(draw, (pad, 8), title)
    x = pad
    y = pad + header_h
    for img, caption in [
        (rgb_img, "front camera input"),
        (gt_img, "GT 3D semantic BEV"),
        (pred_img, "predicted 3D semantic BEV"),
        (err_img, "BEV error map"),
    ]:
        canvas.paste(img, (x, y))
        _draw_caption(draw, (x, y + max(heights) + 8), caption)
        x += img.width + pad
    canvas.save(out_path)


def plot_occupancy_curves(ckpt_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    metrics_path = os.path.join(ckpt_dir, "metrics_log.csv")
    if not os.path.isfile(metrics_path):
        return
    rows = []
    with open(metrics_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    os.makedirs(os.path.join(ckpt_dir, "plots"), exist_ok=True)
    xs = [float(r.get("epoch", 0) or 0) for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for key, label in [("mIoU", "voxel mIoU"), ("occupied_IoU", "occupied IoU"), ("voxel_acc", "voxel acc")]:
        vals = [float(r.get(key, "nan") or "nan") for r in rows]
        ax.plot(xs, vals, marker="o", linewidth=1.6, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(ckpt_dir, "plots", "occupancy_validation_curves.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(xs, [float(r.get("train_loss", "nan") or "nan") for r in rows], marker="o", label="train loss")
    if "train_feedback" in rows[0]:
        ax.plot(xs, [float(r.get("train_feedback", "nan") or "nan") for r in rows], marker="o", label="task feedback")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(ckpt_dir, "plots", "occupancy_training_curves.png"), dpi=180)
    plt.close(fig)


def load_config(path: str) -> Dict[str, Any]:
    with open(os.path.expanduser(path), "r") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("seed", random.randint(0, 10_000))
    cfg.setdefault("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    cfg["device"] = torch.device(cfg["device"]) if not isinstance(cfg["device"], torch.device) else cfg["device"]
    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg.setdefault("sensor", {})
    cfg.setdefault("lens", {})
    cfg.setdefault("pipeline", {"input": "sensor"})
    cfg.setdefault("train", {})
    cfg["train"].setdefault("epochs", 24)
    cfg["train"].setdefault("batch_size", cfg["data"].get("batch_size", 1))
    cfg["train"].setdefault("num_workers", cfg["data"].get("num_workers", 4))
    cfg["train"].setdefault("lr", 2e-4)
    cfg["train"].setdefault("weight_decay", 1e-4)
    cfg["train"].setdefault("sensor_lr_mult", 0.1)
    cfg["train"].setdefault("ckpt_dir", "./runs/occupancy")
    cfg["train"].setdefault("log_interval", 20)
    cfg["train"].setdefault("amp", True)
    cfg["train"].setdefault("grad_clip_norm", 1.0)
    cfg["train"].setdefault("task_feedback_weight", 0.0)
    cfg["train"].setdefault("task_feedback_entropy_weight", 0.5)
    cfg["train"].setdefault("task_feedback_error_weight", 1.0)
    cfg["train"].setdefault("task_feedback_boundary_weight", 1.0)
    cfg["train"].setdefault("task_feedback_occupied_weight", 1.0)
    cfg["train"].setdefault("task_feedback_detach", True)
    cfg["train"].setdefault("task_feedback_controller", True)
    cfg["train"].setdefault("task_feedback_controller_ema", 0.70)
    cfg["train"].setdefault("task_feedback_iou_gamma", 1.50)
    cfg["train"].setdefault("task_feedback_min_pressure", 0.75)
    cfg["train"].setdefault("task_feedback_max_pressure", 2.50)
    cfg["train"].setdefault("resume", False)
    cfg["train"].setdefault("resume_path", "")
    cfg["model"].setdefault("num_classes", 18)
    cfg["model"].setdefault("ignore_index", 255)
    cfg["model"].setdefault("width", 64)
    return cfg


def build_dataset(cfg: Dict[str, Any], split: str) -> OccupancyManifestDataset:
    d = cfg["data"]
    manifest = d[f"{split}_manifest"]
    return OccupancyManifestDataset(
        root=d["root"],
        manifest=manifest,
        image_size=tuple(d.get("image_size", [256, 704])),
        grid_shape=tuple(d.get("grid_shape", [16, 100, 100])),
        num_cameras=int(d.get("num_cameras", 6)),
        ignore_index=int(cfg["model"].get("ignore_index", 255)),
        require_calibration=bool(d.get("require_calibration", True)),
        require_camera_mask=bool(d.get("require_camera_mask", True)),
        max_samples=d.get(f"max_{split}_samples"),
        stride=int(d.get(f"{split}_stride", 1)),
    )


def build_frontend(cfg: Dict[str, Any]) -> MultiCameraSensorFrontend:
    mode = str(cfg.get("pipeline", {}).get("input", "sensor")).lower()
    if mode in ("rgb", "processed_rgb"):
        sensor = PassThroughSensor(cfg)
    elif mode in ("sensor", "codesign", "camera_sim"):
        sensor = CoDesignSensor(cfg)
    else:
        raise ValueError(f"Unsupported pipeline.input={mode!r}")
    return MultiCameraSensorFrontend(sensor)


def voxel_boundary_mask(target: torch.Tensor, valid: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    """Mark semantic transitions in a voxel grid."""
    valid = valid & (target != ignore_index)
    mask = torch.zeros_like(target, dtype=torch.float32)
    dz = (target[:, 1:, :, :] != target[:, :-1, :, :]) & valid[:, 1:, :, :] & valid[:, :-1, :, :]
    dy = (target[:, :, 1:, :] != target[:, :, :-1, :]) & valid[:, :, 1:, :] & valid[:, :, :-1, :]
    dx = (target[:, :, :, 1:] != target[:, :, :, :-1]) & valid[:, :, :, 1:] & valid[:, :, :, :-1]
    mask[:, 1:, :, :] = torch.maximum(mask[:, 1:, :, :], dz.float())
    mask[:, :-1, :, :] = torch.maximum(mask[:, :-1, :, :], dz.float())
    mask[:, :, 1:, :] = torch.maximum(mask[:, :, 1:, :], dy.float())
    mask[:, :, :-1, :] = torch.maximum(mask[:, :, :-1, :], dy.float())
    mask[:, :, :, 1:] = torch.maximum(mask[:, :, :, 1:], dx.float())
    mask[:, :, :, :-1] = torch.maximum(mask[:, :, :, :-1], dx.float())
    return mask


def voxel_task_feedback_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    class_pressure: torch.Tensor | None = None,
    ignore_index: int = 255,
    occupied_class_ids: Iterable[int] | None = None,
    entropy_weight: float = 0.0,
    error_weight: float = 0.0,
    boundary_weight: float = 0.0,
    occupied_weight: float = 0.0,
    detach_feedback: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Training-only task feedback for occupancy co-design."""
    valid = valid & (target != ignore_index)
    if not valid.any():
        zero = logits.mean() * 0.0
        return zero, {"entropy": zero.detach(), "boundary": zero.detach(), "occupied": zero.detach()}

    with torch.no_grad() if detach_feedback else contextlib.nullcontext():
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1)
        entropy = (entropy / math.log(max(logits.shape[1], 2))).clamp(0.0, 1.0)
        safe_target = target.clamp(min=0, max=logits.shape[1] - 1)
        true_prob = probs.gather(1, safe_target.unsqueeze(1)).squeeze(1)
        pred = probs.argmax(dim=1)
        error = 0.5 * (pred != target).to(dtype=entropy.dtype) + 0.5 * (1.0 - true_prob).clamp(0.0, 1.0)
        boundary = voxel_boundary_mask(target, valid, ignore_index=ignore_index).to(device=logits.device, dtype=logits.dtype)
        occupied = torch.zeros_like(entropy)
        ids = list(occupied_class_ids or [])
        if ids:
            for cls_id in ids:
                occupied = torch.maximum(occupied, (target == int(cls_id)).to(dtype=occupied.dtype))
        else:
            occupied = ((target > 0) & valid).to(dtype=entropy.dtype)
        pressure = torch.ones_like(entropy)
        if class_pressure is not None:
            safe_target = target.clamp(min=0, max=int(class_pressure.numel()) - 1)
            pressure = class_pressure.to(device=logits.device, dtype=logits.dtype)[safe_target]
            pressure = torch.where(valid, pressure, torch.ones_like(pressure))
        feedback = torch.ones_like(entropy)
        feedback = feedback + float(entropy_weight) * entropy
        feedback = feedback + float(error_weight) * error
        feedback = feedback + float(boundary_weight) * boundary
        feedback = feedback + float(occupied_weight) * occupied
        feedback = feedback * pressure
        feedback = feedback * valid.to(dtype=feedback.dtype)
        feedback = feedback / feedback[valid].mean().clamp_min(1e-6)

    ce_map = F.cross_entropy(logits, target, ignore_index=ignore_index, reduction="none")
    loss = (ce_map * feedback)[valid].mean()
    stats = {
        "entropy": entropy[valid].mean().detach(),
        "error": error[valid].mean().detach(),
        "boundary": boundary[valid].mean().detach(),
        "occupied": occupied[valid].mean().detach(),
        "pressure": pressure[valid].mean().detach(),
    }
    return loss, stats


@torch.no_grad()
def evaluate(frontend, model, loader, cfg, out_dir: str | None = None, epoch: int = 0) -> Dict[str, Any]:
    device = cfg["device"]
    num_classes = int(cfg["model"]["num_classes"])
    ignore_index = int(cfg["model"].get("ignore_index", 255))
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    frontend.eval()
    model.eval()
    eval_start = time.time()
    eval_log_interval = max(1, len(loader) // 10) if hasattr(loader, "__len__") else 25
    task_variant = str(cfg.get("data", {}).get("task_variant", "observed_semantic_occupancy"))
    task_title = "Observed 3D semantic segmentation" if "3d_semantic" in task_variant else "Observed semantic occupancy"
    for step, batch in enumerate(loader, 1):
        images = batch["images"].to(device)
        target = batch["occupancy"].to(device)
        valid = batch["valid_mask"].to(device)
        raw = frontend(images)
        logits = model(raw, batch["intrinsics"].to(device), batch["extrinsics"].to(device))
        if tuple(logits.shape[-3:]) != tuple(target.shape[-3:]):
            logits = F.interpolate(logits, size=target.shape[-3:], mode="trilinear", align_corners=False)
        pred = logits.argmax(dim=1)
        if out_dir and step == 1:
            save_occupancy_viz(
                images=batch["images"].detach().cpu(),
                pred=pred.detach().cpu(),
                target=target.detach().cpu(),
                valid=valid.detach().cpu(),
                out_path=os.path.join(out_dir, "viz", f"epoch{int(epoch):03d}_occupancy_bev.png"),
                num_classes=num_classes,
                ignore_index=ignore_index,
                title=f"{task_title} validation, epoch {int(epoch)}",
            )
        update_voxel_confusion(conf, pred, target, valid, num_classes=num_classes, ignore_index=ignore_index)
        if step == 1 or step % eval_log_interval == 0 or step == len(loader):
            elapsed = time.time() - eval_start
            eta_eval = elapsed / max(1, step) * max(0, len(loader) - step)
            metrics = summarize_voxel_confusion(conf)
            print(
                f"OCC EVAL progress {_progress_bar(step, len(loader))} "
                f"batch {step}/{len(loader)} ETA {_format_seconds(eta_eval)} "
                f"voxel_acc {metrics['voxel_acc']:.4f} "
                f"running_mIoU {metrics['mIoU']:.4f} "
                f"running_occIoU {metrics['occupied_IoU']:.4f}",
                flush=True,
            )
    metrics = summarize_voxel_confusion(conf)
    if out_dir:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
            fig, ax = plt.subplots(figsize=(9.5, 4.2))
            vals = metrics.get("per_class_IoU", [])
            ax.bar(np.arange(len(vals)), vals, color="#4c78a8")
            ax.set_xlabel("voxel class id")
            ax.set_ylabel("IoU")
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"Per-class voxel IoU, epoch {int(epoch)}")
            ax.grid(True, axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "plots", f"occupancy_per_class_iou_epoch{int(epoch):03d}.png"), dpi=180)
            plt.close(fig)
        except Exception:
            pass
    return metrics


def train(cfg: Dict[str, Any]) -> None:
    set_seed(int(cfg["seed"]))
    device = cfg["device"]
    ckpt_dir = os.path.expanduser(cfg["train"]["ckpt_dir"])
    os.makedirs(ckpt_dir, exist_ok=True)

    train_ds = build_dataset(cfg, "train")
    val_ds = build_dataset(cfg, "val")
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
        collate_fn=occupancy_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
        collate_fn=occupancy_collate,
    )

    frontend = build_frontend(cfg).to(device)
    model = OccupancyReferenceNet(
        in_channels=frontend.output_channels,
        num_classes=int(cfg["model"]["num_classes"]),
        grid_shape=tuple(cfg["data"]["grid_shape"]),
        width=int(cfg["model"].get("width", 64)),
    ).to(device)

    base_lr = float(cfg["train"]["lr"])
    sensor_lr = base_lr * float(cfg["train"].get("sensor_lr_mult", 0.1))
    params = [{"params": model.parameters(), "lr": base_lr}]
    sensor_params = [p for p in frontend.sensor.parameters() if p.requires_grad]
    if sensor_params:
        params.append({"params": sensor_params, "lr": sensor_lr})
    optim = torch.optim.AdamW(params, weight_decay=float(cfg["train"]["weight_decay"]))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"].get("amp", True)) and device.type == "cuda")
    ignore_index = int(cfg["model"].get("ignore_index", 255))
    task_feedback_weight = float(cfg["train"].get("task_feedback_weight", 0.0))
    task_feedback_entropy_weight = float(cfg["train"].get("task_feedback_entropy_weight", 0.5))
    task_feedback_error_weight = float(cfg["train"].get("task_feedback_error_weight", 1.0))
    task_feedback_boundary_weight = float(cfg["train"].get("task_feedback_boundary_weight", 1.0))
    task_feedback_occupied_weight = float(cfg["train"].get("task_feedback_occupied_weight", 1.0))
    task_feedback_detach = bool(cfg["train"].get("task_feedback_detach", True))
    class_names = [f"voxel_class_{idx}" for idx in range(int(cfg["model"]["num_classes"]))]
    feedback_controller = TaskFeedbackController(int(cfg["model"]["num_classes"]), class_names, cfg, device)
    best = -1.0
    start_epoch = 1

    resume_path = str(cfg["train"].get("resume_path", "") or "")
    if not resume_path and bool(cfg["train"].get("resume", False)):
        candidate = os.path.join(ckpt_dir, "last.pt")
        resume_path = candidate if os.path.isfile(candidate) else ""
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        frontend.load_state_dict(ckpt["frontend"], strict=True)
        model.load_state_dict(ckpt["model"], strict=True)
        if ckpt.get("optim"):
            optim.load_state_dict(ckpt["optim"])
        if ckpt.get("scaler"):
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                pass
        if ckpt.get("feedback_controller") is not None:
            feedback_controller.load_state_dict(ckpt.get("feedback_controller"), device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best = float(ckpt.get("best", -1.0))
        print(f"[resume] loaded {resume_path}; start_epoch={start_epoch}; best={best:.4f}")

    if start_epoch == 1:
        _export_sensor_design(frontend.sensor, os.path.join(ckpt_dir, "camera_design_initial.json"))
    with open(os.path.join(ckpt_dir, "resolved_config.yaml"), "w") as f:
        yaml.safe_dump({**cfg, "device": str(device)}, f, sort_keys=False)

    csv_path = os.path.join(ckpt_dir, "metrics_log.csv")
    last_val_miou = None
    for epoch in range(start_epoch, int(cfg["train"]["epochs"]) + 1):
        epoch_start_time = time.time()
        frontend.train()
        model.train()
        running = 0.0
        running_feedback = 0.0
        seen = 0
        for step, batch in enumerate(train_loader, 1):
            images = batch["images"].to(device)
            target = batch["occupancy"].to(device)
            valid = batch["valid_mask"].to(device)
            with torch.cuda.amp.autocast(enabled=bool(cfg["train"].get("amp", True)) and device.type == "cuda"):
                raw = frontend(images)
                logits = model(raw, batch["intrinsics"].to(device), batch["extrinsics"].to(device))
                if tuple(logits.shape[-3:]) != tuple(target.shape[-3:]):
                    logits = F.interpolate(logits, size=target.shape[-3:], mode="trilinear", align_corners=False)
                masked_target = target.clone()
                masked_target[~valid] = ignore_index
                loss = F.cross_entropy(logits, masked_target, ignore_index=ignore_index)
                feedback_for_log = logits.new_tensor(0.0)
                if task_feedback_weight > 0.0:
                    feedback_loss, feedback_stats = voxel_task_feedback_loss(
                        logits,
                        masked_target,
                        valid,
                        class_pressure=feedback_controller.current_pressure(),
                        ignore_index=ignore_index,
                        occupied_class_ids=cfg["data"].get("occupied_class_ids", []),
                        entropy_weight=task_feedback_entropy_weight,
                        error_weight=task_feedback_error_weight,
                        boundary_weight=task_feedback_boundary_weight,
                        occupied_weight=task_feedback_occupied_weight,
                        detach_feedback=task_feedback_detach,
                    )
                    feedback_for_log = feedback_loss
                    loss = loss + task_feedback_weight * feedback_loss
                if hasattr(frontend.sensor, "regularization"):
                    loss = loss + float(cfg["train"].get("sensor_reg_weight", 0.0)) * frontend.sensor.regularization()["total"]
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if float(cfg["train"].get("grad_clip_norm", 0.0)) > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(frontend.parameters()), float(cfg["train"]["grad_clip_norm"]))
            scaler.step(optim)
            scaler.update()
            running += float(loss.detach()) * images.size(0)
            running_feedback += float(feedback_for_log.detach()) * images.size(0)
            seen += images.size(0)
            if step % int(cfg["train"]["log_interval"]) == 0:
                with torch.no_grad():
                    pred = logits.argmax(1)
                    batch_conf = np.zeros((int(cfg["model"]["num_classes"]), int(cfg["model"]["num_classes"])), dtype=np.int64)
                    update_voxel_confusion(
                        batch_conf,
                        pred,
                        target,
                        valid,
                        num_classes=int(cfg["model"]["num_classes"]),
                        ignore_index=ignore_index,
                    )
                    batch_metrics = summarize_voxel_confusion(batch_conf)
                elapsed = time.time() - epoch_start_time
                steps_left = max(0, len(train_loader) - step)
                eta_epoch = elapsed / max(1, step) * steps_left
                epochs_left = max(0, int(cfg["train"]["epochs"]) - epoch)
                eta_total = eta_epoch + epochs_left * (elapsed / max(1, step) * max(1, len(train_loader)))
                print(
                    f"OCC TRAIN epoch {epoch}/{int(cfg['train']['epochs'])} "
                    f"{_progress_bar(step, len(train_loader))} "
                    f"step {step}/{len(train_loader)} "
                    f"ETA_epoch {_format_seconds(eta_epoch)} "
                    f"ETA_total {_format_seconds(eta_total)} "
                    f"loss {running / max(seen, 1):.4f} "
                    f"feedback {running_feedback / max(seen, 1):.4f} "
                    f"batch_voxel_acc {batch_metrics['voxel_acc']:.4f} "
                    f"batch_mIoU {batch_metrics['mIoU']:.4f} "
                    f"batch_occIoU {batch_metrics['occupied_IoU']:.4f} "
                    f"prev_val_mIoU {'n/a' if last_val_miou is None else f'{last_val_miou:.4f}'} "
                    f"best_mIoU {best:.4f}",
                    flush=True,
                )

        metrics = evaluate(frontend, model, val_loader, cfg, out_dir=ckpt_dir, epoch=epoch)
        metrics["epoch"] = epoch
        metrics["train_loss"] = running / max(seen, 1)
        metrics["train_feedback"] = running_feedback / max(seen, 1)
        feedback_controller.update_from_validation(
            metrics.get("per_class_IoU", []),
            metrics.get("per_class_support", []),
        )
        feedback_controller.write_epoch_report(os.path.join(ckpt_dir, "monitoring"), epoch)
        last_val_miou = float(metrics["mIoU"])
        print(
            f"OCC EVAL epoch {epoch}/{int(cfg['train']['epochs'])} "
            f"voxel_acc {metrics['voxel_acc']:.4f} "
            f"mIoU {metrics['mIoU']:.4f} "
            f"occupied_IoU {metrics['occupied_IoU']:.4f}",
            flush=True,
        )
        print(json.dumps(metrics, indent=2))
        write_header = not os.path.isfile(csv_path)
        with open(csv_path, "a", newline="") as f:
            fields = ["epoch", "train_loss", "train_feedback", "voxel_acc", "mIoU", "occupied_IoU"]
            writer = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow({k: metrics.get(k, "") for k in fields})
        with open(os.path.join(ckpt_dir, f"metrics_epoch{epoch}.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        plot_occupancy_curves(ckpt_dir)

        torch.save(
            {
                "frontend": frontend.state_dict(),
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "scaler": scaler.state_dict(),
                "cfg": {**cfg, "device": str(device)},
                "epoch": epoch,
                "best": best,
                "feedback_controller": feedback_controller.state_dict(),
            },
            os.path.join(ckpt_dir, "last.pt"),
        )
        if metrics["mIoU"] > best:
            best = float(metrics["mIoU"])
            torch.save(
                {
                    "frontend": frontend.state_dict(),
                    "model": model.state_dict(),
                    "cfg": {**cfg, "device": str(device)},
                    "epoch": epoch,
                    "best": best,
                    "feedback_controller": feedback_controller.state_dict(),
                },
                os.path.join(ckpt_dir, f"best_ep{epoch}_miou{best:.4f}.pt"),
            )
            _export_sensor_design(frontend.sensor, os.path.join(ckpt_dir, "camera_design_best.json"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
