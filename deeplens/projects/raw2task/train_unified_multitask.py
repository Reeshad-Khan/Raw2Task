"""Train one optics-sensor-model stack for 2D, 3D semantics, and occupancy."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from deeplens.projects.raw2task.data.kitti360_seg import KITTI360Seg
from deeplens.projects.raw2task.models.unified_multitask import UnifiedRaw2TaskModel
from deeplens.projects.raw2task.occupancy.dataset import OccupancyManifestDataset, occupancy_collate
from deeplens.projects.raw2task.occupancy.metrics import summarize_voxel_confusion, update_voxel_confusion
from deeplens.projects.raw2task.occupancy.train_occupancy import save_occupancy_viz
from deeplens.projects.raw2task.train_extended import (
    CoDesignSensor,
    PassThroughSensor,
    _export_sensor_design,
    _format_seconds,
    _progress_bar,
    _save_sensor_preview,
    iou_from_confusion,
    set_seed,
    update_confusion,
    vis_batch_with_stages,
)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("train", {})
    cfg.setdefault("model", {})
    cfg.setdefault("data_2d", {})
    cfg.setdefault("data_voxel", {})
    return cfg


def _cycle(loader: Iterable):
    while True:
        for batch in loader:
            yield batch


def build_2d_dataset(cfg: Dict[str, Any], split: str) -> KITTI360Seg:
    d = cfg["data_2d"]
    return KITTI360Seg(
        root=d["root_seg"],
        split=split,
        img_size=tuple(d.get("img_size", [256, 768])),
        camera=d.get("camera", "image_00"),
        label_encoding=d.get("label_encoding", "original_id"),
        split_file=d.get(f"{split}_split_file", None),
        max_samples=d.get(f"max_{split}_samples", None),
        stride=int(d.get(f"{split}_stride", 1)),
        photometric_jitter=float(d.get("photometric_jitter", 0.0)) if split == "train" else 0.0,
    )


def build_voxel_dataset(cfg: Dict[str, Any], split: str) -> OccupancyManifestDataset:
    d = cfg["data_voxel"]
    return OccupancyManifestDataset(
        root=d["root"],
        manifest=d[f"{split}_manifest"],
        image_size=tuple(d.get("image_size", [192, 640])),
        grid_shape=tuple(d.get("grid_shape", [16, 64, 128])),
        num_cameras=int(d.get("num_cameras", 1)),
        ignore_index=int(cfg["model"].get("ignore_index", 255)),
        require_calibration=bool(d.get("require_calibration", True)),
        require_camera_mask=bool(d.get("require_camera_mask", True)),
        max_samples=d.get(f"max_{split}_samples", None),
        stride=int(d.get(f"{split}_stride", 1)),
    )


def build_loaders(cfg: Dict[str, Any]):
    workers = int(cfg["train"].get("num_workers", 4))
    train_2d = build_2d_dataset(cfg, "train")
    val_2d = build_2d_dataset(cfg, "val")
    train_vox = build_voxel_dataset(cfg, "train")
    val_vox = build_voxel_dataset(cfg, "val")
    bs_2d = int(cfg["data_2d"].get("batch_size", cfg["train"].get("batch_size_2d", 4)))
    bs_vox = int(cfg["data_voxel"].get("batch_size", cfg["train"].get("batch_size_voxel", 1)))
    return {
        "train_2d": DataLoader(train_2d, batch_size=bs_2d, shuffle=True, num_workers=workers, pin_memory=True),
        "val_2d": DataLoader(val_2d, batch_size=bs_2d, shuffle=False, num_workers=workers, pin_memory=True),
        "train_vox": DataLoader(train_vox, batch_size=bs_vox, shuffle=True, num_workers=workers, pin_memory=True, collate_fn=occupancy_collate),
        "val_vox": DataLoader(val_vox, batch_size=bs_vox, shuffle=False, num_workers=workers, pin_memory=True, collate_fn=occupancy_collate),
    }


def sensor_images(sensor, images: torch.Tensor, return_stages: bool = False):
    if images.ndim == 4:
        return sensor(images, return_stages=return_stages)
    b, n, c, h, w = images.shape
    if return_stages:
        out, stages = sensor(images.reshape(b * n, c, h, w), return_stages=True)
    else:
        out = sensor(images.reshape(b * n, c, h, w))
    _, co, ho, wo = out.shape
    out = out.reshape(b, n, co, ho, wo)
    if not return_stages:
        return out
    return out, stages


@torch.no_grad()
def evaluate_2d(sensor, model, loader, device, ignore_index: int, num_classes: int) -> Dict[str, Any]:
    sensor.eval()
    model.eval()
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for rgb, target in loader:
        rgb = rgb.to(device)
        target = target.to(device)
        raw = sensor_images(sensor, rgb)
        logits = model.forward_2d(raw, out_size=target.shape[-2:])
        pred = logits.argmax(1)
        update_confusion(conf, pred, target, num_classes=num_classes, ignore_index=ignore_index)
    miou, per_class = iou_from_confusion(conf)
    acc = float(np.diag(conf).sum() / max(conf.sum(), 1))
    return {"mIoU_2d": float(miou), "pixel_acc_2d": acc, "per_class_iou_2d": per_class.tolist()}


@torch.no_grad()
def evaluate_voxel(sensor, model, loader, device, ignore_index: int, num_classes: int) -> Dict[str, Any]:
    sensor.eval()
    model.eval()
    conf3d = np.zeros((num_classes, num_classes), dtype=np.int64)
    conf_occ = np.zeros((2, 2), dtype=np.int64)
    occ_pos = 0
    occ_valid = 0
    for batch in loader:
        images = batch["images"].to(device)
        target = batch["occupancy"].to(device)
        valid = batch["valid_mask"].to(device)
        raw = sensor_images(sensor, images)
        seg3d, occ = model.forward_voxel(raw)
        pred3d = seg3d.argmax(1)
        update_voxel_confusion(conf3d, pred3d, target, valid, num_classes=num_classes, ignore_index=ignore_index)
        occ_target = ((target != ignore_index) & valid).long()
        occ_pos += int(occ_target[valid].sum().detach().cpu())
        occ_valid += int(valid.sum().detach().cpu())
        occ_pred = occ.argmax(1)
        update_voxel_confusion(conf_occ, occ_pred, occ_target, valid, num_classes=2, ignore_index=ignore_index)
    m3d = summarize_voxel_confusion(conf3d)
    occ_neg = max(0, occ_valid - occ_pos)
    occ_degenerate = occ_valid == 0 or occ_pos == 0 or occ_neg == 0
    mocc = summarize_voxel_confusion(conf_occ) if not occ_degenerate else {}
    return {
        "mIoU_3d": float(m3d["mIoU"]),
        "voxel_acc_3d": float(m3d["voxel_acc"]),
        "semantic_occupied_IoU": float(m3d["occupied_IoU"]),
        "occupancy_mIoU": None if occ_degenerate else float(mocc["mIoU"]),
        "occupancy_IoU": None if occ_degenerate else float(mocc["occupied_IoU"]),
        "occupancy_degenerate": bool(occ_degenerate),
        "occupancy_positive_fraction": float(occ_pos / max(occ_valid, 1)),
        "occupancy_valid_voxels": int(occ_valid),
        "occupancy_positive_voxels": int(occ_pos),
        "occupancy_negative_voxels": int(occ_neg),
    }


def plot_unified_curves(ckpt_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    csv_path = os.path.join(ckpt_dir, "metrics_log.csv")
    if not os.path.isfile(csv_path):
        return
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    os.makedirs(os.path.join(ckpt_dir, "plots"), exist_ok=True)

    def vals(key: str) -> list[float]:
        out = []
        for row in rows:
            text = str(row.get(key, "")).strip()
            out.append(float(text) if text else float("nan"))
        return out

    xs = vals("epoch")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for key, label in [
        ("mIoU_2d", "2D semantic mIoU"),
        ("mIoU_3d", "3D semantic mIoU"),
        ("occupancy_IoU", "occupancy IoU"),
        ("unified_score", "reported unified score"),
    ]:
        if key in rows[0]:
            ax.plot(xs, vals(key), marker="o", linewidth=1.8, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation score")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(ckpt_dir, "plots", "unified_validation_curves.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for key, label in [("train_loss", "total"), ("train_loss_2d", "2D"), ("train_loss_3d", "3D"), ("train_loss_occ", "occupancy")]:
        if key in rows[0]:
            ax.plot(xs, vals(key), marker="o", linewidth=1.5, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(ckpt_dir, "plots", "unified_training_curves.png"), dpi=200)
    plt.close(fig)


@torch.no_grad()
def save_unified_visuals(sensor, model, loaders, device, cfg: Dict[str, Any], epoch: int, ckpt_dir: str) -> None:
    model.eval()
    sensor.eval()
    ignore_index = int(cfg["model"].get("ignore_index", 255))
    num_3d = int(cfg["model"].get("num_3d_classes", cfg["model"].get("num_classes", 19)))
    viz_dir = os.path.join(ckpt_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    try:
        rgb, target = next(iter(loaders["val_2d"]))
        rgb_dev = rgb.to(device)
        raw, stages = sensor_images(sensor, rgb_dev, return_stages=True)
        logits = model.forward_2d(raw, out_size=target.shape[-2:])
        pred = logits.argmax(1).detach().cpu()
        stages_cpu = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in stages.items()}
        vis_batch_with_stages(
            rgb=rgb.detach().cpu(),
            pred=pred,
            target=target.detach().cpu(),
            stages=stages_cpu,
            out_png=os.path.join(viz_dir, f"epoch{epoch:03d}_2d_sensor_task_story.png"),
            ignore_index=ignore_index,
            max_items=int(cfg["train"].get("viz_items", 3)),
            diagnostic_stages=True,
            panel_caption="Unified optics-sensor-task frontend: physical stages, task tokens, GT, and prediction",
        )
    except Exception as exc:
        print(f"[unified viz] skipped 2D visualization: {exc}", flush=True)

    try:
        batch = next(iter(loaders["val_vox"]))
        images = batch["images"].to(device)
        target = batch["occupancy"].to(device)
        valid = batch["valid_mask"].to(device)
        raw = sensor_images(sensor, images)
        seg3d, _ = model.forward_voxel(raw)
        pred3d = seg3d.argmax(1)
        save_occupancy_viz(
            images=images.detach().cpu(),
            pred=pred3d.detach().cpu(),
            target=target.detach().cpu(),
            valid=valid.detach().cpu(),
            out_path=os.path.join(viz_dir, f"epoch{epoch:03d}_3d_semantic_bev_story.png"),
            num_classes=num_3d,
            ignore_index=ignore_index,
            title="Unified 3D semantic segmentation from the shared camera frontend",
        )
    except Exception as exc:
        print(f"[unified viz] skipped 3D visualization: {exc}", flush=True)


def train(cfg: Dict[str, Any]) -> None:
    set_seed(int(cfg.get("seed", 0)))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    ckpt_dir = cfg["train"].get("ckpt_dir", "./runs/unified_multitask")
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "resolved_config.yaml"), "w") as f:
        yaml.safe_dump({**cfg, "device": str(device)}, f, sort_keys=False)

    loaders = build_loaders(cfg)
    if str(cfg.get("pipeline", {}).get("input", "sensor")).lower() == "rgb":
        sensor = PassThroughSensor({**cfg, "device": device}).to(device)
    else:
        sensor = CoDesignSensor({**cfg, "device": str(device), "data": cfg.get("data_2d", {})}).to(device)
    in_ch = int(getattr(sensor, "output_channels", 3))
    model = UnifiedRaw2TaskModel(
        in_channels=in_ch,
        num_2d_classes=int(cfg["model"].get("num_2d_classes", cfg["model"].get("num_classes", 19))),
        num_3d_classes=int(cfg["model"].get("num_3d_classes", cfg["model"].get("num_classes", 19))),
        grid_shape=tuple(cfg["data_voxel"].get("grid_shape", [16, 64, 128])),
        width=int(cfg["model"].get("width", 64)),
    ).to(device)
    _export_sensor_design(sensor, os.path.join(ckpt_dir, "camera_design_initial.json"))
    _save_sensor_preview(sensor, os.path.join(ckpt_dir, "design_preview"), "initial")

    base_lr = float(cfg["train"].get("lr", 2e-4))
    sensor_lr = base_lr * float(cfg["train"].get("sensor_lr_mult", 0.1))
    params = [{"params": model.parameters(), "lr": base_lr}]
    sensor_params = [p for p in sensor.parameters() if p.requires_grad]
    if sensor_params:
        params.append({"params": sensor_params, "lr": sensor_lr})
    optim = torch.optim.AdamW(params, weight_decay=float(cfg["train"].get("weight_decay", 1e-4)))
    amp_enabled = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    ignore_index = int(cfg["model"].get("ignore_index", 255))
    n2d = int(cfg["model"].get("num_2d_classes", cfg["model"].get("num_classes", 19)))
    n3d = int(cfg["model"].get("num_3d_classes", cfg["model"].get("num_classes", 19)))
    lambda_2d = float(cfg["train"].get("lambda_2d", 1.0))
    lambda_3d = float(cfg["train"].get("lambda_3d", 1.0))
    lambda_occ = float(cfg["train"].get("lambda_occupancy", 0.25))
    sensor_reg_weight = float(cfg["train"].get("sensor_reg_weight", 5e-4))
    steps_per_epoch = int(cfg["train"].get("steps_per_epoch", max(len(loaders["train_2d"]), len(loaders["train_vox"]))))
    epochs = int(cfg["train"].get("epochs", 10))
    voxel_iter = _cycle(loaders["train_vox"])
    csv_path = os.path.join(ckpt_dir, "metrics_log.csv")
    best = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        sensor.train()
        two_d_iter = _cycle(loaders["train_2d"])
        running = {"loss": 0.0, "loss_2d": 0.0, "loss_3d": 0.0, "loss_occ": 0.0}
        t0 = time.time()
        for step in range(1, steps_per_epoch + 1):
            rgb, target_2d = next(two_d_iter)
            vox = next(voxel_iter)
            rgb = rgb.to(device)
            target_2d = target_2d.to(device)
            images = vox["images"].to(device)
            target_3d = vox["occupancy"].to(device)
            valid = vox["valid_mask"].to(device)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                raw2d = sensor_images(sensor, rgb)
                logits2d = model.forward_2d(raw2d, out_size=target_2d.shape[-2:])
                loss_2d = F.cross_entropy(logits2d, target_2d, ignore_index=ignore_index)

                raw_vox = sensor_images(sensor, images)
                logits3d, logits_occ = model.forward_voxel(raw_vox)
                masked_3d = target_3d.clone()
                masked_3d[~valid] = ignore_index
                loss_3d = F.cross_entropy(logits3d, masked_3d, ignore_index=ignore_index)
                occ_target = ((target_3d != ignore_index) & valid).long()
                masked_occ = occ_target.clone()
                masked_occ[~valid] = ignore_index
                valid_occ_values = occ_target[valid]
                if valid_occ_values.numel() == 0 or torch.unique(valid_occ_values).numel() < 2:
                    loss_occ = logits_occ.new_zeros(())
                else:
                    loss_occ = F.cross_entropy(logits_occ, masked_occ, ignore_index=ignore_index)
                loss = lambda_2d * loss_2d + lambda_3d * loss_3d + lambda_occ * loss_occ
                if hasattr(sensor, "regularization"):
                    loss = loss + sensor_reg_weight * sensor.regularization()["total"]

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if float(cfg["train"].get("grad_clip_norm", 0.0)) > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(sensor.parameters()), float(cfg["train"]["grad_clip_norm"]))
            scaler.step(optim)
            scaler.update()
            running["loss"] += float(loss.detach())
            running["loss_2d"] += float(loss_2d.detach())
            running["loss_3d"] += float(loss_3d.detach())
            running["loss_occ"] += float(loss_occ.detach())
            if step % int(cfg["train"].get("log_interval", 20)) == 0:
                elapsed = time.time() - t0
                eta_epoch = elapsed / max(1, step) * max(0, steps_per_epoch - step)
                eta_total = eta_epoch + (epochs - epoch) * (elapsed / max(1, step) * steps_per_epoch)
                print(
                    f"UNIFIED TRAIN epoch {epoch}/{epochs} {_progress_bar(step, steps_per_epoch)} "
                    f"step {step}/{steps_per_epoch} ETA_epoch {_format_seconds(eta_epoch)} "
                    f"ETA_total {_format_seconds(eta_total)} loss {running['loss']/step:.4f} "
                    f"L2D {running['loss_2d']/step:.4f} L3D {running['loss_3d']/step:.4f} "
                    f"LOcc {running['loss_occ']/step:.4f}",
                    flush=True,
                )

        metrics = {}
        metrics.update(evaluate_2d(sensor, model, loaders["val_2d"], device, ignore_index, n2d))
        metrics.update(evaluate_voxel(sensor, model, loaders["val_vox"], device, ignore_index, n3d))
        metrics["epoch"] = epoch
        metrics["train_loss"] = running["loss"] / max(1, steps_per_epoch)
        metrics["train_loss_2d"] = running["loss_2d"] / max(1, steps_per_epoch)
        metrics["train_loss_3d"] = running["loss_3d"] / max(1, steps_per_epoch)
        metrics["train_loss_occ"] = running["loss_occ"] / max(1, steps_per_epoch)
        score_terms = [float(metrics["mIoU_2d"]), float(metrics["mIoU_3d"])]
        if metrics.get("occupancy_IoU") is not None:
            score_terms.append(float(metrics["occupancy_IoU"]))
        score = float(sum(score_terms) / max(1, len(score_terms)))
        metrics["unified_score"] = score
        print("UNIFIED EVAL " + json.dumps(metrics, indent=2), flush=True)
        with open(os.path.join(ckpt_dir, f"metrics_epoch{epoch}.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        write_header = not os.path.isfile(csv_path)
        with open(csv_path, "a", newline="") as f:
            fields = [
                "epoch",
                "train_loss",
                "train_loss_2d",
                "train_loss_3d",
                "train_loss_occ",
                "mIoU_2d",
                "pixel_acc_2d",
                "mIoU_3d",
                "voxel_acc_3d",
                "semantic_occupied_IoU",
                "occupancy_mIoU",
                "occupancy_IoU",
                "occupancy_degenerate",
                "occupancy_positive_fraction",
                "occupancy_valid_voxels",
                "occupancy_positive_voxels",
                "occupancy_negative_voxels",
                "unified_score",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow({k: metrics.get(k, "") for k in fields})
        plot_unified_curves(ckpt_dir)
        if epoch % int(cfg["train"].get("viz_interval", 1)) == 0:
            save_unified_visuals(sensor, model, loaders, device, cfg, epoch, ckpt_dir)
        export_design_every = int(cfg["train"].get("export_design_every", 1))
        if export_design_every > 0 and epoch % export_design_every == 0:
            _export_sensor_design(sensor, os.path.join(ckpt_dir, f"camera_design_epoch{epoch}.json"))
            _save_sensor_preview(sensor, os.path.join(ckpt_dir, "design_preview"), f"epoch{epoch}")
        improved = score > best
        if improved:
            best = score
        payload = {
            "sensor": sensor.state_dict(),
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "cfg": {**cfg, "device": str(device)},
            "epoch": epoch,
            "best": best,
        }
        torch.save(payload, os.path.join(ckpt_dir, "last.pt"))
        if improved:
            torch.save(payload, os.path.join(ckpt_dir, "best.pt"))
            _export_sensor_design(sensor, os.path.join(ckpt_dir, "camera_design_best.json"))
            _save_sensor_preview(sensor, os.path.join(ckpt_dir, "design_preview"), "best")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
