#!/usr/bin/env python3
import os
import glob
import json
import argparse
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

# Reuse training components from the active pipeline
from deeplens.projects.raw2task.train_extended import (
    CoDesignSensor,
    build_seg_model,
    PALETTE_19,
    colorize_mask,
    vis_batch_with_stages,
    update_confusion,
    iou_from_confusion,
    imagewise_miou,
)

# -------------------------
# Simple "external dataset" loader
# -------------------------
class ExternalDrivingDataset(Dataset):
    """
    Folder structure expected (minimum):
      root/images/*.png (or jpg)
    Optional GT:
      root/masks/*.png   (with training ids 0..18 and ignore=255)

    If no masks, we run qualitative-only.
    """
    def __init__(self, root: str, img_glob: str = "*.png", mask_ext: str = ".png"):
        self.root = root
        self.img_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")

        # allow png/jpg
        paths = sorted(glob.glob(os.path.join(self.img_dir, img_glob)))
        if len(paths) == 0:
            # fallback: jpg
            paths = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg"))) + \
                    sorted(glob.glob(os.path.join(self.img_dir, "*.jpeg")))
        if len(paths) == 0:
            raise FileNotFoundError(f"No images found under: {self.img_dir}")

        self.img_paths = paths
        self.has_masks = os.path.isdir(self.mask_dir)

        self.mask_ext = mask_ext

    def __len__(self):
        return len(self.img_paths)

    def _load_rgb(self, p: str) -> torch.Tensor:
        img = Image.open(p).convert("RGB")
        x = torch.from_numpy(np.array(img)).float() / 255.0  # (H,W,3)
        x = x.permute(2, 0, 1).contiguous()                  # (3,H,W)
        return x

    def _load_mask(self, img_path: str) -> torch.Tensor:
        # mask filename assumed to match image stem
        stem = os.path.splitext(os.path.basename(img_path))[0]
        mp = os.path.join(self.mask_dir, stem + self.mask_ext)
        if not os.path.isfile(mp):
            # if missing mask, treat as no-mask sample
            return torch.full((1, 1), 255, dtype=torch.long)
        m = Image.open(mp)
        m = torch.from_numpy(np.array(m)).long()
        if m.ndim == 3:
            # if RGB masks accidentally, take channel 0
            m = m[..., 0]
        return m

    def __getitem__(self, idx):
        ip = self.img_paths[idx]
        rgb = self._load_rgb(ip)
        if self.has_masks:
            mask = self._load_mask(ip)  # (H,W) long
        else:
            mask = None
        return rgb, mask, ip


# -------------------------
# Utilities
# -------------------------
def find_best_ckpt(ckpt_dir: str) -> str:
    # Prefer best_ep*.pt, otherwise last.pt
    best = sorted(glob.glob(os.path.join(ckpt_dir, "best_ep*.pt")))
    if best:
        return best[-1]
    last = os.path.join(ckpt_dir, "last.pt")
    if os.path.isfile(last):
        return last
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir} (expected best_ep*.pt or last.pt).")

def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    if "cfg" not in ckpt:
        raise ValueError("Checkpoint missing 'cfg'. Use the best_ep*.pt saved by your training script.")
    return ckpt

def ensure_outdir(p: str):
    os.makedirs(p, exist_ok=True)

@dataclass
class EvalResults:
    dataset: str
    num_images: int
    pixel_acc: Optional[float]
    miou: Optional[float]
    image_miou: Optional[float]
    per_class_iou: Optional[List[float]]


# -------------------------
# Core evaluation
# -------------------------
@torch.no_grad()
def eval_on_dataset(
    dataset_name: str,
    ds_root: str,
    sensor: nn.Module,
    model: nn.Module,
    device: torch.device,
    out_dir: str,
    batch_size: int,
    num_workers: int,
    ignore_index: int = 255,
    num_classes: int = 19,
    viz_every: int = 25,
    max_viz_batches: int = 999999,
):
    ensure_outdir(out_dir)
    viz_dir = os.path.join(out_dir, "viz")
    ensure_outdir(viz_dir)

    ds = ExternalDrivingDataset(ds_root, img_glob="*.png")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    sensor.eval()
    model.eval()

    # if no masks in dataset: qualitative-only
    has_masks = ds.has_masks

    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_correct = 0
    total_labeled = 0
    image_miou_sum = 0.0
    image_miou_count = 0

    rows = []  # per-image summary table

    for it, batch in enumerate(dl, 1):
        rgb, mask, paths = batch
        rgb = rgb.to(device)

        raw, stages = sensor(rgb, return_stages=True)
        logits = model(raw)
        pred = logits.argmax(1)  # (B,H,W)

        # Save viz
        if (it % viz_every == 0) and (it <= max_viz_batches):
            out_png = os.path.join(viz_dir, f"{dataset_name}_step{it:05d}.png")
            if has_masks:
                # stack masks into (B,H,W)
                gt = torch.stack(mask, dim=0).to(device) if isinstance(mask, list) else mask.to(device)
                vis_batch_with_stages(rgb=rgb, pred=pred, target=gt, stages=stages,
                                      out_png=out_png, palette=PALETTE_19, ignore_index=ignore_index, max_items=4)
            else:
                # no GT -> create a dummy GT just for function signature (won't be meaningful)
                dummy = torch.full_like(pred, ignore_index)
                vis_batch_with_stages(rgb=rgb, pred=pred, target=dummy, stages=stages,
                                      out_png=out_png, palette=PALETTE_19, ignore_index=ignore_index, max_items=4)

        # Metrics if GT exists
        if has_masks:
            # mask comes as list of tensors if default collate fails; handle both
            if isinstance(mask, list):
                gt = torch.stack(mask, dim=0).to(device)
            else:
                gt = mask.to(device)

            valid = (gt != ignore_index)
            if valid.any():
                correct = (pred[valid] == gt[valid]).sum().item()
                labeled = valid.sum().item()
                total_correct += correct
                total_labeled += labeled

            update_confusion(conf, pred, gt, num_classes=num_classes, ignore_index=ignore_index)
            im = imagewise_miou(pred, gt, num_classes=num_classes, ignore_index=ignore_index)
            image_miou_sum += im * pred.shape[0]
            image_miou_count += pred.shape[0]

            # per-image row (simple)
            for b in range(pred.shape[0]):
                rows.append({"dataset": dataset_name, "path": paths[b], "image_mIoU": float(im)})

        else:
            for b in range(pred.shape[0]):
                rows.append({"dataset": dataset_name, "path": paths[b]})

    # finalize metrics
    if has_masks:
        pixel_acc = float(total_correct / max(total_labeled, 1))
        miou, per_class_iou = iou_from_confusion(conf)
        image_miou = float(image_miou_sum / max(image_miou_count, 1))
        per_class_iou = [float(x) if np.isfinite(x) else float("nan") for x in per_class_iou]

        np.save(os.path.join(out_dir, f"confusion_{dataset_name}.npy"), conf)

        metrics = {
            "dataset": dataset_name,
            "num_images": len(ds),
            "pixel_acc": pixel_acc,
            "mIoU": float(miou),
            "image_mIoU": image_miou,
            "per_class_IoU": per_class_iou,
        }
    else:
        metrics = {
            "dataset": dataset_name,
            "num_images": len(ds),
            "pixel_acc": None,
            "mIoU": None,
            "image_mIoU": None,
            "per_class_IoU": None,
            "note": "No masks found under root/masks; produced qualitative outputs only.",
        }

    with open(os.path.join(out_dir, f"metrics_{dataset_name}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # per-image table
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"per_image_{dataset_name}.csv"), index=False)

    return metrics


def make_summary_tables_and_plots(all_metrics: List[dict], out_root: str):
    ensure_outdir(out_root)

    # Summary table
    rows = []
    for m in all_metrics:
        rows.append({
            "dataset": m["dataset"],
            "num_images": m["num_images"],
            "pixel_acc": m.get("pixel_acc"),
            "mIoU": m.get("mIoU"),
            "image_mIoU": m.get("image_mIoU"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_root, "summary_metrics.csv"), index=False)

    # Plots (skip if no metrics)
    df_num = df.dropna(subset=["mIoU"])
    if len(df_num) > 0:
        # mIoU bar
        plt.figure()
        plt.bar(df_num["dataset"], df_num["mIoU"])
        plt.xticks(rotation=20, ha="right")
        plt.ylabel("mIoU")
        plt.tight_layout()
        plt.savefig(os.path.join(out_root, "plot_miou_by_dataset.png"), dpi=200)
        plt.close()

        # pixel acc bar
        plt.figure()
        plt.bar(df_num["dataset"], df_num["pixel_acc"])
        plt.xticks(rotation=20, ha="right")
        plt.ylabel("Pixel Accuracy")
        plt.tight_layout()
        plt.savefig(os.path.join(out_root, "plot_pixelacc_by_dataset.png"), dpi=200)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True, help="Training run directory containing best_ep*.pt (or last.pt)")
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="List of dataset specs: name=PATH (expects PATH/images/*.png and optional PATH/masks/*.png)")
    ap.add_argument("--out_root", required=True, help="Where to write results")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--viz_every", type=int, default=25)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu)

    ckpt_path = find_best_ckpt(args.ckpt_dir)
    ckpt = load_checkpoint(ckpt_path, device=device)
    cfg = ckpt["cfg"]

    # Build models
    sensor = CoDesignSensor(cfg).to(device)
    model = build_seg_model(cfg, in_channels=1, num_classes=int(cfg.get("model", {}).get("num_classes", 19))).to(device)

    # Load weights
    sensor.load_state_dict(ckpt["sensor"], strict=True)
    model.load_state_dict(ckpt["model"], strict=True)

    ignore_index = int(cfg.get("model", {}).get("ignore_index", 255))
    num_classes  = int(cfg.get("model", {}).get("num_classes", 19))

    ensure_outdir(args.out_root)
    all_metrics = []

    for spec in args.datasets:
        if "=" not in spec:
            raise ValueError(f"Bad dataset spec '{spec}'. Use name=/path/to/root")
        name, path = spec.split("=", 1)
        out_dir = os.path.join(args.out_root, name)
        m = eval_on_dataset(
            dataset_name=name,
            ds_root=path,
            sensor=sensor,
            model=model,
            device=device,
            out_dir=out_dir,
            batch_size=args.batch,
            num_workers=args.workers,
            ignore_index=ignore_index,
            num_classes=num_classes,
            viz_every=args.viz_every,
        )
        all_metrics.append(m)

    # Summary + plots
    make_summary_tables_and_plots(all_metrics, args.out_root)
    with open(os.path.join(args.out_root, "all_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("Done. Results written to:", args.out_root)


if __name__ == "__main__":
    main()
