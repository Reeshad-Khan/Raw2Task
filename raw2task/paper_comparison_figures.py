"""Generate paper-ready side-by-side qualitative comparison figures.

For each selected validation image, renders:
  Input RGB | GT | RGB-baseline | No-Optics | Fixed-Camera | Co-Design

Images are ranked by how much co-design gains over the rgb baseline on
rare/thin classes, so the most visually compelling examples come first.

Usage (on Newton after training completes):
    python -m raw2task.paper_comparison_figures \\
        --runs-dir ./runs/kitti360_sfb4 \\
        --out-dir  ./paper_figures/kitti360 \\
        --n-figures 20 \\
        --dataset kitti360

    python -m raw2task.paper_comparison_figures \\
        --runs-dir ./runs/acdc_sfb4 \\
        --out-dir  ./paper_figures/acdc \\
        --n-figures 20 \\
        --dataset acdc \\
        --per-condition
"""
from __future__ import annotations

import argparse
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from raw2task.eval_robustness import _build_from_checkpoint
from raw2task.train_extended import align_logits_to_labels, dense_task_palette

# ── Cityscapes trainId palette (19 classes) ───────────────────────────────────
PALETTE_19 = [
    (128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156), (190,153,153),
    (153,153,153), (250,170, 30), (220,220,  0), (107,142, 35), (152,251,152),
    ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
    (  0, 60,100), (  0, 80,100), (  0,  0,230), (119, 11, 32),
]

# Classes where sensor co-design most plausibly helps
RARE_CLASSES = {11, 12, 13, 14, 15, 16, 17, 18}  # person, rider, car, truck, bus, train, moto, bicycle

EXP_ORDER = ["rgb", "ablate_no_optics", "ablate_fixed_camera", "codesign"]
EXP_LABELS = {
    "rgb":                    "RGB baseline",
    "ablate_no_optics":       "No optics",
    "ablate_fixed_camera":    "Fixed camera",
    "codesign":               "Co-design (ours)",
}
HEADER_BG = (30, 30, 30)
HEADER_FG = (255, 255, 255)


def colorize(pred: np.ndarray, palette: List[Tuple[int,int,int]], ignore: int = 255) -> np.ndarray:
    H, W = pred.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cls, color in enumerate(palette):
        rgb[pred == cls] = color
    rgb[pred == ignore] = (0, 0, 0)
    return rgb


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """CHW float [0,1] → HWC uint8."""
    return (t.permute(1, 2, 0).cpu().float().clamp(0, 1).numpy() * 255).astype(np.uint8)


def add_header(img: np.ndarray, text: str, height: int = 28) -> np.ndarray:
    H, W = img.shape[:2]
    banner = np.full((height, W, 3), HEADER_BG, dtype=np.uint8)
    pil = Image.fromarray(banner)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    x = max(4, (W - (bbox[2] - bbox[0])) // 2)
    draw.text((x, 4), text, fill=HEADER_FG, font=font)
    return np.vstack([np.array(pil), img])


def score_image(pred_codesign: np.ndarray, pred_rgb: np.ndarray,
                gt: np.ndarray, ignore: int = 255) -> float:
    """Score = fraction of rare-class pixels where codesign is correct but rgb is not."""
    valid = gt != ignore
    rare_mask = np.isin(gt, list(RARE_CLASSES)) & valid
    if rare_mask.sum() == 0:
        return 0.0
    cd_correct = (pred_codesign == gt) & rare_mask
    rgb_wrong  = (pred_rgb      != gt) & rare_mask
    gain = (cd_correct & rgb_wrong).sum()
    return float(gain) / max(float(rare_mask.sum()), 1.0)


def find_best_checkpoint(exp_dir: str) -> Optional[str]:
    """Return best.pt if present, else last.pt."""
    for name in ("best.pt", "last.pt"):
        p = os.path.join(exp_dir, name)
        if os.path.isfile(p):
            return p
    return None


def find_exp_dir(runs_dir: str, exp_name: str, suffix: str) -> Optional[str]:
    """Find seed0 checkpoint dir for a given experiment."""
    patterns = [
        f"{runs_dir}/{exp_name}{suffix}_seed0_s0",
        f"{runs_dir}/{exp_name}{suffix}_seed0",
        f"{runs_dir}/{exp_name}{suffix}",
    ]
    for pat in patterns:
        dirs = sorted(glob.glob(pat))
        if dirs:
            return dirs[0]
    return None


@torch.no_grad()
def generate_figures(
    runs_dir: str,
    out_dir: str,
    n_figures: int,
    dataset: str,
    per_condition: bool,
    max_scan_batches: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    suffix = f"_{dataset}" if dataset not in ("kitti360",) else "_sfb4"
    if dataset == "kitti360":
        suffix = "_sfb4"
    elif dataset == "acdc":
        suffix = "_acdc"

    # ── load all available experiments ────────────────────────────────────────
    systems: Dict[str, Tuple] = {}
    for exp_key in EXP_ORDER:
        exp_dir = find_exp_dir(runs_dir, exp_key, suffix)
        if exp_dir is None:
            print(f"  [skip] {exp_key}: no run dir found in {runs_dir}")
            continue
        ckpt = find_best_checkpoint(exp_dir)
        if ckpt is None:
            print(f"  [skip] {exp_key}: no checkpoint in {exp_dir}")
            continue
        print(f"  loading {exp_key} from {ckpt} ...")
        cfg, sensor, model, val_loader, num_classes = _build_from_checkpoint(ckpt, device)
        sensor.eval(); model.eval()
        ignore_index = int(cfg.get("model", {}).get("ignore_index", 255))
        systems[exp_key] = (sensor, model, val_loader, num_classes, ignore_index, cfg)

    if "codesign" not in systems:
        print("ERROR: codesign checkpoint not found — cannot score images.")
        return
    if "rgb" not in systems:
        print("WARNING: rgb checkpoint not found — using codesign loader for reference images.")

    # Use codesign's loader as the canonical image source (same val set)
    _, _, val_loader, num_classes, ignore_index, cfg = systems["codesign"]
    palette = PALETTE_19[:num_classes]

    # ── score and collect candidates ─────────────────────────────────────────
    print(f"Scanning up to {max_scan_batches} batches for best images ...")
    candidates = []  # (score, img_np, gt_np, preds_dict, batch_idx, item_idx)

    for batch_idx, (imgs, labels) in enumerate(val_loader, 1):
        if batch_idx > max_scan_batches:
            break
        imgs_dev   = imgs.to(device)
        labels_dev = labels.to(device)

        preds: Dict[str, np.ndarray] = {}
        for exp_key, (sensor, model, _, _, _, _) in systems.items():
            raw    = sensor(imgs_dev)
            logits = align_logits_to_labels(model(raw), labels_dev)
            preds[exp_key] = logits.argmax(1).cpu().numpy()

        B = imgs.shape[0]
        for i in range(B):
            gt_np = labels[i].numpy()
            cd_pred = preds.get("codesign", preds[list(preds.keys())[0]])[i]
            rgb_pred = preds.get("rgb", cd_pred)[i]
            score = score_image(cd_pred, rgb_pred, gt_np, ignore=ignore_index)
            candidates.append((score, imgs[i], gt_np, {k: v[i] for k, v in preds.items()}, batch_idx, i))

    candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates[:n_figures]
    print(f"Selected {len(top)} images (top score={top[0][0]:.4f} if any).")

    # ── render panels ─────────────────────────────────────────────────────────
    col_keys = ["input"] + [k for k in EXP_ORDER if k in systems] + ["gt"]
    col_labels = {
        "input": "Input RGB",
        "gt":    "Ground Truth",
        **{k: EXP_LABELS[k] for k in EXP_ORDER},
    }

    for rank, (score, img_t, gt_np, pred_dict, batch_idx, item_idx) in enumerate(top, 1):
        img_np  = tensor_to_uint8(img_t)
        gt_color = colorize(gt_np, palette, ignore=ignore_index)

        panels = []
        for col in col_keys:
            if col == "input":
                arr = img_np
            elif col == "gt":
                arr = gt_color
            else:
                arr = colorize(pred_dict[col], palette, ignore=ignore_index)

            arr = add_header(arr, col_labels[col])
            panels.append(arr)

        # difference overlay: codesign correct, rgb wrong
        if "codesign" in pred_dict and "rgb" in pred_dict:
            cd_correct  = (pred_dict["codesign"] == gt_np) & (gt_np != ignore_index)
            rgb_wrong   = (pred_dict["rgb"]       != gt_np) & (gt_np != ignore_index)
            gain_mask   = cd_correct & rgb_wrong & np.isin(gt_np, list(RARE_CLASSES))
            diff = img_np.copy()
            diff[gain_mask] = (0, 255, 128)   # bright green = codesign gain
            loss_mask = (~cd_correct) & (pred_dict["rgb"] == gt_np) & (gt_np != ignore_index)
            diff[loss_mask] = (255, 80, 0)    # orange = codesign regressed
            diff = add_header(diff, "Co-design gain (green) / loss (orange)")
            panels.append(diff)

        row = np.concatenate(panels, axis=1)
        out_path = os.path.join(out_dir, f"rank{rank:03d}_score{score:.4f}_b{batch_idx}i{item_idx}.png")
        Image.fromarray(row).save(out_path)
        print(f"  saved {out_path}")

    # ── save per-class IoU comparison table ───────────────────────────────────
    _save_miou_table(systems, val_loader, device, num_classes, ignore_index, out_dir, max_scan_batches)
    print("Done.")


@torch.no_grad()
def _save_miou_table(systems, val_loader, device, num_classes, ignore_index, out_dir, max_batches):
    from raw2task.train_extended import update_confusion, iou_from_confusion
    CLASS_NAMES = [
        "road","sidewalk","building","wall","fence","pole",
        "traffic light","traffic sign","vegetation","terrain","sky",
        "person","rider","car","truck","bus","train","motorcycle","bicycle",
    ]

    results = {}
    for exp_key, (sensor, model, _, _, _, _) in systems.items():
        conf = np.zeros((num_classes, num_classes), dtype=np.int64)
        for batch_idx, (imgs, labels) in enumerate(val_loader, 1):
            if batch_idx > max_batches:
                break
            imgs_d   = imgs.to(device)
            labels_d = labels.to(device)
            logits   = align_logits_to_labels(model(sensor(imgs_d)), labels_d)
            pred     = logits.argmax(1).cpu().numpy()
            gt       = labels.numpy()
            valid    = gt != ignore_index
            for c in range(num_classes):
                for p in range(num_classes):
                    conf[c, p] += ((gt == c) & (pred == p) & valid).sum()
        per_cls = []
        for c in range(num_classes):
            tp = conf[c, c]
            fn = conf[c, :].sum() - tp
            fp = conf[:, c].sum() - tp
            denom = tp + fn + fp
            per_cls.append(float(tp) / max(float(denom), 1.0))
        results[exp_key] = per_cls

    # write CSV
    import csv
    csv_path = os.path.join(out_dir, "per_class_iou.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["class"] + [EXP_LABELS.get(k, k) for k in results]
        w.writerow(header)
        for i, name in enumerate(CLASS_NAMES[:num_classes]):
            row = [name] + [f"{results[k][i]:.4f}" for k in results]
            w.writerow(row)
        miou_row = ["mIoU"] + [f"{np.mean(results[k]):.4f}" for k in results]
        w.writerow(miou_row)
    print(f"  per-class IoU table → {csv_path}")

    # write a quick matplotlib bar chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(16, 5))
        x = np.arange(num_classes)
        width = 0.8 / max(len(results), 1)
        colors = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]
        for i, (exp_key, ious) in enumerate(results.items()):
            offset = (i - len(results) / 2 + 0.5) * width
            ax.bar(x + offset, ious, width, label=EXP_LABELS.get(exp_key, exp_key),
                   color=colors[i % len(colors)], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES[:num_classes], rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("IoU")
        ax.set_title("Per-class IoU comparison")
        ax.legend()
        ax.set_ylim(0, 1)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "per_class_iou.png"), dpi=150)
        plt.close(fig)
        print(f"  per-class IoU chart → {os.path.join(out_dir, 'per_class_iou.png')}")
    except Exception as e:
        print(f"  matplotlib chart skipped: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir",  required=True, help="e.g. ./runs/kitti360_sfb4")
    ap.add_argument("--out-dir",   required=True, help="output directory for figures")
    ap.add_argument("--n-figures", type=int, default=20, help="number of comparison panels to save")
    ap.add_argument("--dataset",   default="kitti360", choices=["kitti360", "acdc"])
    ap.add_argument("--per-condition", action="store_true", help="also generate per ACDC condition panels")
    ap.add_argument("--max-scan-batches", type=int, default=200,
                    help="how many val batches to scan for image selection")
    args = ap.parse_args()
    generate_figures(
        runs_dir=args.runs_dir,
        out_dir=args.out_dir,
        n_figures=args.n_figures,
        dataset=args.dataset,
        per_condition=args.per_condition,
        max_scan_batches=args.max_scan_batches,
    )


if __name__ == "__main__":
    main()
