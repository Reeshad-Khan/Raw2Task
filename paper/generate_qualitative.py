"""Generate paper-ready qualitative comparison figures for the spectral CFA paper.

Compares: Fixed Camera | No Optics (tile-2) | Tile-3 | Tile-4
on the same validation images (KITTI-360 or ACDC).

Images are ranked by how much the best sensor config (no_optics) outperforms
fixed_camera on rare/dynamic classes — ensuring the most compelling examples.

Usage (run on Newton after all jobs complete):

  # KITTI-360
  python paper/generate_qualitative.py \\
      --spectral-dir ~/Raw2Task/runs/kitti360_sfb4_spectral \\
      --base-dir     ~/Raw2Task/runs/kitti360_sfb4 \\
      --out-dir      ~/Raw2Task/paper/figures/qualitative_kitti \\
      --dataset kitti360 --n-figures 6

  # ACDC
  python paper/generate_qualitative.py \\
      --spectral-dir ~/Raw2Task/runs/acdc_sfb4_spectral \\
      --base-dir     ~/Raw2Task/runs/acdc_sfb4 \\
      --out-dir      ~/Raw2Task/paper/figures/qualitative_acdc \\
      --dataset acdc --n-figures 6
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

# ── Cityscapes trainId palette (19 classes) ───────────────────────────────────
PALETTE_19 = [
    (128, 64, 128), (244, 35, 232), (70,  70,  70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170,  30), (220, 220,   0),
    (107, 142,  35), (152, 251, 152), (70,  130, 180), (220,  20,  60),
    (255,   0,   0), (  0,   0, 142), (  0,   0,  70), (  0,  60, 100),
    (  0,  80, 100), (  0,   0, 230), (119,  11,  32),
]

# Dynamic/rare classes where CFA matters most
RARE_CLASSES = {11, 12, 13, 14, 15, 16, 17, 18}

# Experiment definitions for the spectral paper
SPECTRAL_EXPS = [
    # (key,                    dir_suffix,          display_label)
    ("fixed_camera",  "ablate_fixed_camera_{suffix}_seed0", "Fixed Camera"),
    ("no_optics",     "ablate_no_optics_{suffix}_seed0",    "No Optics (Tile-2) ★"),
    ("tile3",         "cfa_tile3_{suffix}_seed0",           "Tile-3"),
    ("tile4",         "cfa_tile4_{suffix}_seed0",           "Tile-4"),
]


def _suffix(dataset: str) -> str:
    return "sfb4" if dataset == "kitti360" else "acdc"


def colorize(pred: np.ndarray, palette: list, ignore: int = 255) -> np.ndarray:
    H, W = pred.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cls, color in enumerate(palette):
        rgb[pred == cls] = color
    rgb[pred == ignore] = (10, 10, 10)
    return rgb


def tensor_to_float(t: torch.Tensor) -> np.ndarray:
    """CHW float [0,1] → HWC float."""
    return t.permute(1, 2, 0).cpu().float().clamp(0, 1).numpy()


def find_checkpoint(root: str, exp_pattern: str) -> Optional[str]:
    """Find best.pt or last.pt in the directory matching exp_pattern."""
    dirs = sorted(glob.glob(os.path.join(root, exp_pattern)))
    if not dirs:
        return None
    run_dir = dirs[0]
    for name in ("best.pt", "last.pt"):
        p = os.path.join(run_dir, name)
        if os.path.isfile(p):
            return p
    return None


def score_image(
    pred_best: np.ndarray,
    pred_baseline: np.ndarray,
    gt: np.ndarray,
    ignore: int = 255,
) -> float:
    """Score = fraction of rare-class pixels where best is correct and baseline is wrong."""
    valid = gt != ignore
    rare = np.isin(gt, list(RARE_CLASSES)) & valid
    if rare.sum() == 0:
        return 0.0
    correct_best  = (pred_best     == gt) & rare
    wrong_baseline = (pred_baseline != gt) & rare
    return float((correct_best & wrong_baseline).sum()) / max(float(rare.sum()), 1.0)


@torch.no_grad()
def generate_qualitative(
    spectral_dir: str,
    base_dir: str,
    out_dir: str,
    dataset: str,
    n_figures: int,
    max_scan_batches: int,
) -> None:
    try:
        from raw2task.eval_robustness import _build_from_checkpoint
        from raw2task.train_extended import align_logits_to_labels
    except ImportError as e:
        print(f"ERROR: raw2task not importable — run from the Raw2Task repo root: {e}")
        return

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suf = _suffix(dataset)

    # ── load all available models ──────────────────────────────────────────────
    systems: Dict[str, tuple] = {}
    for key, pattern, label in SPECTRAL_EXPS:
        # Try spectral dir first, then base dir
        ckpt = None
        for root in [spectral_dir, base_dir]:
            if root:
                pat = pattern.format(suffix=suf)
                ckpt = find_checkpoint(root, pat)
                if ckpt:
                    break

        if ckpt is None:
            print(f"  [skip] {label}: no checkpoint found")
            continue

        print(f"  Loading {label} from {ckpt} ...")
        try:
            cfg, sensor, model, val_loader, num_classes = _build_from_checkpoint(ckpt, device)
            sensor.eval()
            model.eval()
            ignore_index = int(cfg.get("model", {}).get("ignore_index", 255))
            systems[key] = (sensor, model, val_loader, num_classes, ignore_index, cfg, label)
        except Exception as ex:
            print(f"  [warn] Failed to load {label}: {ex}")

    if not systems:
        print("ERROR: no models loaded — cannot generate qualitative figures.")
        return

    # Use the no_optics loader (or first available) as canonical image source
    ref_key = "no_optics" if "no_optics" in systems else next(iter(systems))
    _, _, val_loader, num_classes, ignore_index, cfg, _ = systems[ref_key]
    palette = PALETTE_19[:num_classes]

    print(f"\nScanning up to {max_scan_batches} batches for best images...")
    candidates = []

    for batch_idx, (imgs, labels) in enumerate(val_loader, 1):
        if batch_idx > max_scan_batches:
            break
        imgs_dev   = imgs.to(device)
        labels_dev = labels.to(device)

        preds: Dict[str, np.ndarray] = {}
        for key, (sensor, model, _, _, ig, _, _) in systems.items():
            raw    = sensor(imgs_dev)
            logits = align_logits_to_labels(model(raw), labels_dev)
            preds[key] = logits.argmax(1).cpu().numpy()

        B = imgs.shape[0]
        for i in range(B):
            gt_np      = labels[i].numpy()
            best_pred  = preds.get("no_optics", preds[ref_key])[i]
            fixed_pred = preds.get("fixed_camera", best_pred)[i]
            score = score_image(best_pred, fixed_pred, gt_np, ignore=ignore_index)
            candidates.append((
                score,
                imgs[i],
                gt_np,
                {k: v[i] for k, v in preds.items()},
                batch_idx,
                i,
            ))

    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = candidates[:n_figures]
    print(f"Selected {len(selected)} images (best score={selected[0][0]:.4f}).")

    # ── render comparison panels ───────────────────────────────────────────────
    col_defs = [("input", "Input RGB"), ("gt", "Ground Truth")]
    for key, _, label in SPECTRAL_EXPS:
        if key in systems:
            col_defs.append((key, systems[key][6]))  # label from systems tuple

    n_cols = len(col_defs)

    for rank, (score, img_t, gt_np, pred_dict, batch_idx, item_idx) in enumerate(selected, 1):
        img_np = tensor_to_float(img_t)
        gt_color = colorize(gt_np, palette, ignore=ignore_index).astype(np.float32) / 255.0

        fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3.2))
        if n_cols == 1:
            axes = [axes]

        for ax, (col_key, col_label) in zip(axes, col_defs):
            if col_key == "input":
                ax.imshow(img_np, interpolation="bilinear")
            elif col_key == "gt":
                ax.imshow(gt_color, interpolation="nearest")
            else:
                pred_np = pred_dict.get(col_key)
                if pred_np is None:
                    ax.set_visible(False)
                    continue
                pred_color = colorize(pred_np, palette, ignore=ignore_index).astype(np.float32) / 255.0
                ax.imshow(pred_color, interpolation="nearest")

            ax.set_title(col_label, fontsize=8, fontweight="bold", pad=3)
            ax.axis("off")

        fig.suptitle(
            f"{dataset.upper()} Qualitative — rank {rank}  (best-config gain={score:.4f})",
            fontsize=8, y=1.01,
        )
        fig.tight_layout(pad=0.3)

        out_png = os.path.join(out_dir, f"qual_{dataset}_rank{rank:02d}_b{batch_idx}_i{item_idx}.png")
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Wrote {out_png}")

    # ── legend figure ──────────────────────────────────────────────────────────
    _save_legend(palette, num_classes, out_dir, dataset)
    print(f"\nDone. {len(selected)} qualitative figures saved to {out_dir}/")


def _save_legend(palette, n_classes, out_dir, dataset):
    """Save a colour legend for the segmentation palette."""
    # Cityscapes class names (trainId order)
    names = [
        "road", "sidewalk", "building", "wall", "fence", "pole",
        "traffic light", "traffic sign", "vegetation", "terrain",
        "sky", "person", "rider", "car", "truck", "bus",
        "train", "motorcycle", "bicycle",
    ][:n_classes]

    n = len(names)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, ax = plt.subplots(figsize=(cols * 2.4, rows * 0.35 + 0.3))
    ax.axis("off")
    patches = [
        mpatches.Patch(facecolor=np.array(palette[i]) / 255.0, label=names[i])
        for i in range(n)
    ]
    ax.legend(handles=patches, ncol=cols, loc="center", fontsize=7,
              frameon=False, handlelength=1.5, handletextpad=0.5, columnspacing=1.0)
    fig.tight_layout()
    out = os.path.join(out_dir, f"legend_{dataset}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote legend: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spectral-dir", required=True,
                    help="Root dir of spectral experiments (kitti360_sfb4_spectral or acdc_sfb4_spectral)")
    ap.add_argument("--base-dir",     default=None,
                    help="Root dir of baseline experiments (for fixed_camera checkpoint)")
    ap.add_argument("--out-dir",      required=True, help="Output directory for PNG files")
    ap.add_argument("--dataset",      choices=["kitti360", "acdc"], required=True)
    ap.add_argument("--n-figures",    type=int, default=6,
                    help="Number of comparison figures to generate")
    ap.add_argument("--max-scan-batches", type=int, default=100,
                    help="Max validation batches to scan when selecting best images")
    args = ap.parse_args()

    generate_qualitative(
        spectral_dir=args.spectral_dir,
        base_dir=args.base_dir,
        out_dir=args.out_dir,
        dataset=args.dataset,
        n_figures=args.n_figures,
        max_scan_batches=args.max_scan_batches,
    )


if __name__ == "__main__":
    main()
