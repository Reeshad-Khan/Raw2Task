import os
import argparse
import json
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

# Reuse everything from the training script
from raw2task.train_extended import (
    CoDesignSensor,
    build_seg_model,
    get_dataset,
    evaluate_seg,
    colorize_mask,
    PALETTE_19,
    _make_grid,
    _save_image,
)
from deeplens.utils import set_seed


def _prepare_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Eval] Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Eval] Using CPU")
    return device


def _load_checkpoint(path: str, device: torch.device) -> Dict:
    print(f"[Eval] Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    if "cfg" not in ckpt:
        raise RuntimeError(f"Checkpoint {path} missing 'cfg' field; "
                           f"please train with the latest train_extended.py.")
    return ckpt


def _build_model_from_ckpt(
    ckpt: Dict,
    device: torch.device,
    allow_missing_sensor: bool = False,
    allow_missing_model: bool = False,
):
    cfg = ckpt["cfg"]

    # Ensure deterministic-ish behaviour and a fresh device
    cfg.setdefault("seed", 0)
    set_seed(cfg["seed"])

    # Overwrite device info in cfg
    cfg["device"] = device
    cfg["num_gpus"] = 1

    # Build dataset (train loader is unused here)
    train_loader, val_loader, num_classes = get_dataset(cfg)

    sensor = CoDesignSensor(cfg).to(device)
    in_ch = 1
    model = build_seg_model(cfg, in_channels=in_ch, num_classes=num_classes).to(device)

    # Some UNetTiny versions might not define ignore_index internally
    model.ignore_index = int(cfg.get("model", {}).get("ignore_index", 255))

    # Load weights
    print("[Eval] Loading sensor/model weights from checkpoint.")

    # Sensor loading
    if allow_missing_sensor:
        print("[Eval] --allow-missing-sensor set: loading sensor with strict=False")
        sensor.load_state_dict(ckpt["sensor"], strict=False)
    else:
        try:
            sensor.load_state_dict(ckpt["sensor"], strict=True)
        except Exception as e:
            print("[Eval] sensor.load_state_dict strict=True failed:", e)
            print("[Eval] Retrying sensor.load_state_dict with strict=False (allowing missing/extra keys).")
            sensor.load_state_dict(ckpt["sensor"], strict=False)

    # Model loading
    if allow_missing_model:
        print("[Eval] --allow-missing-model set: loading model with strict=False")
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        try:
            model.load_state_dict(ckpt["model"], strict=True)
        except Exception as e:
            print("[Eval] model.load_state_dict strict=True failed:", e)
            print("[Eval] Retrying model.load_state_dict with strict=False (allowing missing/extra keys).")
            model.load_state_dict(ckpt["model"], strict=False)

    return cfg, sensor, model, val_loader, num_classes


def _save_metrics_json(metrics: Dict, out_dir: str, tag: str):
    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, f"metrics_{tag}.json")
    with open(jpath, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Eval] Saved metrics JSON: {jpath}")


def _append_summary_csv(all_results: List[Dict], out_root: str):
    """
    all_results: list of dicts with keys:
      - 'name', 'pixel_acc', 'mIoU', 'per_class_IoU'
    """
    import csv

    csv_path = os.path.join(out_root, "summary_metrics.csv")
    os.makedirs(out_root, exist_ok=True)
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            header = ["name", "pixel_acc", "mIoU"] + [f"IoU_{i}" for i in range(len(all_results[0]["per_class_IoU"]))]
            w.writerow(header)
        for res in all_results:
            row = [res["name"], res["pixel_acc"], res["mIoU"], *res["per_class_IoU"]]
            w.writerow(row)
    print(f"[Eval] Appended summary to: {csv_path}")


@torch.no_grad()
def _predict_single(sensor, model, img_t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    img_t: (3,H,W) tensor on CPU in [0,1]
    Returns pred: (H,W) long on CPU
    """
    sensor.eval()
    model.eval()
    img = img_t.unsqueeze(0).to(device)
    raw = sensor(img)           # (1,1,H,W)
    logits = model(raw)         # (1,K,H,W)
    pred = logits.argmax(1)[0].cpu()
    return pred


def _save_comparison_grids(
    ckpt_infos: List[Dict],
    ref_val_loader: DataLoader,
    device: torch.device,
    out_root: str,
    num_vis: int = 4,
):
    """
    ckpt_infos: list of dicts with fields:
       - 'name': model name
       - 'sensor': CoDesignSensor
       - 'model': UNetTiny
    ref_val_loader: we use its dataset for picking images
    """
    dataset = ref_val_loader.dataset
    N = len(dataset)
    if N == 0:
        print("[Compare] Empty validation dataset; skipping comparison grids.")
        return

    k = min(num_vis, N)
    indices = list(range(k))  # first K samples; could be randomized

    comp_dir = os.path.join(out_root, "comparisons")
    os.makedirs(comp_dir, exist_ok=True)

    print(f"[Compare] Saving {k} comparison grids to {comp_dir}")

    for idx in indices:
        img_t, lbl_t = dataset[idx]  # (3,H,W), (H,W)
        H, W = lbl_t.shape

        # Prepare base tiles
        rgb = img_t.clamp(0, 1)
        gt_color = colorize_mask(lbl_t, PALETTE_19)  # (3,H,W)

        row_tiles = [rgb.cpu(), gt_color.cpu()]
        col_titles = ["RGB", "GT"]

        # Predictions from each model
        for info in ckpt_infos:
            name = info["name"]
            sensor = info["sensor"]
            model = info["model"]

            pred = _predict_single(sensor, model, img_t, device)
            pred_color = colorize_mask(pred, PALETTE_19)
            row_tiles.append(pred_color.cpu())
            col_titles.append(name)

        # Stack as (C, H, W) tiles and make a grid
        tiles = torch.stack(row_tiles, dim=0)  # (Ncols,3,H,W)
        grid = _make_grid(tiles, nrow=len(row_tiles), padding=4)

        # Draw simple titles on top (manual, no fancy fonts)
        # We'll just print titles in console and save raw grid.
        out_png = os.path.join(comp_dir, f"idx{idx:05d}.png")
        _save_image(grid, out_png)
        print(f"[Compare] Saved grid for idx={idx} -> {out_png}")
        print("   columns:", " | ".join(col_titles))


def main():
    parser = argparse.ArgumentParser(description="Evaluate one or more DeepLens seg models and save metrics + comparison images.")
    parser.add_argument(
        "--ckpts",
        nargs="+",
        required=True,
        help="List of checkpoint .pt files to evaluate (e.g., best_epXX_accYY.pt).",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="./test_results",
        help="Root directory to store evaluation metrics and comparison images.",
    )
    parser.add_argument(
        "--num-vis",
        type=int,
        default=4,
        help="Number of validation samples to visualize in cross-model comparison grids.",
    )
    parser.add_argument(
        "--allow-missing-sensor",
        action="store_true",
        help="If set, load sensor state dict with strict=False (allow missing/extra keys).",
    )
    parser.add_argument(
        "--allow-missing-model",
        action="store_true",
        help="If set, load model state dict with strict=False (allow missing/extra keys).",
    )

    args = parser.parse_args()
    device = _prepare_device()

    # Evaluate each checkpoint individually
    all_results = []
    ckpt_infos_for_compare = []
    ref_val_loader = None

    for ckpt_path in args.ckpts:
        name = os.path.splitext(os.path.basename(ckpt_path))[0]
        out_dir = os.path.join(args.out_root, name)
        os.makedirs(out_dir, exist_ok=True)

        ckpt = _load_checkpoint(ckpt_path, device)
        cfg, sensor, model, val_loader, num_classes = _build_model_from_ckpt(
            ckpt,
            device,
            allow_missing_sensor=args.allow_missing_sensor,
            allow_missing_model=args.allow_missing_model,
        )

        # Use the ref val_loader for comparison dataset (first model wins)
        if ref_val_loader is None:
            ref_val_loader = val_loader

        # Run evaluation (this also saves per-model viz into out_dir/viz)
        metrics = evaluate_seg(sensor, model, val_loader, device,
                               num_classes=num_classes, out_dir=out_dir, epoch=0)

        # Save metrics JSON & record for summary
        _save_metrics_json(metrics, out_dir, tag="eval")
        all_results.append({
            "name": name,
            "pixel_acc": metrics["pixel_acc"],
            "mIoU": metrics["mIoU"],
            "per_class_IoU": metrics["per_class_IoU"],
        })

        # Keep sensor+model for comparison grids
        ckpt_infos_for_compare.append({
            "name": name,
            "sensor": sensor,
            "model": model,
        })

    # Write a global summary CSV
    if all_results:
        _append_summary_csv(all_results, args.out_root)

    # Save cross-model comparison grids (if more than 1 model or even for 1)
    if ref_val_loader is not None and ckpt_infos_for_compare:
        _save_comparison_grids(
            ckpt_infos=ckpt_infos_for_compare,
            ref_val_loader=ref_val_loader,
            device=device,
            out_root=args.out_root,
            num_vis=args.num_vis,
        )

    print("[Eval] Done.")


if __name__ == "__main__":
    main()