"""Render corrected qualitative figures from a saved checkpoint."""

from __future__ import annotations

import argparse
import os

import torch

from raw2task.eval_robustness import _build_from_checkpoint
from raw2task.train_extended import align_logits_to_labels, dense_task_palette, vis_batch_with_stages


@torch.no_grad()
def render(ckpt: str, out_dir: str, max_batches: int, max_items: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, sensor, model, loader, num_classes = _build_from_checkpoint(ckpt, device)
    palette = dense_task_palette(cfg, num_classes)
    sensor.eval()
    model.eval()
    ignore_index = int(getattr(model, "ignore_index", 255))

    for batch_idx, (imgs, labels) in enumerate(loader, 1):
        if batch_idx > max_batches:
            break
        imgs = imgs.to(device)
        labels = labels.to(device)
        raw, stages = sensor(imgs, return_stages=True)
        pred = align_logits_to_labels(model(raw), labels).argmax(1)
        out_png = os.path.join(out_dir, f"checkpoint_viz_step{batch_idx:05d}.png")
        vis_batch_with_stages(
            rgb=imgs,
            pred=pred,
            target=labels,
            stages=stages,
            out_png=out_png,
            palette=palette,
            ignore_index=ignore_index,
            max_items=max_items,
            diagnostic_stages=True,
            panel_caption=(
                "Validation qualitative panel: RGB input, learned-camera diagnostics, "
                "ground truth, and prediction for the same KITTI-360 frame."
            ),
        )
        print(f"wrote {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--max-items", type=int, default=4)
    args = parser.parse_args()
    render(args.ckpt, args.out_dir, args.max_batches, args.max_items)


if __name__ == "__main__":
    main()
