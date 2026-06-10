#!/usr/bin/env python3
"""Compute KITTI-360 class frequency counts for train IDs 0..18."""

import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from raw2task.data.kitti360_seg import KITTI360Seg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-seg", required=True, help="KITTI-360 root directory")
    ap.add_argument("--out", default="deeplens/projects/raw2task/kitti360_class_freqs.npy")
    ap.add_argument("--img-size", default="192,640", help="H,W")
    ap.add_argument("--label-encoding", default="original_id", choices=["original_id", "train_id"])
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()

    h_str, w_str = args.img_size.split(",")
    img_size = (int(h_str), int(w_str))

    ds = KITTI360Seg(
        root=args.root_seg,
        split="train",
        img_size=img_size,
        camera="image_00",
        label_encoding=args.label_encoding,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_classes = 19
    ignore_index = 255
    counts = np.zeros(num_classes, dtype=np.int64)

    for _, labels in tqdm(loader, desc="Counting pixels"):
        target = labels.detach().cpu().numpy().astype(np.int64)
        valid = target[target != ignore_index]
        bincount = np.bincount(valid.ravel(), minlength=num_classes)
        counts += bincount[:num_classes]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, counts)
    print("Saved class frequencies to", args.out)
    print("counts:", counts.tolist())


if __name__ == "__main__":
    main()
