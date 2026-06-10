#!/usr/bin/env python3
import os, glob
import numpy as np
from PIL import Image

def head(paths, k=5):
    return paths[:k]

def list_images(img_dir):
    exts = ["*.png","*.jpg","*.jpeg"]
    out = []
    for e in exts:
        out += glob.glob(os.path.join(img_dir, e))
    return sorted(out)

def list_masks(mask_dir):
    if not os.path.isdir(mask_dir):
        return []
    return sorted(glob.glob(os.path.join(mask_dir, "*.png")))

def summarize_mask(mask_path):
    m = np.array(Image.open(mask_path))
    info = {"shape": m.shape, "dtype": str(m.dtype)}
    if m.ndim == 3:
        # sample unique colors
        flat = m.reshape(-1, m.shape[-1])
        # subsample for speed
        idx = np.random.choice(flat.shape[0], size=min(20000, flat.shape[0]), replace=False)
        u = np.unique(flat[idx], axis=0)
        info["ndim"] = 3
        info["unique_colors_sample"] = u[:15].tolist()
        info["num_unique_colors_sample"] = int(u.shape[0])
    else:
        u = np.unique(m)
        info["ndim"] = 2
        info["min"] = int(u.min()) if u.size else None
        info["max"] = int(u.max()) if u.size else None
        info["unique_head"] = u[:50].tolist()
        info["num_unique"] = int(u.size)
    return info

def main(root):
    img_dir = os.path.join(root, "images")
    mask_dir = os.path.join(root, "masks")

    imgs = list_images(img_dir)
    masks = list_masks(mask_dir)

    print("ROOT:", root)
    print("images dir:", img_dir, "exists:", os.path.isdir(img_dir), "count:", len(imgs))
    print("masks  dir:", mask_dir, "exists:", os.path.isdir(mask_dir), "count:", len(masks))
    print()

    print("IMAGE FILES (head):")
    for p in head(imgs):
        print(" ", os.path.basename(p))
    print()

    print("MASK FILES (head):")
    for p in head(masks):
        print(" ", os.path.basename(p))
    print()

    if masks:
        print("MASK SUMMARY (first mask):")
        print(summarize_mask(masks[0]))
        print()

        # also check if filenames have suffixes like _train_id
        bn = os.path.basename(masks[0])
        print("mask basename example:", bn)
        print("stem example:", os.path.splitext(bn)[0])
    else:
        print("No masks found. Qualitative-only dataset.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    main(args.root)
