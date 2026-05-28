#!/usr/bin/env python3
# deeplens/projects/raw2task/scripts/sanity_check_labels.py
"""
Quick label sanity checks for KITTI-360 / Cityscapes 19-class segmentation.

Checks:
  1) Unique label IDs in GT masks (expect only {0..18, 255}).
  2) Detects likely "original ids" (7,8,11,...) vs expected trainIds.
  3) Optional remap original ids -> trainIds and re-checks.
  4) Nearest-neighbor integrity (warns if masks have non-integer values).
  5) Class histogram and inverse-frequency class weights (normalized).
  6) Saves RGB + colorized GT (trainId palette) previews.

Usage:
  python deeplens/projects/raw2task/scripts/sanity_check_labels.py \
    --root /path/to/kitti360/root \
    --split val \
    --camera image_00 \
    --img-size 196,640 \
    --num-samples 8 \
    --save-dir ./checkpoints/label_audit \
    --apply-remap    # preview remapped trainIds (files on disk unchanged)
"""

import argparse, os, sys, json
import numpy as np
from PIL import Image
import torch
from torchvision.utils import save_image

# ---------------------------------------------------------------------
# Repo-root auto-detect (no more manual PYTHONPATH)
# ---------------------------------------------------------------------
def _maybe_add_repo_root():
    here = os.path.abspath(os.path.dirname(__file__))  # .../tools
    candidates = []
    # ascend up to 6 levels and stop when we see deeplens/ or projects/raw2task/
    cur = here
    for _ in range(6):
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        has_deeplens = os.path.isdir(os.path.join(parent, "deeplens"))
        has_raw2task = os.path.isdir(os.path.join(parent, "projects", "raw2task"))
        if has_deeplens or has_raw2task:
            candidates.append(parent)
        cur = parent

    for c in candidates:
        if c not in sys.path:
            sys.path.insert(0, c)
    return candidates

_ADDED_PATHS = _maybe_add_repo_root()

# Try to import dataset class
HAVE_DATASET = False
IMPORT_ERR = None
try:
    from projects.raw2task.data.kitti360_seg import KITTI360Seg
    HAVE_DATASET = True
except Exception as e1:
    IMPORT_ERR = f"{type(e1).__name__}: {e1}"
    # second chance: some repos name the package 'deeplens.projects...'
    try:
        from deeplens.projects.raw2task.data.kitti360_seg import KITTI360Seg  # type: ignore
        HAVE_DATASET = True
        IMPORT_ERR = None
    except Exception as e2:
        IMPORT_ERR = f"{IMPORT_ERR} | fallback: {type(e2).__name__}: {e2}"

# -------------------- Cityscapes 19-class mapping --------------------
# originalId -> trainId; anything else -> 255
CITYSCAPES_TRAINID = {
     7: 0,   # road
     8: 1,   # sidewalk
    11: 2,   # building
    12: 3,   # wall
    13: 4,   # fence
    17: 5,   # pole
    19: 6,   # traffic light
    20: 7,   # traffic sign
    21: 8,   # vegetation
    22: 9,   # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
}

# Cityscapes-like palette (19 classes)
PALETTE_19 = [
    (128,  64, 128),  # 0 road
    (244,  35, 232),  # 1 sidewalk
    ( 70,  70,  70),  # 2 building
    (102, 102, 156),  # 3 wall
    (190, 153, 153),  # 4 fence
    (153, 153, 153),  # 5 pole
    (250, 170,  30),  # 6 tlgt
    (220, 220,   0),  # 7 tsign
    (107, 142,  35),  # 8 vegetation
    (152, 251, 152),  # 9 terrain
    ( 70, 130, 180),  #10 sky
    (220,  20,  60),  #11 person
    (255,   0,   0),  #12 rider
    (  0,   0, 142),  #13 car
    (  0,   0,  70),  #14 truck
    (  0,  60, 100),  #15 bus
    (  0,  80, 100),  #16 train
    (  0,   0, 230),  #17 motorcycle
    (119,  11,  32),  #18 bicycle
]

def to_color(mask_np: np.ndarray) -> np.ndarray:
    """trainId mask -> RGB palette (uint8). 255 stays black."""
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i, (r,g,b) in enumerate(PALETTE_19):
        rgb[mask_np == i] = (r,g,b)
    return rgb

def remap_to_trainid(mask_np: np.ndarray) -> np.ndarray:
    out = np.full_like(mask_np, 255, dtype=np.uint8)
    for k, v in CITYSCAPES_TRAINID.items():
        out[mask_np == k] = v
    return out

def class_histogram(masks: list[np.ndarray], num_classes=19, ignore=255):
    hist = np.zeros(num_classes, dtype=np.int64)
    ign = 0
    for m in masks:
        vals, cnts = np.unique(m, return_counts=True)
        for v, c in zip(vals, cnts):
            if v == ignore: ign += int(c)
            elif 0 <= v < num_classes: hist[int(v)] += int(c)
    return hist, ign

def inv_freq_weights(hist: np.ndarray, eps=1e-3):
    freq = hist.astype(np.float64) + eps
    inv = 1.0 / freq
    w = inv / inv.mean()
    return np.clip(w, 0.25, 4.0)

def save_grid(rgb_list, mask_color_list, out_path):
    """Save a simple grid: [RGB | colorized GT] per row."""
    rows = []
    for rgb_np, mcol_np in zip(rgb_list, mask_color_list):
        a = torch.from_numpy(rgb_np.transpose(2,0,1)).float()/255.0
        b = torch.from_numpy(mcol_np.transpose(2,0,1)).float()/255.0
        rows.append(torch.stack([a, b], dim=0))
    grid = torch.cat(rows, dim=0)  # (2*N, 3, H, W)
    save_image(grid, out_path, nrow=2, padding=4)

def _pretty_repo_hint():
    return (
        "\n[HINT] Repo auto-detect looked for 'deeplens/' or 'projects/raw2task/' "
        f"in parents of: {os.path.abspath(os.path.dirname(__file__))}\n"
        f"        Added to sys.path: {_ADDED_PATHS or 'NONE'}\n"
        "        If this still fails, run:\n"
        "          export PYTHONPATH=/path/to/DeepLens:$PYTHONPATH\n"
        "        or run the script as a module from repo root:\n"
        "          python -m tools.sanity_check_labels --root ...\n"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=str)
    ap.add_argument("--split", default="val", choices=["train","val","test"])
    ap.add_argument("--camera", default="image_00", type=str)
    ap.add_argument("--img-size", default="196,640", type=str,
                    help="HxW; use the same as training")
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--save-dir", type=str, default="./checkpoints/label_audit")
    ap.add_argument("--apply-remap", action="store_true",
                    help="Preview remapped trainIds (files on disk unchanged).")
    ap.add_argument("--expected-ignore", type=int, default=255)
    ap.add_argument("--num-classes", type=int, default=19)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    H, W = [int(x) for x in args.img_size.split(",")]

    if not HAVE_DATASET:
        print("[ERROR] Could not import KITTI360Seg:", IMPORT_ERR, file=sys.stderr)
        print(_pretty_repo_hint(), file=sys.stderr)
        sys.exit(1)

    # Load dataset
    ds = KITTI360Seg(root=args.root, split=args.split, img_size=(H, W), camera=args.camera)
    print(f"[INFO] Loaded KITTI360Seg split={args.split} size={len(ds)} "
          f"img_size={(H,W)} camera={args.camera}")

    # Sample a few
    n = min(args.num_samples, len(ds))
    rgb_list, raw_mask_list, maybe_remap_list = [], [], []
    bad_integer = 0

    for idx in range(n):
        rgb, mask = ds[idx]  # rgb: (3,H,W) float; mask: (H,W) or (1,H,W)
        if torch.is_tensor(rgb):
            rgb_np = (rgb.permute(1,2,0).numpy() * 255.0).clip(0,255).astype(np.uint8)
        else:
            rgb_np = np.array(rgb)

        if torch.is_tensor(mask):
            mask_np = mask.squeeze().numpy()
        else:
            mask_np = np.array(mask)

        # detect interpolation artifacts (should be integers)
        if not np.allclose(mask_np, np.round(mask_np)):
            bad_integer += 1
        mask_np = np.round(mask_np).astype(np.int32)

        rgb_list.append(rgb_np)
        raw_mask_list.append(mask_np.astype(np.uint8))

        if args.apply_remap:
            remapped = remap_to_trainid(mask_np.astype(np.uint8))
            maybe_remap_list.append(remapped)
        else:
            maybe_remap_list.append(mask_np.astype(np.uint8))

    # stats before remap
    uni_before = np.unique(np.concatenate([m.flatten() for m in raw_mask_list]))
    print("[STATS] Unique label ids (as provided):", uni_before.tolist())
    if np.any(uni_before > 18) and (7 in uni_before or 11 in uni_before):
        print("  -> Looks like ORIGINAL Cityscapes ids. You likely need a trainId remap.")
    else:
        print("  -> These already look like trainIds (0..18) + ignore.")

    # stats after (maybe) remap
    uni_after = np.unique(np.concatenate([m.flatten() for m in maybe_remap_list]))
    print("[STATS] Unique label ids (after preview):", uni_after.tolist())
    allowed = set(range(args.num_classes)) | {args.expected_ignore}
    outside = set(uni_after.tolist()) - allowed
    if outside:
        print(f"  [WARN] Found ids outside {{0..{args.num_classes-1}, {args.expected_ignore}}}: {sorted(outside)}")
        print("        Check your mapping or masks.")

    if bad_integer > 0:
        print(f"[WARN] {bad_integer}/{n} sampled masks had non-integer values "
              f"(likely non-NEAREST resize). Fix all label resizes to use NEAREST.")

    # histogram & weights
    hist, ign = class_histogram(maybe_remap_list, num_classes=args.num_classes, ignore=args.expected_ignore)
    w = inv_freq_weights(hist)
    print("[HIST] counts per class (0..18):", hist.tolist())
    print("[HIST] ignore pixel count:", int(ign))
    print("[WEIGHTS] inverse-freq (clamped [0.25, 4.0]):", np.round(w, 4).tolist())

    # Save a grid preview
    color_list = [to_color(m) for m in maybe_remap_list]
    out_path = os.path.join(args.save_dir, f"preview_{args.split}_{n}x.png")
    save_grid(rgb_list, color_list, out_path)
    print(f"[SAVE] preview grid -> {out_path}")

    # Dump JSON report
    report = {
        "unique_ids_before": uni_before.tolist(),
        "unique_ids_after": uni_after.tolist(),
        "histogram": hist.tolist(),
        "ignore_count": int(ign),
        "weights": [float(x) for x in w.tolist()],
        "had_non_integer_masks": int(bad_integer > 0),
        "repo_paths_added": _ADDED_PATHS,
    }
    with open(os.path.join(args.save_dir, "label_audit.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[SAVE] report -> {os.path.join(args.save_dir, 'label_audit.json')}")

if __name__ == "__main__":
    main()
