"""
Compose publication-quality qualitative figures from pre-rendered strip PNGs.

Each strip PNG (from generate_qualitative.py) has 6 equal-width columns:
  [Input RGB] [Ground Truth] [Fixed Camera] [No Optics (T=2)] [Tile-3] [Tile-4]

Processing:
  - NO palette snapping (original colours from generate_qualitative are already exact).
  - Void/ignore pixels (10,10,10) are detected from the GT column and blanked out
    with a neutral grey in GT *and* the same pixel positions in prediction columns,
    so GT and predictions are visually comparable without void artefacts.
  - ONE combined figure: ACDC rows on top, KITTI rows below.
  - Per-dataset supplement figures kept for appendix.

Usage:
  python paper/compose_qualitative.py

Outputs (in paper/figures/):
  qualitative_combined_paper.pdf / .png   ← main paper figure
  qualitative_acdc_paper.pdf / .png       ← supplement
  qualitative_kitti_paper.pdf / .png      ← supplement
"""
from __future__ import annotations

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import numpy as np
from PIL import Image

# ── Style ──────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "font.size":          8,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
})

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")

# ── Strip layout ───────────────────────────────────────────────────────────────
N_STRIP_COLS = 6   # panels per strip
STRIP_COL = {      # key → 0-based column index in the strip
    "Input RGB":        0,
    "Ground Truth":     1,
    "Fixed Camera":     2,
    "No Optics (Best)": 3,
    "Tile-3":           4,
    "Tile-4":           5,
}

VOID_BG = (220, 220, 220)   # light grey for void/unlabelled pixels


# ── Void handling ──────────────────────────────────────────────────────────────

def _void_mask(gt_panel: np.ndarray, threshold: int = 22) -> np.ndarray:
    """True where ALL channels < threshold — the (10,10,10) ignore class."""
    return (
        (gt_panel[:, :, 0] < threshold) &
        (gt_panel[:, :, 1] < threshold) &
        (gt_panel[:, :, 2] < threshold)
    )


def _apply_void(panels: list[np.ndarray],
                gt_index: int = 1) -> list[np.ndarray]:
    """Use the GT column's void mask to blank the same pixels in ALL panels.

    This makes GT and predictions directly comparable: void regions appear as
    the same neutral grey everywhere, eliminating the GT-black / prediction-colour
    mismatch.
    """
    gt = panels[gt_index]
    mask = _void_mask(gt)
    out = []
    for p in panels:
        pc = p.copy()
        pc[mask] = VOID_BG
        out.append(pc)
    return out


# ── Strip loading ──────────────────────────────────────────────────────────────

def _estimate_title_crop(path: str) -> int:
    """Height (px) of the white suptitle bar at the top of each strip."""
    img = np.array(Image.open(path).convert("RGB"))
    row_means = img.mean(axis=(1, 2))
    crop = 0
    for i, m in enumerate(row_means):
        if m > 232:
            crop = i + 1
        else:
            break
    return crop


def _load_scene(path: str,
                col_keys: list[str],
                title_crop: int) -> list[np.ndarray]:
    """Return one panel per col_key as uint8 HWC arrays, with void blanked."""
    img = np.array(Image.open(path).convert("RGB"))
    if title_crop > 0:
        img = img[title_crop:, :, :]

    H, W = img.shape[:2]
    pw = W // N_STRIP_COLS

    # Crop per-panel axis-label row at the very top (≈7 % of panel height)
    col_indices = [STRIP_COL[k] for k in col_keys]
    raw = []
    for ci in col_indices:
        x0 = ci * pw
        x1 = (ci + 1) * pw if ci < N_STRIP_COLS - 1 else W
        p = img[:, x0:x1, :].copy()
        axis_crop = max(0, int(p.shape[0] * 0.07))
        raw.append(p[axis_crop:, :, :])

    # Blank void using GT column (index 1 in the full strip, maps to col_keys index)
    gt_local_idx = col_keys.index("Ground Truth") if "Ground Truth" in col_keys else None
    if gt_local_idx is not None:
        raw = _apply_void(raw, gt_index=gt_local_idx)

    return raw


def _pick_strips(strip_dir: str, ranks: list[int] | None,
                 n: int) -> list[str]:
    pattern = os.path.join(strip_dir, "qual_*.png")
    paths = sorted(glob.glob(pattern),
                   key=lambda p: int(p.split("rank")[1][:2]) if "rank" in p else 0)
    if not paths:
        return []
    if ranks is not None:
        sel = []
        for r in ranks:
            m = [p for p in paths if f"rank{r:02d}" in p]
            if m:
                sel.append(m[0])
        return sel
    return paths[:n]


# ── Figure builder ────────────────────────────────────────────────────────────

COL_SPECS = [
    ("Input RGB",        "Input RGB"),
    ("Ground Truth",     "Ground Truth"),
    ("Fixed Camera",     "Fixed Camera"),
    ("No Optics (Best)", "No Optics (Ours)"),
]

COL_KEYS   = [k  for k, _ in COL_SPECS]
COL_LABELS = [lb for _, lb in COL_SPECS]


def _render_grid(
    datasets: list[tuple[str, list[list[np.ndarray]]]],
    out_stem: str,
    fig_w: float = 7.2,
    section_bg: str = "#EEF2F8",
    section_fg: str = "#0D2A6E",
) -> None:
    """Render a multi-dataset grid figure.

    datasets: [(dataset_label, [scene_panels, ...]), ...]
              scene_panels = list of np.ndarray per column
    """
    n_cols   = len(COL_LABELS)

    # Compute panel sizes from first available panel
    first_panel = datasets[0][1][0][0]
    ph, pw     = first_panel.shape[:2]
    aspect     = ph / pw

    panel_w_in  = fig_w / n_cols
    panel_h_in  = panel_w_in * aspect
    hdr_in      = 0.22    # column-label header row
    sec_in      = 0.22    # dataset section label row
    gap_in      = 0.10    # gap between datasets

    total_scenes = sum(len(rows) for _, rows in datasets)
    n_datasets   = len(datasets)
    fig_h = (hdr_in
             + n_datasets * sec_in
             + (n_datasets - 1) * gap_in
             + total_scenes * panel_h_in
             + 0.05)

    # Build height-ratio list for GridSpec rows
    # Order: [hdr, ds0_sec, ds0_scene×N, gap, ds1_sec, ds1_scene×M, ...]
    u   = panel_h_in   # unit
    hrs = [hdr_in / u]
    for di, (_, rows) in enumerate(datasets):
        hrs.append(sec_in / u)
        hrs.extend([1.0] * len(rows))
        if di < n_datasets - 1:
            hrs.append(gap_in / u)

    n_gs_rows = len(hrs)
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs  = gridspec.GridSpec(n_gs_rows, n_cols,
                            figure=fig,
                            height_ratios=hrs,
                            hspace=0.005,
                            wspace=0.005)

    # ── Column headers (row 0) ──────────────────────────────────────────────
    for j, lbl in enumerate(COL_LABELS):
        ax = fig.add_subplot(gs[0, j])
        ax.text(0.5, 0.15, lbl, ha="center", va="bottom",
                fontsize=9, fontweight="bold",
                transform=ax.transAxes, color="#111111")
        ax.axis("off")

    # ── Dataset sections ────────────────────────────────────────────────────
    gs_row = 1
    for di, (ds_label, rows) in enumerate(datasets):
        # Section banner
        ax_sec = fig.add_subplot(gs[gs_row, :])
        ax_sec.set_facecolor(section_bg)
        ax_sec.text(0.008, 0.5, ds_label,
                    ha="left", va="center", fontsize=8.5,
                    fontweight="bold", color=section_fg,
                    transform=ax_sec.transAxes)
        ax_sec.axis("off")
        gs_row += 1

        # Scenes
        for si, panels in enumerate(rows):
            for j, panel_img in enumerate(panels):
                ax = fig.add_subplot(gs[gs_row, j])
                ax.imshow(panel_img, interpolation="bilinear", aspect="auto")
                ax.axis("off")
                # thin white border to separate panels
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color("white")
                    spine.set_linewidth(0.7)
                # scene index on leftmost column
                if j == 0:
                    ax.text(-0.025, 0.5, f"({si + 1})",
                            ha="right", va="center", fontsize=7.5,
                            fontweight="bold", color="#666666",
                            transform=ax.transAxes)
            gs_row += 1

        # Gap row between datasets
        if di < n_datasets - 1:
            ax_gap = fig.add_subplot(gs[gs_row, :])
            ax_gap.axis("off")
            gs_row += 1

    out_pdf = os.path.join(FIGURES_DIR, f"{out_stem}.pdf")
    out_png = os.path.join(FIGURES_DIR, f"{out_stem}.png")
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


# ── Public API ────────────────────────────────────────────────────────────────

def compose_dataset(strip_dir: str,
                    out_stem: str,
                    dataset_label: str,
                    ranks: list[int] | None = None,
                    n: int = 3) -> list[list[np.ndarray]]:
    """Build and save a single-dataset figure; return scene rows for reuse."""
    strips = _pick_strips(strip_dir, ranks, n)
    if not strips:
        print(f"[skip] No strips in {strip_dir}")
        return []

    title_crop = _estimate_title_crop(strips[0])
    rows = [_load_scene(p, COL_KEYS, title_crop) for p in strips]

    _render_grid([(dataset_label, rows)], out_stem, fig_w=6.8)
    return rows


def compose_combined(acdc_dir: str,
                     kitti_dir: str,
                     out_stem: str = "qualitative_combined_paper",
                     acdc_ranks: list[int] | None = None,
                     kitti_ranks: list[int] | None = None,
                     n_acdc: int = 3,
                     n_kitti: int = 3) -> None:
    """Single figure: ACDC rows (top) + KITTI rows (bottom)."""
    datasets = []

    if os.path.isdir(acdc_dir):
        strips = _pick_strips(acdc_dir, acdc_ranks, n_acdc)
        if strips:
            tc = _estimate_title_crop(strips[0])
            rows = [_load_scene(p, COL_KEYS, tc) for p in strips]
            datasets.append(("ACDC  (Adverse Conditions: fog · night · rain · snow)", rows))

    if os.path.isdir(kitti_dir):
        strips = _pick_strips(kitti_dir, kitti_ranks, n_kitti)
        if strips:
            tc = _estimate_title_crop(strips[0])
            rows = [_load_scene(p, COL_KEYS, tc) for p in strips]
            datasets.append(("KITTI-360  (Normal Conditions)", rows))

    if not datasets:
        print("[warn] No data found for combined figure")
        return

    _render_grid(datasets, out_stem, fig_w=7.2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    acdc_dir  = os.path.join(FIGURES_DIR, "qualitative_acdc")
    kitti_dir = os.path.join(FIGURES_DIR, "qualitative_kitti")

    # Per-dataset supplements
    compose_dataset(acdc_dir,  "qualitative_acdc_paper",
                    "ACDC (Adverse Conditions)", ranks=[1, 2, 3])
    compose_dataset(kitti_dir, "qualitative_kitti_paper",
                    "KITTI-360", ranks=[1, 2, 3])

    # Main paper figure — single combined grid
    compose_combined(acdc_dir, kitti_dir,
                     out_stem="qualitative_combined_paper",
                     acdc_ranks=[1, 2, 3],
                     kitti_ranks=[1, 2, 3],
                     n_acdc=3, n_kitti=3)


if __name__ == "__main__":
    main()
