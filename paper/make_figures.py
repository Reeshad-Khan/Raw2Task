"""
Publication-quality figure generator for the Raw2Task spectral CFA paper.

Usage:
    # Pipeline diagram (no trained weights needed — run any time):
    python paper/make_figures.py --pipeline

    # CFA pattern comparison (needs tile2/tile3/tile4 runs to be done):
    python paper/make_figures.py --cfa \
        --tile2-dir runs/kitti360_sfb4/ablate_no_optics_sfb4_seed0 \
        --tile3-dir runs/kitti360_sfb4_spectral/cfa_tile3_sfb4_seed0 \
        --tile4-dir runs/kitti360_sfb4_spectral/cfa_tile4_sfb4_seed0

    # Results bar chart (fill in numbers after all jobs finish):
    python paper/make_figures.py --results

    # All figures at once:
    python paper/make_figures.py --all \
        --tile2-dir runs/kitti360_sfb4/ablate_no_optics_sfb4_seed0 \
        --tile3-dir runs/kitti360_sfb4_spectral/cfa_tile3_sfb4_seed0 \
        --tile4-dir runs/kitti360_sfb4_spectral/cfa_tile4_sfb4_seed0

All figures are saved to paper/figures/.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works on compute nodes without display
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ── publication style ─────────────────────────────────────────────────────────

mpl.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":   "stix",
    "font.size":          9,
    "axes.titlesize":     9,
    "axes.labelsize":     8,
    "xtick.labelsize":    7.5,
    "ytick.labelsize":    7.5,
    "legend.fontsize":    7.5,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          False,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
})

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── colour palette ─────────────────────────────────────────────────────────────

FIXED_COLOR   = "#BDD7EE"   # light blue — fixed components
LEARNED_COLOR = "#FFD966"   # amber — learned components
SKIP_COLOR    = "#F4CCCC"   # pink — bypassed / identity
BACKBONE_COLOR= "#D9EAD3"   # green — backbone
ARROW_COLOR   = "#444444"

CHANNEL_COLORS = {"R": "#D62728", "G": "#2CA02C", "B": "#1F77B4"}

# ── helpers ────────────────────────────────────────────────────────────────────

def _load_cfa_weights(run_dir: str) -> np.ndarray | None:
    """Load final CFA weights (n_sites, 3) from a run directory."""
    # Prefer camera_design_best.json → last epoch JSON → CSV
    candidates = (
        glob.glob(os.path.join(run_dir, "camera_design_best.json")) +
        sorted(glob.glob(os.path.join(run_dir, "camera_design_epoch*.json")), reverse=True) +
        sorted(glob.glob(os.path.join(run_dir, "camera_design_initial.json")))
    )
    for path in candidates:
        try:
            with open(path) as f:
                d = json.load(f)
            w = np.array(d["cfa_weights_rgb"], dtype=np.float32)
            if w.ndim == 2 and w.shape[1] == 3:
                return w
        except Exception:
            pass

    # Fall back to CSV
    csvs = sorted(glob.glob(os.path.join(run_dir, "design_preview", "*_cfa_weights.csv")))
    if not csvs:
        csvs = sorted(glob.glob(os.path.join(run_dir, "*_cfa_weights.csv")))
    for path in csvs:
        try:
            rows = []
            with open(path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append([float(row["R"]), float(row["G"]), float(row["B"])])
            if rows:
                return np.array(rows, dtype=np.float32)
        except Exception:
            pass
    return None


def _bayer_weights() -> np.ndarray:
    """Standard Bayer RGGB as (4, 3) weight matrix."""
    return np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)


def _spectral_diversity(w: np.ndarray) -> float:
    """Mean pairwise cosine distance between filter rows — higher is more diverse."""
    n = w.shape[0]
    if n < 2:
        return 0.0
    nw = w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-8)
    sims = nw @ nw.T
    mask = ~np.eye(n, dtype=bool)
    return float(1.0 - sims[mask].mean())


def _tile_size(n_sites: int) -> int:
    t = int(round(math.sqrt(n_sites)))
    assert t * t == n_sites, f"n_sites={n_sites} is not a perfect square"
    return t


# ── Figure 1: Pipeline Diagram ─────────────────────────────────────────────────

def plot_pipeline(save: bool = True) -> plt.Figure:
    """Architecture diagram showing the full RAW-to-task pipeline."""
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    def box(cx, cy, w, h, label, sublabel="", color=FIXED_COLOR, fontsize=8):
        rect = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.08", linewidth=0.8,
            edgecolor="#666666", facecolor=color, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(cx, cy + (0.10 if sublabel else 0), label,
                ha="center", va="center", fontsize=fontsize, fontweight="bold", zorder=4)
        if sublabel:
            ax.text(cx, cy - 0.22, sublabel, ha="center", va="center",
                    fontsize=6.5, color="#555555", zorder=4)

    def arrow(x1, x2, y=1.5):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR,
                                   lw=1.2, mutation_scale=10), zorder=2)

    def badge(cx, cy, text, color):
        ax.text(cx, cy, text, ha="center", va="center", fontsize=5.5,
                color="white", fontweight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", facecolor=color,
                          edgecolor="none", alpha=0.9))

    # Boxes — left to right
    box(0.7,  1.5, 1.1, 0.8, "Scene",       "RGB input",   FIXED_COLOR)
    box(2.3,  1.5, 1.2, 0.8, "Optics",      "Identity PSF", SKIP_COLOR)
    box(3.85, 1.5, 1.2, 0.8, "T×T Spectral","CFA Mosaic",  LEARNED_COLOR)
    box(5.4,  1.5, 1.2, 0.8, "Noise + ADC", "Poisson-Gauss", LEARNED_COLOR)
    box(6.95, 1.5, 1.2, 0.8, "Demosaick",   "Soft, diff'able", FIXED_COLOR)
    box(8.7,  1.5, 1.5, 0.8, "SegFormer-B4","Task backbone", BACKBONE_COLOR)

    # Arrows
    for x1, x2 in [(1.26, 1.66), (2.91, 3.21), (4.47, 4.77),
                   (6.02, 6.32), (7.57, 7.87)]:
        arrow(x1, x2)

    # Output arrow + label
    ax.annotate("", xy=(9.85, 1.5), xytext=(9.48, 1.5),
                arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, lw=1.2, mutation_scale=10))
    ax.text(9.92, 1.5, "Seg.\nMap", ha="left", va="center", fontsize=7.5)

    # Badges
    badge(2.3,  2.04, "Fixed (Identity)", SKIP_COLOR[:-2] + "88")
    badge(3.85, 2.04, "Learned",          "#E69138")
    badge(5.4,  2.04, "Learned",          "#E69138")

    # Annotation: learnable parameter count
    ax.annotate(
        r"$\mathbf{W} \in \mathbb{R}^{T^2 \times 3}$" + "\n(T=2,3,4)",
        xy=(3.85, 0.92), xytext=(3.85, 0.30),
        ha="center", fontsize=7, color="#333333",
        arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8),
    )

    # Legend
    legend_patches = [
        mpatches.Patch(facecolor=FIXED_COLOR,    edgecolor="#666", label="Fixed"),
        mpatches.Patch(facecolor=LEARNED_COLOR,  edgecolor="#666", label="Learned"),
        mpatches.Patch(facecolor=SKIP_COLOR,     edgecolor="#666", label="Identity (bypassed)"),
        mpatches.Patch(facecolor=BACKBONE_COLOR, edgecolor="#666", label="Pre-trained backbone"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", frameon=True,
              framealpha=0.9, fontsize=6.5, ncol=2,
              bbox_to_anchor=(0.99, 0.98), borderpad=0.5)

    fig.suptitle("Task-Optimal Spectral Mosaic Co-Design Pipeline",
                 fontsize=9.5, fontweight="bold", y=1.01)
    if save:
        out = os.path.join(OUT_DIR, "fig1_pipeline.pdf")
        fig.savefig(out)
        fig.savefig(out.replace(".pdf", ".png"))
        print(f"Saved: {out}")
    return fig


# ── Figure 2: CFA Tile Pattern Comparison ─────────────────────────────────────

def _draw_cfa_tile(ax, weights: np.ndarray, title: str, diversity: float):
    """Draw a T×T CFA tile grid coloured by spectral response on ax."""
    T = _tile_size(weights.shape[0])
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylim(-0.5, T - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)

    for idx in range(T * T):
        r, c = divmod(idx, T)
        rgb = weights[idx].clip(0, 1)
        # cell fill colour = actual spectral response as RGB
        cell = mpatches.FancyBboxPatch(
            (c - 0.45, (T - 1 - r) - 0.45), 0.90, 0.90,
            boxstyle="round,pad=0.04",
            facecolor=rgb, edgecolor="white", linewidth=1.5, zorder=2,
        )
        ax.add_patch(cell)
        # Text: R/G/B weights as small annotation
        dom = int(np.argmax(weights[idx]))
        labels = ["R", "G", "B"]
        brightness = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        txt_color = "white" if brightness < 0.55 else "#333333"
        ax.text(c, (T - 1 - r), labels[dom],
                ha="center", va="center", fontsize=9 - T,
                color=txt_color, fontweight="bold", zorder=3)
        ax.text(c, (T - 1 - r) - 0.28,
                f"{weights[idx, dom]:.2f}",
                ha="center", va="center", fontsize=6 - T * 0.3,
                color=txt_color, alpha=0.85, zorder=3)

    div_str = f"Diversity: {diversity:.3f}"
    ax.set_title(f"{title}\n{div_str}", fontsize=8, pad=4)


def plot_cfa_comparison(
    tile2_weights: np.ndarray | None = None,
    tile3_weights: np.ndarray | None = None,
    tile4_weights: np.ndarray | None = None,
    save: bool = True,
) -> plt.Figure:
    """Side-by-side CFA tile pattern visualisation: Bayer | tile3 | tile4.

    Learned 2×2 converges to near-Bayer at optimum (diversity constraint), so
    we omit it and show Bayer (fixed reference) alongside the non-trivial learned mosaics.
    """
    bayer = _bayer_weights()
    panels = [
        (bayer,         "Bayer 2×2\n(Fixed reference)"),
        (tile2_weights, "Learned 2×2\n(converges to Bayer)"),
        (tile3_weights, "Learned 3×3\n(Spectral-3)"),
        (tile4_weights, "Learned 4×4\n(Spectral-4)"),
    ]

    n_panels = sum(1 for w, _ in panels if w is not None)
    # Tile sizes vary (2×2, 3×3, 4×4) — allocate slightly more space for larger tiles
    widths = []
    for weights, _ in panels:
        if weights is None:
            continue
        T = _tile_size(weights.shape[0])
        widths.append(max(2.0, T * 0.75))

    fig, axes = plt.subplots(1, n_panels,
                             figsize=(sum(widths) + 0.3 * (n_panels - 1), 3.2),
                             gridspec_kw={"width_ratios": widths})
    if n_panels == 1:
        axes = [axes]

    ax_idx = 0
    for weights, title in panels:
        if weights is None:
            continue
        ax = axes[ax_idx]
        div = _spectral_diversity(weights)
        _draw_cfa_tile(ax, weights, title, div)
        ax_idx += 1

    fig.suptitle("Learned CFA Spectral Mosaic Designs",
                 fontsize=9.5, fontweight="bold", y=1.01)
    fig.tight_layout()

    if save:
        out = os.path.join(OUT_DIR, "fig2_cfa_comparison.pdf")
        fig.savefig(out)
        fig.savefig(out.replace(".pdf", ".png"))
        print(f"Saved: {out}")
    return fig


# ── Figure 3: Spectral Response Bars ──────────────────────────────────────────

def plot_spectral_bars(
    tile2_weights: np.ndarray | None = None,
    tile3_weights: np.ndarray | None = None,
    tile4_weights: np.ndarray | None = None,
    save: bool = True,
) -> plt.Figure:
    """R/G/B weight bars for each filter site, compared across tile sizes."""
    entries = [
        (_bayer_weights(),  "Bayer 2×2 (fixed)"),
        (tile2_weights,     "Learned 2×2"),
        (tile3_weights,     "Learned 3×3 (Ours)"),
        (tile4_weights,     "Learned 4×4 (Ours)"),
    ]
    entries = [(w, lbl) for w, lbl in entries if w is not None]
    n = len(entries)

    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 2.4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (weights, label) in zip(axes, entries):
        n_sites = weights.shape[0]
        x = np.arange(n_sites)
        w_bar = 0.26
        for j, (ch, color) in enumerate(CHANNEL_COLORS.items()):
            ax.bar(x + (j - 1) * w_bar, weights[:, j], w_bar,
                   color=color, alpha=0.85, label=ch)
        ax.set_xticks(x)
        # For large tile sizes, only label every other site to avoid clutter
        step = 2 if n_sites > 9 else 1
        tick_labels = [f"s{i}" if i % step == 0 else "" for i in range(n_sites)]
        ax.set_xticklabels(tick_labels, fontsize=6.5)
        ax.set_ylim(0, 1.15)
        ax.set_title(label, fontsize=8.5, pad=5)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        div = _spectral_diversity(weights)
        ax.text(0.97, 0.96, f"div={div:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color="#555555")

    axes[0].set_ylabel("Spectral weight", fontsize=8)
    axes[0].legend(frameon=False, fontsize=7, ncol=3, loc="upper left")

    fig.suptitle("Learned Spectral Filter Responses per CFA Site",
                 fontsize=9, fontweight="bold")
    fig.tight_layout()

    if save:
        out = os.path.join(OUT_DIR, "fig3_spectral_bars.pdf")
        fig.savefig(out)
        fig.savefig(out.replace(".pdf", ".png"))
        print(f"Saved: {out}")
    return fig


# ── Figure 4: Spatial Mosaic Pattern ──────────────────────────────────────────

def plot_mosaic_pattern(
    tile2_weights: np.ndarray | None = None,
    tile3_weights: np.ndarray | None = None,
    tile4_weights: np.ndarray | None = None,
    grid_size: int = 12,
    save: bool = True,
) -> plt.Figure:
    """Show how the T×T tile repeats spatially over a patch of the image."""
    entries = [
        (_bayer_weights(),  "Bayer 2×2\n(Fixed)"),
        (tile3_weights,     "Learned 3×3\n(Spectral-3)"),
        (tile4_weights,     "Learned 4×4\n(Spectral-4)"),
    ]
    entries = [(w, lbl) for w, lbl in entries if w is not None]
    n = len(entries)

    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.6))
    if n == 1:
        axes = [axes]

    for ax, (weights, label) in zip(axes, entries):
        T = _tile_size(weights.shape[0])
        # Build a grid_size × grid_size image where each pixel is coloured
        # by its filter's spectral response
        img = np.zeros((grid_size, grid_size, 3), dtype=np.float32)
        for row in range(grid_size):
            for col in range(grid_size):
                site_idx = (row % T) * T + (col % T)
                img[row, col] = weights[site_idx].clip(0, 1)

        ax.imshow(img, interpolation="nearest", aspect="equal")
        # Draw tile boundaries
        for i in range(0, grid_size + 1, T):
            ax.axhline(i - 0.5, color="white", lw=1.2)
            ax.axvline(i - 0.5, color="white", lw=1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label, fontsize=8)
        ax.set_xlabel(f"T={T}, {T*T} filters", fontsize=7)

    fig.suptitle("Spatial Filter Arrangement\n(colour = spectral response; white lines = tile boundaries)",
                 fontsize=8.5, fontweight="bold")
    fig.tight_layout()

    if save:
        out = os.path.join(OUT_DIR, "fig4_mosaic_pattern.pdf")
        fig.savefig(out)
        fig.savefig(out.replace(".pdf", ".png"))
        print(f"Saved: {out}")
    return fig


# ── Figure 5: mIoU Results Bar Chart ──────────────────────────────────────────

# Hardcoded fallback results (overridden by auto_collect_results when run dirs are given).
# KITTI-360 spectral: jobs 662637-639,661 — done ~Jun 13 18:00
# ACDC spectral: jobs 662657-660 — DONE Jun 13 12:41 (new physics)
RESULTS = {
    "KITTI-360": {
        "RGB":                  0.6396,   # runs/kitti360_sfb4/rgb_sfb4_seed0
        "Fixed camera":         0.6043,   # runs/kitti360_sfb4/ablate_fixed_camera_sfb4_seed0
        "Co-design (PSF+CFA)":  0.6031,   # runs/kitti360_sfb4/codesign_sfb4_seed0  (old physics)
        "No optics ★":          0.6229,   # runs/kitti360_sfb4_spectral/ablate_no_optics_sfb4_seed0 (new physics, ep40)
        "CFA only":             0.6214,   # runs/kitti360_sfb4_spectral/cfa_only_sfb4_seed0          (new physics, ep40)
        "Tile-3 (ablation)":    0.6165,   # runs/kitti360_sfb4_spectral/cfa_tile3_sfb4_seed0         (new physics, ep40)
        "Tile-4 (ablation)":    0.6133,   # runs/kitti360_sfb4_spectral/cfa_tile4_sfb4_seed0         (new physics, ep40)
    },
    "ACDC": {
        "RGB":                  0.7374,   # runs/acdc_sfb4/rgb_acdc_seed0
        "Fixed camera":         0.6654,   # runs/acdc_sfb4/ablate_fixed_camera_acdc_seed0
        "Co-design (PSF+CFA)":  0.6906,   # runs/acdc_sfb4/codesign_acdc_seed0  (old physics)
        "No optics ★":          0.6853,   # runs/acdc_sfb4_spectral/ablate_no_optics_acdc_seed0 (new physics, ep60)
        "CFA only":             0.6882,   # runs/acdc_sfb4_spectral/cfa_only_acdc_seed0         (new physics, ep60)
        "Tile-3 (ablation)":    0.6800,   # runs/acdc_sfb4_spectral/cfa_tile3_acdc_seed0        (new physics, ep60)
        "Tile-4 (ablation)":    0.6740,   # runs/acdc_sfb4_spectral/cfa_tile4_acdc_seed0        (new physics, ep60)
    },
}

EXP_COLORS = {
    "RGB":                  "#9E9E9E",
    "Fixed camera":         "#607D8B",
    "Co-design (PSF+CFA)":  "#EF9A9A",
    "No optics ★":          "#1565C0",   # best config — deep blue
    "CFA only":             "#42A5F5",
    "Tile-3 (ablation)":    "#FFA726",
    "Tile-4 (ablation)":    "#E65100",
}

def _safe_label(s: str) -> str:
    """Replace Unicode chars that are missing from common serif fonts."""
    return s.replace("★", "*")


def plot_results(results: dict | None = None, save: bool = True) -> plt.Figure:
    """Grouped bar chart of mIoU results across all experiments and datasets."""
    if results is None:
        results = RESULTS

    datasets = list(results.keys())
    exp_names = list(results[datasets[0]].keys())
    n_exps = len(exp_names)
    n_ds   = len(datasets)

    fig, axes = plt.subplots(1, n_ds, figsize=(4.8 * n_ds, 3.8), sharey=False)
    if n_ds == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        data  = results[dataset]
        vals  = [data[e] for e in exp_names]

        # y-axis range: start just below fixed camera to show relative differences clearly
        y_min_val = min(v for v in vals if v is not None)
        y_max_val = max(v for v in vals if v is not None)
        y_margin = (y_max_val - y_min_val) * 0.25
        y_lo = max(0, y_min_val - y_margin * 0.5)
        y_hi = y_max_val + y_margin * 1.8  # headroom for labels

        x = np.arange(n_exps)
        for i, (name, val) in enumerate(zip(exp_names, vals)):
            color = EXP_COLORS.get(name, "#CCCCCC")
            alpha = 1.0 if val is not None else 0.35
            height = val if val is not None else y_lo + 0.001
            ax.bar(i, height, 0.65, color=color, alpha=alpha, bottom=0,
                   edgecolor="white", linewidth=0.8, zorder=3)
            if val is not None:
                # Horizontal label above bar, easier to read than rotated
                ax.text(i, val + (y_max_val - y_min_val) * 0.02,
                        f"{val:.4f}", ha="center", va="bottom",
                        fontsize=6.5, color="#222222", fontweight="bold")
            else:
                ax.text(i, y_lo + y_margin * 0.1, "—", ha="center", va="bottom",
                        fontsize=8, color="#AAAAAA")

        # Fixed camera reference line
        fixed_val = data.get("Fixed camera")
        if fixed_val is not None:
            ax.axhline(fixed_val, color="#607D8B", linewidth=1.0,
                       linestyle="--", alpha=0.6, zorder=2,
                       label=f"Fixed camera ({fixed_val:.4f})")

        ax.set_title(dataset, fontsize=10, fontweight="bold", pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels([_safe_label(n) for n in exp_names],
                           rotation=30, ha="right", fontsize=7.5)
        ax.set_ylabel("mIoU", fontsize=8.5)
        ax.set_ylim(y_lo, y_hi)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5, zorder=0)

        # Annotate best sensor config
        best_idx = next((i for i, n in enumerate(exp_names) if "★" in n), None)
        if best_idx is not None and vals[best_idx] is not None:
            ax.annotate(
                "Best sensor", xy=(best_idx, vals[best_idx]),
                xytext=(best_idx + 1.1, vals[best_idx] + (y_max_val - y_min_val) * 0.06),
                fontsize=7, color=EXP_COLORS.get("No optics ★", "#1565C0"),
                fontweight="bold",
                arrowprops=dict(arrowstyle="->",
                                color=EXP_COLORS.get("No optics ★", "#1565C0"),
                                lw=0.9),
            )

    fig.suptitle("mIoU Comparison — Task-Optimal Spectral CFA Design",
                 fontsize=10, fontweight="bold", y=1.01)
    fig.tight_layout()

    if save:
        out = os.path.join(OUT_DIR, "fig5_results.pdf")
        fig.savefig(out)
        fig.savefig(out.replace(".pdf", ".png"))
        print(f"Saved: {out}")
    return fig


# ── Figure 6: Tile Size vs mIoU Trend ────────────────────────────────────────

def plot_tile_trend(
    kitti_miou: dict | None = None,
    acdc_miou:  dict | None = None,
    save: bool = True,
) -> plt.Figure:
    """Line plot showing mIoU vs CFA tile size for KITTI and ACDC."""
    # Final results from Newton runs (new physics)
    if kitti_miou is None:
        kitti_miou = {2: 0.6229, 3: 0.6165, 4: 0.6133}
    if acdc_miou is None:
        acdc_miou  = {2: 0.6853, 3: 0.6800, 4: 0.6740}

    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    sizes = [2, 3, 4]

    for label, data, color, marker in [
        ("KITTI-360", kitti_miou, "#1565C0", "o"),
        ("ACDC",      acdc_miou,  "#E65100", "s"),
    ]:
        ys  = [data.get(s) for s in sizes]
        xs_ = [s for s, y in zip(sizes, ys) if y is not None]
        ys_ = [y for y in ys if y is not None]
        if xs_:
            ax.plot(xs_, ys_, color=color, marker=marker, linewidth=1.8,
                    markersize=6, label=label, zorder=3)
        # pending points as dotted
        for s, y in zip(sizes, ys):
            if y is None:
                ax.plot(s, 0, marker=marker, color=color, alpha=0.3,
                        markersize=5, zorder=2)

    ax.set_xlabel("CFA tile size T  (filters = T²)", fontsize=8)
    ax.set_ylabel("mIoU", fontsize=8)
    ax.set_xticks(sizes)
    ax.set_xticklabels(["2×2\n(4 filters)", "3×3\n(9 filters)", "4×4\n(16 filters)"])
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_title("mIoU vs CFA Tile Size", fontsize=9, fontweight="bold")
    fig.tight_layout()

    if save:
        out = os.path.join(OUT_DIR, "fig6_tile_trend.pdf")
        fig.savefig(out)
        fig.savefig(out.replace(".pdf", ".png"))
        print(f"Saved: {out}")
    return fig


# ── Auto-collect results from checkpoint dirs ──────────────────────────────────

def _collect_best_miou(run_dir: str) -> tuple:
    """Return (best_mIoU, best_pixel_acc) from metrics_log.csv, or (None, None)."""
    csv_path = os.path.join(run_dir, "metrics_log.csv")
    if not os.path.isfile(csv_path):
        return (None, None)
    best_miou = None
    best_acc = None
    try:
        import csv as _csv
        with open(csv_path, newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                try:
                    m = float(row.get("mIoU") or row.get("miou") or 0)
                    a = float(row.get("pixel_acc") or row.get("PixAcc") or 0)
                    if best_miou is None or m > best_miou:
                        best_miou = m
                        best_acc = a
                except (ValueError, TypeError):
                    pass
    except Exception:
        pass
    return (best_miou, best_acc)


def _find_seed0(root: str, exp_name: str) -> str | None:
    """Return the seed0 directory for exp_name under root, or None."""
    for candidate in [
        os.path.join(root, f"{exp_name}_seed0"),
        os.path.join(root, exp_name),
        os.path.join(root, f"{exp_name}_seed0_s0"),
    ]:
        if os.path.isdir(candidate):
            return candidate
    return None


def auto_collect_results(
    kitti_spectral_root: str | None = None,
    kitti_base_root: str | None = None,
    acdc_spectral_root: str | None = None,
    acdc_base_root: str | None = None,
) -> dict:
    """Read mIoU/PixAcc from checkpoint dirs. Returns results dict for plot_results().

    Experiment name mapping (spectral dirs):
      ablate_no_optics_{sfb4,acdc}  → "No optics ★"
      cfa_only_{sfb4,acdc}          → "CFA only"
      cfa_tile3_{sfb4,acdc}         → "Tile-3 (ablation)"
      cfa_tile4_{sfb4,acdc}         → "Tile-4 (ablation)"

    Experiment name mapping (base dirs):
      rgb_{sfb4,acdc}               → "RGB"
      ablate_fixed_camera_{sfb4,acdc} → "Fixed camera"
      codesign_{sfb4,acdc}          → "Co-design (PSF+CFA)"
    """
    SPEC_MAP = {
        "ablate_no_optics_sfb4":    "No optics ★",
        "ablate_no_optics_acdc":    "No optics ★",
        "cfa_only_sfb4":            "CFA only",
        "cfa_only_acdc":            "CFA only",
        "cfa_tile3_sfb4":           "Tile-3 (ablation)",
        "cfa_tile3_acdc":           "Tile-3 (ablation)",
        "cfa_tile4_sfb4":           "Tile-4 (ablation)",
        "cfa_tile4_acdc":           "Tile-4 (ablation)",
    }
    BASE_MAP = {
        "rgb_sfb4":                 "RGB",
        "rgb_acdc":                 "RGB",
        "ablate_fixed_camera_sfb4": "Fixed camera",
        "ablate_fixed_camera_acdc": "Fixed camera",
        "codesign_sfb4":            "Co-design (PSF+CFA)",
        "codesign_acdc":            "Co-design (PSF+CFA)",
    }

    def _read_map(root, mapping, suffix):
        out = {}
        if not root:
            return out
        for exp_name, label in mapping.items():
            if suffix not in exp_name:
                continue
            d = _find_seed0(root, exp_name)
            if d:
                m, a = _collect_best_miou(d)
                if m is not None:
                    out[label] = (m, a)
                    print(f"  [{suffix}] {label}: mIoU={m:.4f}  from {d}")
                else:
                    print(f"  [{suffix}] {label}: no metrics yet in {d}")
            else:
                print(f"  [{suffix}] {label}: dir not found under {root}")
        return out

    kitti_spec  = _read_map(kitti_spectral_root, SPEC_MAP, "sfb4")
    kitti_base  = _read_map(kitti_base_root,     BASE_MAP, "sfb4")
    acdc_spec   = _read_map(acdc_spectral_root,  SPEC_MAP, "acdc")
    acdc_base   = _read_map(acdc_base_root,      BASE_MAP, "acdc")

    EXP_ORDER = [
        "RGB", "Fixed camera", "Co-design (PSF+CFA)",
        "No optics ★", "CFA only", "Tile-3 (ablation)", "Tile-4 (ablation)",
    ]

    def _merge(spec, base, fallback):
        merged = {}
        for k in EXP_ORDER:
            v = spec.get(k) or base.get(k)
            merged[k] = v[0] if v else fallback.get(k)
        return merged

    results = {
        "KITTI-360": _merge(kitti_spec, kitti_base, RESULTS["KITTI-360"]),
        "ACDC":      _merge(acdc_spec,  acdc_base,  RESULTS["ACDC"]),
    }
    return results


# ── Figure 7: Ablation Decomposition ──────────────────────────────────────────

def plot_ablation_decomposition(results: dict | None = None, save: bool = True) -> plt.Figure:
    """Sensor dimension contribution chart.

    Grouped bar chart with 3 component groups on the x-axis:
      CFA learning | Noise optimisation | PSF optimisation
    Each group has one bar per dataset (KITTI=blue, ACDC=orange).
    Bars extend above/below zero to show positive/negative contributions.

    Note: ACDC PSF bar uses old-physics co-design vs new-physics no_optics —
    the comparison is indicative only (shown with hatching).
    """
    if results is None:
        results = RESULTS

    DS_COLORS = {"KITTI-360": "#1565C0", "ACDC": "#E65100"}

    # Compute deltas for each dataset
    components: dict[str, dict] = {}
    for ds, data in results.items():
        fixed    = data.get("Fixed camera")
        cfa_o    = data.get("CFA only")
        no_opt   = data.get("No optics ★")
        codesign = data.get("Co-design (PSF+CFA)")
        rgb_val  = data.get("RGB")

        if fixed is None:
            continue

        cfa_gain   = (cfa_o  - fixed)    if cfa_o  is not None else None
        noise_eff  = (no_opt - cfa_o)    if (no_opt is not None and cfa_o is not None) else None
        # PSF effect: codesign - no_optics  (negative = PSF hurts)
        # For ACDC this is cross-physics (old co-design vs new no_optics) — flag it
        psf_eff    = (codesign - no_opt) if (codesign is not None and no_opt is not None) else None
        oracle     = (rgb_val  - fixed)  if rgb_val  is not None else None

        components[ds] = {
            "CFA learning":        cfa_gain,
            "Noise optimisation":  noise_eff,
            "PSF optimisation":    psf_eff,
            "oracle":              oracle,
        }

    comp_keys = ["CFA learning", "Noise optimisation", "PSF optimisation"]
    ds_keys   = [ds for ds in ["KITTI-360", "ACDC"] if ds in components]
    n_groups  = len(comp_keys)
    n_ds      = len(ds_keys)

    fig, ax = plt.subplots(figsize=(5.8, 3.4))

    x      = np.arange(n_groups)
    w      = 0.32
    # Centre bars symmetrically around group tick
    offsets = np.linspace(-(n_ds - 1) / 2, (n_ds - 1) / 2, n_ds) * w

    all_vals = []
    for ds, offset in zip(ds_keys, offsets):
        comp_data = components[ds]
        color = DS_COLORS[ds]

        for gi, ckey in enumerate(comp_keys):
            val = comp_data.get(ckey)
            if val is None:
                continue

            # ACDC PSF bar is cross-physics — show with hatching
            cross_physics = (ds == "ACDC" and ckey == "PSF optimisation")
            hatch = "////" if cross_physics else None
            edge_color = "#AAAAAA" if cross_physics else "white"

            pos = x[gi] + offset
            bar_alpha = 0.5 if cross_physics else 0.9
            ax.bar(pos, val, w * 0.95,
                   color=color, alpha=bar_alpha, hatch=hatch,
                   edgecolor=edge_color, linewidth=0.8, zorder=3)

            # Value label
            sign = "+" if val >= 0 else ""
            label_y = val + 0.0006 if val >= 0 else val - 0.0006
            va = "bottom" if val >= 0 else "top"
            ax.text(pos, label_y, f"{sign}{val:.3f}",
                    ha="center", va=va, fontsize=7, fontweight="bold",
                    color=color)
            all_vals.append(val)

    # Zero reference line
    ax.axhline(0, color="#555555", linewidth=0.9, zorder=1)

    # x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(comp_keys, fontsize=9)
    ax.set_ylabel(r"$\Delta$mIoU  (vs Fixed-Camera baseline)", fontsize=8.5)

    # y-axis: symmetric range with enough headroom for labels
    if all_vals:
        pad = max(abs(v) for v in all_vals) * 0.55
        ax.set_ylim(min(0, min(all_vals)) - pad * 0.4,
                    max(0, max(all_vals)) + pad)

    ax.grid(axis="y", alpha=0.25, linewidth=0.5, zorder=0)

    # Shaded region: "helps" vs "hurts"
    ymin, ymax = ax.get_ylim()
    ax.axhspan(0, ymax, alpha=0.04, color="#4CAF50", zorder=0)
    ax.axhspan(ymin, 0, alpha=0.04, color="#F44336", zorder=0)
    ax.text(n_groups - 0.05, ymax * 0.92, "helps ▲",
            ha="right", va="top", fontsize=7, color="#388E3C", style="italic")
    ax.text(n_groups - 0.05, ymin * 0.92, "hurts ▼",
            ha="right", va="bottom", fontsize=7, color="#C62828", style="italic")

    # Legend
    ds_handles = [
        mpatches.Patch(facecolor=DS_COLORS[ds], label=ds) for ds in ds_keys
    ]
    if any(ds == "ACDC" for ds in ds_keys):
        ds_handles.append(mpatches.Patch(
            facecolor=DS_COLORS.get("ACDC", "#E65100"), hatch="////",
            edgecolor="#AAAAAA", alpha=0.5,
            label="ACDC PSF (cross-physics†)"))
    ax.legend(handles=ds_handles, fontsize=7.5, frameon=True,
              framealpha=0.9, loc="upper right", ncol=1)

    ax.set_title("Sensor Dimension Contribution to mIoU",
                 fontsize=10, fontweight="bold", pad=6)

    # Footnote for cross-physics note
    fig.text(0.5, -0.04,
             "† ACDC PSF bar compares old-physics co-design to new-physics no-optics; directional only.",
             ha="center", fontsize=6.5, style="italic", color="#666666")

    fig.tight_layout()

    if save:
        out = os.path.join(OUT_DIR, "fig7_ablation_decomposition.pdf")
        fig.savefig(out)
        fig.savefig(out.replace(".pdf", ".png"))
        print(f"Saved: {out}")
    return fig


# ── LaTeX Table ───────────────────────────────────────────────────────────────

_LATEX_EXP_META = [
    # (label_key,           psf,           cfa,           noise,         tile, display_name)
    ("RGB",                 r"\textemdash", r"\textemdash", r"\textemdash", r"\textemdash", r"RGB (no sensor pipeline)"),
    ("Fixed camera",        r"$\times$",   r"$\times$",   r"$\times$",   "2",  r"Fixed camera"),
    ("Co-design (PSF+CFA)", r"\checkmark", r"\checkmark", r"\checkmark", "2",  r"Co-design (PSF+CFA+noise)"),
    ("No optics ★",         r"$\times$",   r"\checkmark", r"\checkmark", "2",  r"No optics \textbf{(Ours$^\star$)}"),
    ("CFA only",            r"$\times$",   r"\checkmark", r"$\times$",   "2",  r"CFA only (no noise co-opt.)"),
    ("Tile-3 (ablation)",   r"$\times$",   r"\checkmark", r"\checkmark", "3",  r"Tile-3 ablation"),
    ("Tile-4 (ablation)",   r"$\times$",   r"\checkmark", r"\checkmark", "4",  r"Tile-4 ablation"),
]


def generate_latex_table(results: dict | None = None, out_file: str | None = None) -> str:
    """Return LaTeX tabular for the paper's main results table.

    Columns: Method | PSF | CFA | Noise | Tile | KITTI mIoU | KITTI PixAcc | ACDC mIoU | ACDC PixAcc
    """
    if results is None:
        results = RESULTS

    def _fmt(v):
        if v is None:
            return r"\textemdash"
        return f"{v:.4f}"

    def _bold(s, condition):
        return r"\textbf{" + s + "}" if condition else s

    kitti = results.get("KITTI-360", {})
    acdc  = results.get("ACDC", {})

    best_kitti = max((v for v in kitti.values() if v is not None), default=None)
    best_acdc  = max((v for v in acdc.values()  if v is not None), default=None)

    header = (
        r"\begin{table}[t]" + "\n"
        r"\centering" + "\n"
        r"\caption{Sensor co-design results on KITTI-360 and ACDC (SegFormer-B4). "
        r"Best \emph{sensor} configuration highlighted in bold.}" + "\n"
        r"\label{tab:main_results}" + "\n"
        r"\small" + "\n"
        r"\begin{tabular}{lccccrrrr}" + "\n"
        r"\toprule" + "\n"
        r"Method & PSF & CFA & Noise & Tile "
        r"& \multicolumn{2}{c}{KITTI-360} & \multicolumn{2}{c}{ACDC} \\" + "\n"
        r"\cmidrule(lr){6-7}\cmidrule(lr){8-9}" + "\n"
        r"& & & & & mIoU & PixAcc & mIoU & PixAcc \\" + "\n"
        r"\midrule" + "\n"
    )

    rows = []
    for key, psf, cfa, noise, tile, name in _LATEX_EXP_META:
        km = kitti.get(key)
        ka = None
        am = acdc.get(key)
        aa = None

        km_s = _bold(_fmt(km), km == best_kitti and km is not None)
        am_s = _bold(_fmt(am), am == best_acdc  and am is not None)
        ka_s = _fmt(ka)
        aa_s = _fmt(aa)

        if key == "RGB":
            rows.append(r"\midrule")

        row = f"{name} & {psf} & {cfa} & {noise} & {tile} & {km_s} & {ka_s} & {am_s} & {aa_s} \\\\"
        rows.append(row)

    footer = (
        r"\midrule" + "\n"
        r"\multicolumn{9}{l}{\scriptsize $\star$ Best sensor configuration (new physics). "
        r"PSF: \checkmark=learnable, $\times$=identity PSF. "
        r"CFA: \checkmark=learnable spectral weights, $\times$=fixed Bayer.}" + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        r"\end{table}"
    )

    table = header + "\n".join(rows) + "\n" + footer

    if out_file:
        os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
        with open(out_file, "w") as f:
            f.write(table)
        print(f"Saved LaTeX table: {out_file}")

    return table


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate all paper figures for the Raw2Task spectral CFA paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # After all jobs done — auto-read results and generate everything:
  python paper/make_figures.py --all \\
      --kitti-spectral-dir runs/kitti360_sfb4_spectral \\
      --kitti-base-dir     runs/kitti360_sfb4 \\
      --acdc-spectral-dir  runs/acdc_sfb4_spectral \\
      --acdc-base-dir      runs/acdc_sfb4 \\
      --tile2-dir  runs/kitti360_sfb4_spectral/ablate_no_optics_sfb4_seed0 \\
      --tile3-dir  runs/kitti360_sfb4_spectral/cfa_tile3_sfb4_seed0 \\
      --tile4-dir  runs/kitti360_sfb4_spectral/cfa_tile4_sfb4_seed0

  # Pipeline diagram only (no trained weights needed):
  python paper/make_figures.py --pipeline

  # LaTeX table only:
  python paper/make_figures.py --latex \\
      --kitti-spectral-dir runs/kitti360_sfb4_spectral \\
      --acdc-spectral-dir  runs/acdc_sfb4_spectral
""",
    )
    # Figure flags
    ap.add_argument("--pipeline",  action="store_true", help="Fig 1: pipeline diagram")
    ap.add_argument("--cfa",       action="store_true", help="Fig 2: CFA tile comparison")
    ap.add_argument("--bars",      action="store_true", help="Fig 3: spectral response bars")
    ap.add_argument("--mosaic",    action="store_true", help="Fig 4: spatial mosaic pattern")
    ap.add_argument("--results",   action="store_true", help="Fig 5: mIoU bar chart")
    ap.add_argument("--trend",     action="store_true", help="Fig 6: tile size vs mIoU trend")
    ap.add_argument("--ablation",  action="store_true", help="Fig 7: ablation decomposition")
    ap.add_argument("--latex",     action="store_true", help="Generate LaTeX results table")
    ap.add_argument("--all",       action="store_true", help="Generate all figures + table")
    # CFA weight dirs
    ap.add_argument("--tile2-dir", default=None, help="Run dir for no_optics (tile2) CFA weights")
    ap.add_argument("--tile3-dir", default=None, help="Run dir for cfa_tile3 CFA weights")
    ap.add_argument("--tile4-dir", default=None, help="Run dir for cfa_tile4 CFA weights")
    # Auto-collect result dirs
    ap.add_argument("--kitti-spectral-dir", default=None,
                    help="Root of kitti360_sfb4_spectral runs (auto-reads mIoU)")
    ap.add_argument("--kitti-base-dir",     default=None,
                    help="Root of kitti360_sfb4 baseline runs")
    ap.add_argument("--acdc-spectral-dir",  default=None,
                    help="Root of acdc_sfb4_spectral runs")
    ap.add_argument("--acdc-base-dir",      default=None,
                    help="Root of acdc_sfb4 baseline runs")
    ap.add_argument("--show", action="store_true", help="Display figures interactively")
    args = ap.parse_args()

    # Auto-collect numeric results if any dir is given
    auto_results = None
    if any([args.kitti_spectral_dir, args.kitti_base_dir,
            args.acdc_spectral_dir,  args.acdc_base_dir]):
        print("\nAuto-collecting results from checkpoint dirs...")
        auto_results = auto_collect_results(
            kitti_spectral_root=args.kitti_spectral_dir,
            kitti_base_root=args.kitti_base_dir,
            acdc_spectral_root=args.acdc_spectral_dir,
            acdc_base_root=args.acdc_base_dir,
        )

    def load(d):
        if d is None:
            return None
        w = _load_cfa_weights(d)
        if w is None:
            print(f"  [warn] No CFA weights found in {d}")
        else:
            T = _tile_size(w.shape[0])
            print(f"  Loaded tile{T} weights from {d}  shape={w.shape}")
        return w

    # Auto-discover tile dirs from spectral root if not explicitly given
    def _auto_tile_dir(explicit, root, *exp_names):
        if explicit:
            return explicit
        if not root:
            return None
        for name in exp_names:
            d = _find_seed0(root, name)
            if d:
                return d
        return None

    w2_dir = _auto_tile_dir(args.tile2_dir, args.kitti_spectral_dir,
                            "ablate_no_optics_sfb4", "cfa_only_sfb4")
    w3_dir = _auto_tile_dir(args.tile3_dir, args.kitti_spectral_dir, "cfa_tile3_sfb4")
    w4_dir = _auto_tile_dir(args.tile4_dir, args.kitti_spectral_dir, "cfa_tile4_sfb4")

    w2 = load(w2_dir)
    w3 = load(w3_dir)
    w4 = load(w4_dir)

    do_all = args.all

    if args.pipeline or do_all:
        plot_pipeline()

    if args.cfa or do_all:
        if w3 is None and w4 is None:
            print("[warn] --cfa: no tile3/tile4 dirs found; skipping CFA comparison")
        else:
            plot_cfa_comparison(w2, w3, w4)

    if args.bars or do_all:
        if w2 is None and w3 is None and w4 is None:
            print("[warn] --bars: no CFA weight dirs found; skipping")
        else:
            plot_spectral_bars(w2, w3, w4)

    if args.mosaic or do_all:
        if w3 is None and w4 is None:
            print("[warn] --mosaic: no tile3/tile4 dirs found; skipping")
        else:
            plot_mosaic_pattern(w2, w3, w4)

    if args.results or do_all:
        plot_results(results=auto_results)

    if args.trend or do_all:
        # Pull tile mIoU from auto_results for tile trend
        kitti_tile = acdc_tile = None
        if auto_results:
            kr = auto_results.get("KITTI-360", {})
            ar = auto_results.get("ACDC", {})
            kitti_tile = {
                2: kr.get("No optics ★"),
                3: kr.get("Tile-3 (ablation)"),
                4: kr.get("Tile-4 (ablation)"),
            }
            acdc_tile = {
                2: ar.get("No optics ★"),
                3: ar.get("Tile-3 (ablation)"),
                4: ar.get("Tile-4 (ablation)"),
            }
        plot_tile_trend(kitti_miou=kitti_tile, acdc_miou=acdc_tile)

    if args.ablation or do_all:
        plot_ablation_decomposition(results=auto_results)

    if args.latex or do_all:
        out_tex = os.path.join(OUT_DIR, "table_main_results.tex")
        tbl = generate_latex_table(results=auto_results, out_file=out_tex)
        print("\n── LaTeX table ──────────────────────────────────────────────────")
        print(tbl)

    if args.show:
        plt.show()

    print(f"\nAll outputs saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
