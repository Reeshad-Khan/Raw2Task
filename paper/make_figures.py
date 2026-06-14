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
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ── Publication style — CVPR/ECCV 2024-2025 conventions ───────────────────────
# Sans-serif in figures (even if LaTeX body uses serif) is now dominant at
# top-tier venues. pdf.fonttype=42 embeds fonts (required by many venues).

mpl.rcParams.update({
    # Font — sans-serif matches most CVPR 2024 papers
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset":     "dejavusans",
    "pdf.fonttype":         42,    # embed fonts in PDF (prevents conference rejection)
    "ps.fonttype":          42,
    # Sizes
    "font.size":            9,
    "axes.titlesize":       9.5,
    "axes.labelsize":       8.5,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "legend.fontsize":      8,
    "legend.title_fontsize": 8.5,
    # Output quality
    "figure.dpi":           150,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.04,
    # Axes — minimal, clean
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.linewidth":       0.6,
    "axes.axisbelow":       True,   # gridlines behind data
    # Ticks
    "xtick.major.width":    0.6,
    "ytick.major.width":    0.6,
    "xtick.major.size":     3,
    "ytick.major.size":     3,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    # Lines
    "lines.linewidth":      1.8,
    "patch.linewidth":      0.6,
    # Grid — subtle horizontal lines only (set per-axis below)
    "axes.grid":            True,
    "grid.color":           "#E0E0E0",
    "grid.linewidth":       0.7,
    "grid.alpha":           1.0,
})

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colour palette — Seaborn colorblind (accessible, CVPR-standard) ─────────
# Source: seaborn.color_palette("colorblind")
CB = {
    "blue":   "#0173B2",   # primary — our contribution
    "orange": "#DE8F05",   # tile-3 ablation
    "green":  "#029E73",   # positive gains
    "red":    "#D55E00",   # PSF cost / tile-4
    "purple": "#CC78BC",
    "brown":  "#CA9161",
    "sky":    "#56B4E9",   # lighter blue — CFA only
    "gray":   "#949494",   # neutral / fixed camera
    "lgray":  "#CCCCCC",   # light gray — RGB oracle
    "dkblue": "#01579B",   # dark blue — No optics (best)
}

# Pipeline box colours
FIXED_COLOR    = "#E3F2FD"   # very light blue — fixed component
LEARNED_COLOR  = "#0173B2"   # seaborn blue — our learned component
SKIP_COLOR     = "#FFEBEE"   # very light pink — identity (bypassed)
BACKBONE_COLOR = "#E8F5E9"   # very light green — pretrained backbone
ARROW_COLOR    = "#333333"

# Per-channel colours for spectral bar plots
CHANNEL_COLORS = {"R": "#E53935", "G": "#43A047", "B": "#1E88E5"}

# Method colours — consistent across all figures
EXP_COLORS = {
    "RGB":                 CB["lgray"],  # oracle ceiling
    "Fixed camera":        CB["gray"],   # baseline to beat
    "Co-design (PSF+CFA)": "#F4A582",    # full co-design (worse than no-optics)
    "No optics ★":         CB["dkblue"], # best sensor — primary contribution
    "CFA only":            CB["sky"],    # CFA without noise co-opt
    "Tile-3 (ablation)":   CB["orange"],
    "Tile-4 (ablation)":   CB["red"],
}

# Display labels for x-tick (shorter, cleaner)
EXP_SHORT = {
    "RGB":                 "RGB\noracle",
    "Fixed camera":        "Fixed\ncamera",
    "Co-design (PSF+CFA)": "Co-design\n(PSF+CFA)",
    "No optics ★":         "No optics\n(Ours*)",   # * avoids missing-glyph warning
    "CFA only":            "CFA only\n(no noise)",
    "Tile-3 (ablation)":   "Tile-3",
    "Tile-4 (ablation)":   "Tile-4",
}

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
    """Architecture diagram — modern flat-colour box style (CVPR 2024 convention)."""
    fig, ax = plt.subplots(figsize=(8.5, 2.2))
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 2.8)
    ax.axis("off")
    ax.set_facecolor("white")

    # Text colour: dark on light backgrounds, white on dark
    def _txt_color(bg_hex: str) -> str:
        r, g, b = int(bg_hex[1:3], 16), int(bg_hex[3:5], 16), int(bg_hex[5:7], 16)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        return "white" if lum < 140 else "#222222"

    def box(cx, cy, w, h, label, sublabel="", color=FIXED_COLOR, lw=0.8):
        tc = _txt_color(color)
        # Flat rounded rectangle — no heavy shadow
        rect = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.06",
            linewidth=lw,
            edgecolor="#AAAAAA" if color.startswith("#E") else "#888888",
            facecolor=color,
            zorder=3,
        )
        ax.add_patch(rect)
        label_y = cy + (0.10 if sublabel else 0)
        ax.text(cx, label_y, label,
                ha="center", va="center", fontsize=8.5,
                fontweight="bold", color=tc, zorder=4)
        if sublabel:
            ax.text(cx, cy - 0.23, sublabel, ha="center", va="center",
                    fontsize=6.8, color=tc if tc == "white" else "#555555", zorder=4)

    def arrow(x1, x2, y=1.4):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR,
                                   lw=1.0, mutation_scale=9), zorder=2)

    def chip(cx, cy, text, facecolor, edgecolor):
        ax.text(cx, cy, text, ha="center", va="center", fontsize=6.5,
                fontweight="bold", color=_txt_color(facecolor), zorder=5,
                bbox=dict(boxstyle="round,pad=0.20", facecolor=facecolor,
                          edgecolor=edgecolor, linewidth=0.6))

    # ── Boxes ─────────────────────────────────────────────────────────────────
    box(0.7,  1.4, 1.1, 0.9, "Scene",        "RGB input",      "#F5F5F5")
    box(2.3,  1.4, 1.2, 0.9, "Optics",       "Identity PSF",   SKIP_COLOR)
    box(3.88, 1.4, 1.2, 0.9, "T×T Spectral", "CFA Mosaic",     LEARNED_COLOR)
    box(5.46, 1.4, 1.2, 0.9, "Noise + ADC",  "Poisson-Gauss",  "#1565C0")
    box(7.04, 1.4, 1.2, 0.9, "Demosaick",    "Soft, diff'able","#ECEFF1")
    box(8.82, 1.4, 1.5, 0.9, "SegFormer-B4", "Task backbone",  BACKBONE_COLOR)

    # ── Arrows ────────────────────────────────────────────────────────────────
    for x1, x2 in [(1.26, 1.68), (2.91, 3.25), (4.49, 4.83),
                   (6.07, 6.41), (7.65, 7.99)]:
        arrow(x1, x2)
    ax.annotate("", xy=(9.90, 1.4), xytext=(9.59, 1.4),
                arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR,
                                lw=1.0, mutation_scale=9))
    ax.text(9.96, 1.4, "Seg.\nMap", ha="left", va="center", fontsize=7.5,
            color="#333333")

    # ── Chips (status badges) ─────────────────────────────────────────────────
    chip(2.30, 2.10, "Fixed (Identity)",  "#FFCDD2", "#EF9A9A")
    chip(3.88, 2.10, "Learned",           LEARNED_COLOR, CB["dkblue"])
    chip(5.46, 2.10, "Learned",           "#1565C0", "#0D47A1")
    chip(7.04, 2.10, "Fixed",             "#ECEFF1", "#AAAAAA")
    chip(8.82, 2.10, "Pre-trained",       BACKBONE_COLOR, "#A5D6A7")

    # ── Parameter annotation ──────────────────────────────────────────────────
    ax.annotate(
        r"$\mathbf{W} \in \mathbb{R}^{T^2 \times 3}$" + "  (T=2,3,4)",
        xy=(3.88, 0.88), xytext=(3.88, 0.38),
        ha="center", fontsize=7.5, color="#333333",
        arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.7),
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(facecolor=FIXED_COLOR,    edgecolor="#AAAAAA", label="Fixed"),
        mpatches.Patch(facecolor=LEARNED_COLOR,  edgecolor="#888888", label="Learned (ours)"),
        mpatches.Patch(facecolor=SKIP_COLOR,     edgecolor="#AAAAAA", label="Identity (bypassed)"),
        mpatches.Patch(facecolor=BACKBONE_COLOR, edgecolor="#AAAAAA", label="Pre-trained backbone"),
    ]
    leg = ax.legend(handles=legend_patches, loc="lower right", frameon=True,
                    framealpha=0.95, fontsize=7, ncol=2,
                    borderpad=0.5, edgecolor="#DDDDDD")
    leg.get_frame().set_linewidth(0.5)

    fig.suptitle("Task-Optimal Spectral Mosaic Co-Design Pipeline",
                 fontsize=10, fontweight="bold", y=1.02)
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
    """Main results bar chart — CVPR 2024 style.

    Layout:
      • Two side-by-side panels (KITTI-360 | ACDC), each ~3.3 in wide.
      • Y-axis starts just below the fixed-camera baseline — emphasises relative gain.
      • Dashed horizontal line at fixed-camera level; brace annotation showing our gain.
      • Method bars use consistent semantic colour scheme.
      • No top/right spines; light horizontal gridlines behind bars.
    """
    if results is None:
        results = RESULTS

    datasets  = list(results.keys())
    exp_names = list(results[datasets[0]].keys())
    n_ds      = len(datasets)

    # ── Figure canvas ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_ds, figsize=(5.2 * n_ds, 3.6), sharey=False)
    if n_ds == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        data = results[dataset]
        vals = [data[e] for e in exp_names]

        avail = [v for v in vals if v is not None]
        if not avail:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        y_max = max(avail)
        y_min = min(avail)
        span  = y_max - y_min
        # Tight y-range: show differences clearly, not absolute scale
        y_lo  = max(0, y_min - span * 0.35)
        y_hi  = y_max + span * 1.60   # headroom for labels and annotation

        fixed_val = data.get("Fixed camera")
        best_val  = data.get("No optics ★")

        # ── Bars ──────────────────────────────────────────────────────────────
        for i, (name, val) in enumerate(zip(exp_names, vals)):
            color  = EXP_COLORS.get(name, CB["gray"])
            height = val if val is not None else y_lo
            alpha  = 1.0 if val is not None else 0.3

            ax.bar(i, height, 0.62, color=color, alpha=alpha,
                   edgecolor="white", linewidth=0.5, zorder=3,
                   bottom=0)

            if val is not None:
                # Value label — compact, above bar
                ax.text(i, val + span * 0.03,
                        f"{val:.4f}", ha="center", va="bottom",
                        fontsize=6.5, fontweight="600", color="#333333")
            else:
                ax.text(i, y_lo + span * 0.05, "—",
                        ha="center", va="bottom", fontsize=8, color="#BBBBBB")

        # ── Fixed-camera dashed reference line ────────────────────────────────
        if fixed_val is not None:
            ax.axhline(fixed_val, color=CB["gray"], linewidth=1.0,
                       linestyle=(0, (4, 2)), alpha=0.8, zorder=2)
            ax.text(len(exp_names) - 0.5, fixed_val + span * 0.015,
                    "fixed cam.", fontsize=6.5, color=CB["gray"],
                    va="bottom", ha="right", style="italic")

        # ── Gain bracket — significance bar above fixed→best ─────────────────
        fixed_idx = exp_names.index("Fixed camera") if "Fixed camera" in exp_names else None
        best_idx  = next((i for i, n in enumerate(exp_names) if "★" in n), None)
        if (fixed_idx is not None and best_idx is not None
                and fixed_val is not None and best_val is not None):
            gain = best_val - fixed_val
            # Horizontal bracket above all bars — never overlaps value labels
            bracket_y = y_max + span * 0.80
            tip_y     = y_max + span * 0.55
            for bx in [fixed_idx, best_idx]:
                ax.plot([bx, bx], [tip_y, bracket_y],
                        color=CB["dkblue"], lw=0.8, linestyle="--", alpha=0.7)
            ax.annotate("", xy=(best_idx, bracket_y), xytext=(fixed_idx, bracket_y),
                        arrowprops=dict(arrowstyle="<->", lw=0.9,
                                        color=CB["dkblue"], mutation_scale=8))
            mid = (fixed_idx + best_idx) / 2
            ax.text(mid, bracket_y + span * 0.10,
                    f"+{gain:.3f}", fontsize=7.5, color=CB["dkblue"],
                    fontweight="bold", va="bottom", ha="center")

        # ── Axis styling ──────────────────────────────────────────────────────
        ax.set_title(dataset, fontsize=10, fontweight="bold", pad=5)
        ax.set_xticks(range(len(exp_names)))
        short_labels = [EXP_SHORT.get(n, _safe_label(n)) for n in exp_names]
        ax.set_xticklabels(short_labels, fontsize=7,
                           ha="center", multialignment="center")
        ax.set_ylabel("mIoU" if dataset == datasets[0] else "", fontsize=9)
        ax.set_ylim(y_lo, y_hi)
        ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", which="both", length=0)  # no x-ticks, labels only

    fig.suptitle("Segmentation mIoU — Task-Optimal Spectral CFA Design",
                 fontsize=10, fontweight="bold", y=1.01)
    fig.tight_layout(pad=0.8, w_pad=0.6)

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

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    sizes = [2, 3, 4]

    for label, data, color, marker in [
        ("KITTI-360", kitti_miou, CB["dkblue"], "o"),
        ("ACDC",      acdc_miou,  CB["red"],    "s"),
    ]:
        ys  = [data.get(s) for s in sizes]
        xs_ = [s for s, y in zip(sizes, ys) if y is not None]
        ys_ = [y for y in ys if y is not None]
        if xs_:
            ax.plot(xs_, ys_, color=color, marker=marker, linewidth=2.0,
                    markersize=7, label=label, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.8)
        for s, y in zip(sizes, ys):
            if y is not None:
                ax.text(s, y + 0.0015, f"{y:.4f}", ha="center", va="bottom",
                        fontsize=6.5, color=color)

    ax.set_xlabel("CFA tile size T  (T² filters)", fontsize=9)
    ax.set_ylabel("mIoU", fontsize=9)
    ax.set_xticks(sizes)
    ax.set_xticklabels(["2×2\n(4)", "3×3\n(9)", "4×4\n(16)"], fontsize=8)
    ax.legend(frameon=True, framealpha=0.9, fontsize=8,
              edgecolor="#DDDDDD", loc="center left",
              bbox_to_anchor=(0.04, 0.5))
    ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("mIoU vs CFA Tile Size", fontsize=10, fontweight="bold", pad=5)
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
    """Sensor-dimension contribution — horizontal Cleveland dot (lollipop) chart.

    Modern CVPR style: each row is one (dataset, component) combination.
    A dot marks the ΔmIoU; a line connects it to the zero axis.
    Colour encodes direction: positive = teal/green, negative = orange-red.
    ACDC PSF effect is marked '†' (cross-physics comparison, indicative only).
    """
    if results is None:
        results = RESULTS

    # ── Compute deltas ────────────────────────────────────────────────────────
    rows   = []   # (dataset, component, delta, cross_physics)
    for ds in ["KITTI-360", "ACDC"]:
        data = results.get(ds, {})
        fixed    = data.get("Fixed camera")
        cfa_o    = data.get("CFA only")
        no_opt   = data.get("No optics ★")
        codesign = data.get("Co-design (PSF+CFA)")

        if fixed is None:
            continue
        cfa_gain  = (cfa_o  - fixed)    if cfa_o  is not None else None
        noise_eff = (no_opt - cfa_o)    if (no_opt is not None and cfa_o is not None) else None
        psf_eff   = (codesign - no_opt) if (codesign is not None and no_opt is not None) else None

        rows.append((ds, "CFA learning",       cfa_gain,  False))
        rows.append((ds, "Noise optimisation", noise_eff, False))
        rows.append((ds, "PSF optimisation",   psf_eff,   ds == "ACDC"))

    # Build y-positions — group by component with a small gap between datasets
    COMP_ORDER = ["CFA learning", "Noise optimisation", "PSF optimisation"]
    DS_ORDER   = ["KITTI-360", "ACDC"]
    DS_MARKERS = {"KITTI-360": "o", "ACDC": "s"}
    DS_COLORS  = {"KITTI-360": CB["dkblue"], "ACDC": CB["red"]}

    y_labels = []
    y_ticks  = []
    y_pos    = {}
    cur_y    = 0
    for comp in COMP_ORDER:
        for ds in DS_ORDER:
            y_pos[(ds, comp)] = cur_y
            y_labels.append(f"  {ds}")
            y_ticks.append(cur_y)
            cur_y -= 1
        cur_y -= 0.5   # extra gap between component groups

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    all_vals = [r[2] for r in rows if r[2] is not None]
    x_pad    = max(abs(v) for v in all_vals) * 0.35 if all_vals else 0.01
    x_lo     = min(0, min(all_vals)) - x_pad
    x_hi     = max(0, max(all_vals)) + x_pad

    for ds, comp, val, cross in rows:
        if val is None:
            continue
        yp    = y_pos[(ds, comp)]
        color = CB["green"] if val >= 0 else CB["red"]
        alpha = 0.5 if cross else 0.92

        # Lollipop stem
        ax.plot([0, val], [yp, yp], color=color, lw=1.8 if not cross else 1.2,
                alpha=alpha, solid_capstyle="round", zorder=3,
                linestyle="--" if cross else "-")
        # Dot
        ax.scatter(val, yp, color=color, s=60, zorder=5, alpha=alpha,
                   marker=DS_MARKERS[ds],
                   edgecolors="white", linewidths=0.7)

        # Value label
        sign = "+" if val >= 0 else ""
        label_x = val + (x_hi - x_lo) * 0.025 * np.sign(val if val != 0 else 1)
        ha = "left" if val >= 0 else "right"
        suffix = "†" if cross else ""
        ax.text(label_x, yp, f"{sign}{val:.3f}{suffix}",
                ha=ha, va="center", fontsize=7.5,
                color=color, fontweight="600")

    # Component section labels (right side)
    comp_y_centres = {}
    for comp in COMP_ORDER:
        ys = [y_pos[(ds, comp)] for ds in DS_ORDER if (ds, comp) in y_pos]
        comp_y_centres[comp] = np.mean(ys)

    for comp, cy in comp_y_centres.items():
        ax.text(x_hi + (x_hi - x_lo) * 0.35, cy, comp,
                ha="left", va="center", fontsize=8.5,
                fontweight="bold", color="#333333")
        # Horizontal divider between component groups
        ax.axhspan(cy - 1.3, cy + 1.3, alpha=0.03, color="#E0E0E0", zorder=0)

    # Zero spine
    ax.axvline(0, color="#555555", lw=0.9, zorder=2)

    # ── Helps / Hurts shading ─────────────────────────────────────────────────
    ax.axvspan(0, x_hi, alpha=0.04, color="#029E73", zorder=0)
    ax.axvspan(x_lo, 0, alpha=0.05, color="#D55E00", zorder=0)
    ax.text(x_hi * 0.97, max(y_ticks) + 0.3, "benefits (+)",
            ha="right", va="bottom", fontsize=7, color="#029E73", style="italic")
    ax.text(x_lo * 0.97, max(y_ticks) + 0.3, "hurts (-)",
            ha="left", va="bottom", fontsize=7, color="#D55E00", style="italic")

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlim(x_lo, x_hi * 1.80)   # extra right margin for comp labels
    ax.set_ylim(min(y_ticks) - 0.7, max(y_ticks) + 0.8)
    ax.set_xlabel(r"$\Delta$mIoU  (vs Fixed-Camera baseline)", fontsize=9)
    ax.xaxis.grid(True, color="#E0E0E0", linewidth=0.7, zorder=0)
    ax.yaxis.grid(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # Dataset legend — placed in upper-left (negative/hurts zone, away from all data labels)
    legend_handles = [
        mlines.Line2D([], [], color=CB["dkblue"], marker="o", linewidth=0,
                      markersize=7, markeredgecolor="white", markeredgewidth=0.7,
                      label="KITTI-360"),
        mlines.Line2D([], [], color=CB["red"], marker="s", linewidth=0,
                      markersize=7, markeredgecolor="white", markeredgewidth=0.7,
                      label="ACDC  (†cross-physics PSF)"),
    ]
    ax.legend(handles=legend_handles, fontsize=7.5, frameon=True,
              framealpha=0.92, loc="upper left", ncol=1, edgecolor="#DDDDDD",
              handletextpad=0.4, borderpad=0.6)

    ax.set_title("Sensor Dimension Contribution to Segmentation mIoU",
                 fontsize=10, fontweight="bold", pad=6)

    fig.text(0.5, -0.03,
             "† ACDC PSF uses old-physics co-design vs new-physics no-optics (directional).",
             ha="center", fontsize=6.5, style="italic", color="#777777")

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
