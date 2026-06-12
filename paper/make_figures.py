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
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          False,
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
    """Side-by-side CFA tile pattern visualisation: Bayer | tile2 | tile3 | tile4."""
    bayer  = _bayer_weights()
    panels = [
        (bayer,        "Bayer 2×2\n(Fixed, RGGB)"),
        (tile2_weights, "Learned 2×2\n(CFA-only)"),
        (tile3_weights, "Learned 3×3\n(Spectral-3, Ours)"),
        (tile4_weights, "Learned 4×4\n(Spectral-4, Ours)"),
    ]

    n_panels = sum(1 for w, _ in panels if w is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=(2.2 * n_panels, 2.8))
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

    fig.suptitle("Task-Optimal Spectral Mosaic Designs\n"
                 "(cell colour = learned spectral response; letter = dominant channel)",
                 fontsize=8.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])

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
        ax.set_xticklabels([f"s{i}" for i in x], fontsize=6)
        ax.set_ylim(0, 1.05)
        ax.set_title(label, fontsize=8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        div = _spectral_diversity(weights)
        ax.text(0.97, 0.96, f"div={div:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.5, color="#555555")

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

# Fill these in when all jobs finish (from monitor.py).
RESULTS = {
    "KITTI-360": {
        "RGB (no sensor)":      0.6396,
        "Fixed camera":         0.6043,
        "Co-design (PSF+CFA)":  None,      # fill after job finishes
        "No optics (tile2)":    0.6229,
        "CFA only (tile2)":     None,      # job 662396
        "Spectral-3 (Ours)":    None,      # job 662397
        "Spectral-4 (Ours)":    None,      # job 662398
    },
    "ACDC": {
        "RGB (no sensor)":      0.7374,
        "Fixed camera":         0.6654,
        "Co-design (PSF+CFA)":  0.6906,
        "No optics (tile2)":    0.6992,
        "CFA only (tile2)":     None,      # job 662399
        "Spectral-3 (Ours)":    None,      # job 662400
        "Spectral-4 (Ours)":    None,      # job 662401
    },
}

EXP_COLORS = {
    "RGB (no sensor)":      "#AAAAAA",
    "Fixed camera":         "#999999",
    "Co-design (PSF+CFA)":  "#F4CCCC",
    "No optics (tile2)":    "#BDD7EE",
    "CFA only (tile2)":     "#93C4E0",
    "Spectral-3 (Ours)":    "#FF9900",
    "Spectral-4 (Ours)":    "#E65C00",
}


def plot_results(results: dict | None = None, save: bool = True) -> plt.Figure:
    """Grouped bar chart of mIoU results across all experiments and datasets."""
    if results is None:
        results = RESULTS

    datasets = list(results.keys())
    exp_names = list(results[datasets[0]].keys())
    n_exps = len(exp_names)
    n_ds   = len(datasets)

    fig, axes = plt.subplots(1, n_ds, figsize=(4.5 * n_ds, 3.4), sharey=False)
    if n_ds == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        data  = results[dataset]
        vals  = [data[e] for e in exp_names]
        x     = np.arange(n_exps)
        bars  = []
        for i, (name, val) in enumerate(zip(exp_names, vals)):
            color = EXP_COLORS.get(name, "#CCCCCC")
            alpha = 1.0 if val is not None else 0.35
            height = val if val is not None else 0.0
            b = ax.bar(i, height, 0.65, color=color, alpha=alpha,
                       edgecolor="white", linewidth=0.8, zorder=3)
            bars.append(b)
            if val is not None:
                ax.text(i, val + 0.003, f"{val:.4f}", ha="center", va="bottom",
                        fontsize=5.5, rotation=90, color="#333333")
            else:
                ax.text(i, 0.01, "pending", ha="center", va="bottom",
                        fontsize=5.5, rotation=90, color="#999999")

        ax.set_title(dataset, fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, rotation=35, ha="right", fontsize=6.5)
        ax.set_ylabel("mIoU", fontsize=8)
        ax.set_ylim(0.55, max(v for v in vals if v) * 1.08)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5, zorder=0)

        # Bracket: highlight our methods
        ours_idx = [i for i, n in enumerate(exp_names) if "Ours" in n]
        if ours_idx:
            y_top = ax.get_ylim()[1] * 0.97
            ax.annotate(
                "", xy=(ours_idx[-1], y_top), xytext=(ours_idx[0], y_top),
                arrowprops=dict(arrowstyle="<->", color="#E65C00", lw=1.5),
            )
            ax.text(
                np.mean(ours_idx), y_top + 0.002,
                "Ours", ha="center", va="bottom", fontsize=7,
                color="#E65C00", fontweight="bold",
            )

    fig.suptitle("mIoU Comparison — Task-Optimal Spectral CFA Design",
                 fontsize=9.5, fontweight="bold")
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
    # Replace None with actual numbers once jobs finish
    if kitti_miou is None:
        kitti_miou = {2: 0.6229, 3: None, 4: None}   # 2 = no_optics v2
    if acdc_miou is None:
        acdc_miou  = {2: 0.6992, 3: None, 4: None}

    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    sizes = [2, 3, 4]

    for label, data, color, marker in [
        ("KITTI-360", kitti_miou, "#1F77B4", "o"),
        ("ACDC",      acdc_miou,  "#FF7F0E", "s"),
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


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate paper figures.")
    ap.add_argument("--pipeline", action="store_true")
    ap.add_argument("--cfa",      action="store_true")
    ap.add_argument("--bars",     action="store_true")
    ap.add_argument("--mosaic",   action="store_true")
    ap.add_argument("--results",  action="store_true")
    ap.add_argument("--trend",    action="store_true")
    ap.add_argument("--all",      action="store_true")
    ap.add_argument("--tile2-dir", default=None, help="Run dir for no_optics (tile2)")
    ap.add_argument("--tile3-dir", default=None, help="Run dir for cfa_tile3")
    ap.add_argument("--tile4-dir", default=None, help="Run dir for cfa_tile4")
    ap.add_argument("--show",     action="store_true", help="Display figures interactively")
    args = ap.parse_args()

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

    w2 = load(args.tile2_dir)
    w3 = load(args.tile3_dir)
    w4 = load(args.tile4_dir)

    do_all = args.all

    if args.pipeline or do_all:
        plot_pipeline()

    if args.cfa or do_all:
        if w3 is None and w4 is None:
            print("[warn] --cfa needs at least one of --tile3-dir / --tile4-dir")
        plot_cfa_comparison(w2, w3, w4)

    if args.bars or do_all:
        plot_spectral_bars(w2, w3, w4)

    if args.mosaic or do_all:
        plot_mosaic_pattern(w2, w3, w4)

    if args.results or do_all:
        plot_results()

    if args.trend or do_all:
        plot_tile_trend()

    if args.show:
        plt.show()

    print(f"\nAll figures saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
