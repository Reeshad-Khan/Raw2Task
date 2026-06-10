"""Audit raw2task qualitative panels for paper readiness.

The script checks each tile inside saved stage grids for common figure failures:
blank/near-constant panels, heavy saturation, low dynamic range, and malformed
layout. It does not judge semantic quality; it flags representation problems.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image


DEFAULT_TITLES = [
    "Input RGB",
    "Optics effect",
    "Exposure output",
    "CFA mosaic",
    "Noise/ADC residual",
    "GT",
    "Pred",
]


@dataclass
class TileAudit:
    path: str
    row: int
    col: int
    title: str
    mean: float
    std: float
    min_value: int
    max_value: int
    low_clip_frac: float
    high_clip_frac: float
    status: str
    reason: str


def _iter_pngs(paths: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            out.extend(sorted(p.rglob("*.png")))
        elif p.is_file() and p.suffix.lower() == ".png":
            out.append(p)
    return sorted(set(out))


def _infer_grid(
    img: np.ndarray,
    ncols: int,
    footer_h: int,
    padding: int,
    tile_aspect: float,
) -> tuple[int, int, int]:
    h, w = img.shape[:2]
    tile_w = (w - padding * (ncols + 1)) // ncols
    if tile_w <= 0:
        raise ValueError(f"Cannot infer tile width for image shape={img.shape}")
    expected_h = max(1, int(round(tile_w * float(tile_aspect))))
    row_pitch = expected_h + footer_h + padding
    nrows = max(1, int(round((h - padding) / max(row_pitch, 1))))
    tile_h = max(1, (h - padding * (nrows + 1)) // nrows - footer_h)
    return nrows, tile_h, tile_w


def _audit_tile(
    tile: np.ndarray,
    path: Path,
    row: int,
    col: int,
    title: str,
    std_threshold: float,
    low_dyn_threshold: float,
    clip_threshold: float,
) -> TileAudit:
    gray = tile.astype(np.float32).mean(axis=2)
    mean = float(gray.mean())
    std = float(gray.std())
    min_value = int(tile.min())
    max_value = int(tile.max())
    low_clip = float((tile <= 2).mean())
    high_clip = float((tile >= 253).mean())
    dyn = max_value - min_value

    reasons = []
    if std < std_threshold:
        reasons.append(f"near-blank std={std:.2f}")
    if dyn < low_dyn_threshold:
        reasons.append(f"low dynamic range={dyn}")
    if high_clip > clip_threshold:
        reasons.append(f"high saturation={100.0 * high_clip:.1f}%")
    if low_clip > clip_threshold and title not in ("GT", "Pred"):
        reasons.append(f"black clipping={100.0 * low_clip:.1f}%")

    return TileAudit(
        path=str(path),
        row=row,
        col=col,
        title=title,
        mean=mean,
        std=std,
        min_value=min_value,
        max_value=max_value,
        low_clip_frac=low_clip,
        high_clip_frac=high_clip,
        status="ok" if not reasons else "flag",
        reason="; ".join(reasons),
    )


def audit_png(
    path: Path,
    ncols: int,
    footer_h: int,
    padding: int,
    std_threshold: float,
    low_dyn_threshold: float,
    clip_threshold: float,
    tile_aspect: float,
) -> List[TileAudit]:
    img = np.array(Image.open(path).convert("RGB"))
    nrows, tile_h, tile_w = _infer_grid(
        img,
        ncols=ncols,
        footer_h=footer_h,
        padding=padding,
        tile_aspect=tile_aspect,
    )
    rows: List[TileAudit] = []
    for r in range(nrows):
        for c in range(ncols):
            x = padding + c * (tile_w + padding)
            y = padding + r * (tile_h + footer_h + padding)
            tile = img[y : y + tile_h, x : x + tile_w]
            if tile.shape[0] < 8 or tile.shape[1] < 8:
                continue
            title = DEFAULT_TITLES[c] if c < len(DEFAULT_TITLES) else f"col{c}"
            rows.append(
                _audit_tile(
                    tile,
                    path=path,
                    row=r,
                    col=c,
                    title=title,
                    std_threshold=std_threshold,
                    low_dyn_threshold=low_dyn_threshold,
                    clip_threshold=clip_threshold,
                )
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="PNG files or directories to audit.")
    parser.add_argument("--out-csv", default="runs/paper_figure_audit.csv")
    parser.add_argument("--out-json", default="runs/paper_figure_audit.json")
    parser.add_argument("--ncols", type=int, default=7)
    parser.add_argument("--footer-h", type=int, default=34)
    parser.add_argument("--padding", type=int, default=4)
    parser.add_argument("--std-threshold", type=float, default=3.0)
    parser.add_argument("--low-dyn-threshold", type=float, default=12.0)
    parser.add_argument("--clip-threshold", type=float, default=0.35)
    parser.add_argument("--tile-aspect", type=float, default=0.30, help="Tile height / tile width; KITTI panels are about 0.30.")
    args = parser.parse_args()

    pngs = _iter_pngs(args.paths)
    audits: List[TileAudit] = []
    for p in pngs:
        try:
            audits.extend(
                audit_png(
                    p,
                    ncols=args.ncols,
                    footer_h=args.footer_h,
                    padding=args.padding,
                    std_threshold=args.std_threshold,
                    low_dyn_threshold=args.low_dyn_threshold,
                    clip_threshold=args.clip_threshold,
                    tile_aspect=args.tile_aspect,
                )
            )
        except Exception as exc:
            audits.append(
                TileAudit(
                    path=str(p),
                    row=-1,
                    col=-1,
                    title="image",
                    mean=0.0,
                    std=0.0,
                    min_value=0,
                    max_value=0,
                    low_clip_frac=0.0,
                    high_clip_frac=0.0,
                    status="flag",
                    reason=f"failed to audit: {exc}",
                )
            )

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        fieldnames = list(TileAudit.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in audits:
            writer.writerow(asdict(row))
    with open(args.out_json, "w") as f:
        json.dump([asdict(row) for row in audits], f, indent=2)

    flagged = [row for row in audits if row.status != "ok"]
    print(f"Audited {len(pngs)} PNGs, {len(audits)} tiles.")
    print(f"Flagged {len(flagged)} tiles.")
    for row in flagged[:40]:
        print(f"[flag] {row.path} row={row.row} col={row.col} {row.title}: {row.reason}")
    if len(flagged) > 40:
        print(f"... {len(flagged) - 40} more flags in {args.out_csv}")
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
