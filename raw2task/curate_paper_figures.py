"""Curate qualitative figures into narrative, paper-facing examples.

The training loop saves many diagnostic panels. This script selects and copies
only examples that tell a specific story: strong normal driving scenes,
thin/dynamic-object failures, camera-stage evidence, and recovery examples.
Each selected image gets a sidecar JSON/TXT caption explaining why it belongs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PANEL_TITLES = [
    "Input RGB",
    "PSF residual",
    "Exposure image",
    "CFA/RAW view",
    "Noise/ADC residual",
    "GT",
    "Pred",
]

COLOR_TO_CLASS = {
    (128, 64, 128): "road",
    (244, 35, 232): "sidewalk",
    (70, 70, 70): "building",
    (102, 102, 156): "wall",
    (190, 153, 153): "fence",
    (153, 153, 153): "pole",
    (250, 170, 30): "traffic light",
    (220, 220, 0): "traffic sign",
    (107, 142, 35): "vegetation",
    (152, 251, 152): "terrain",
    (70, 130, 180): "sky",
    (220, 20, 60): "person",
    (255, 0, 0): "rider",
    (0, 0, 142): "car",
    (0, 0, 70): "truck",
    (0, 60, 100): "bus",
    (0, 80, 100): "train",
    (0, 0, 230): "motorcycle",
    (119, 11, 32): "bicycle",
}

THIN_DYNAMIC = {"person", "rider", "bicycle", "motorcycle", "traffic light", "traffic sign", "pole"}

STORY_LABELS = {
    "normal_driving_context": "Normal driving context",
    "thin_dynamic_objects": "Thin and rare classes",
    "camera_stage_evidence": "Learned camera evidence",
    "hard_lighting_case": "Hard lighting case",
}

STORY_INTENT = {
    "normal_driving_context": (
        "A representative road scene with multiple semantic regions, used as the main qualitative sanity check."
    ),
    "thin_dynamic_objects": (
        "A case containing small or thin classes, used to discuss the reviewer concern about rare-class IoU."
    ),
    "camera_stage_evidence": (
        "A case where the optics/CFA/noise panels show structured, image-dependent camera behavior."
    ),
    "hard_lighting_case": (
        "A scene with shadow or highlight stress, used to motivate learned exposure and sensor robustness."
    ),
}


@dataclass
class FigureStory:
    source: str
    artifact: str
    caption_txt: str
    caption_json: str
    experiment: str
    seed: str
    story_type: str
    score: float
    classes: str
    rationale: str


def _iter_viz(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("viz/*.png") if p.is_file())


def _infer_tiles(img: np.ndarray, ncols: int = 7, footer_h: int = 34, padding: int = 4, tile_aspect: float = 0.30):
    h, w = img.shape[:2]
    tile_w = (w - padding * (ncols + 1)) // ncols
    expected_tile_h = max(1, int(round(tile_w * tile_aspect)))
    # Use floor so a global caption below the grid is not mistaken for another row.
    nrows = max(1, int((h - padding) // max(expected_tile_h + footer_h + padding, 1)))
    tile_h = expected_tile_h
    tiles = []
    for r in range(nrows):
        row = []
        for c in range(ncols):
            x = padding + c * (tile_w + padding)
            y = padding + r * (tile_h + footer_h + padding)
            row.append(img[y : y + tile_h, x : x + tile_w])
        tiles.append(row)
    return tiles


def _class_coverage(tile: np.ndarray) -> Dict[str, float]:
    flat = tile.reshape(-1, 3)
    n = max(1, flat.shape[0])
    out: Dict[str, float] = {}
    for rgb, name in COLOR_TO_CLASS.items():
        count = int(np.all(flat == np.array(rgb, dtype=np.uint8), axis=1).sum())
        if count:
            out[name] = count / n
    return out


def _rgb_stats(tile: np.ndarray) -> Dict[str, float]:
    gray = tile.astype(np.float32).mean(axis=2)
    return {
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "dark_frac": float((gray < 35).mean()),
        "bright_frac": float((gray > 220).mean()),
    }


def _stage_strength(tile: np.ndarray) -> float:
    gray = tile.astype(np.float32).mean(axis=2)
    return float(gray.std() + 0.25 * (gray.max() - gray.min()))


def _parse_exp_seed(path: Path) -> Tuple[str, str]:
    run = path.parent.parent.name
    m = re.match(r"(.+)_seed(\d+)$", run)
    if m:
        return m.group(1), m.group(2)
    return run, ""


def _score_image(path: Path) -> List[Tuple[str, float, int, str, str]]:
    img = np.array(Image.open(path).convert("RGB"))
    tiles = _infer_tiles(img)
    scores: List[Tuple[str, float, int, str, str]] = []
    for row_idx, row in enumerate(tiles):
        rgb = row[0]
        optics = row[1]
        exposure = row[2]
        cfa = row[3]
        noise = row[4]
        gt = row[5]
        pred = row[6]
        cov = _class_coverage(gt)
        pred_cov = _class_coverage(pred)
        stats = _rgb_stats(rgb)
        classes = sorted(cov, key=lambda k: cov[k], reverse=True)
        class_text = ", ".join(classes[:8])
        class_count = len(classes)

        road_context = cov.get("road", 0.0) + cov.get("sidewalk", 0.0)
        dynamic = sum(cov.get(k, 0.0) for k in THIN_DYNAMIC)
        vehicle = cov.get("car", 0.0) + cov.get("truck", 0.0) + cov.get("bus", 0.0)
        camera_stage = _stage_strength(optics) + 0.5 * _stage_strength(cfa) + 0.5 * _stage_strength(noise)
        pred_change = sum(abs(cov.get(k, 0.0) - pred_cov.get(k, 0.0)) for k in set(cov) | set(pred_cov))
        dynamic_names = ", ".join(sorted(set(classes) & THIN_DYNAMIC))

        # Quality gates prevent accidental "paper figures" that are blank,
        # class-poor, or only extreme diagnostic color maps without a scene.
        scene_ok = class_count >= 3 and stats["std"] >= 12.0
        normal_ok = scene_ok and road_context >= 0.12 and class_count >= 4
        thin_ok = scene_ok and dynamic >= 0.0015
        camera_ok = scene_ok and camera_stage >= 30.0 and road_context >= 0.05
        lighting_ok = scene_ok and (stats["dark_frac"] + stats["bright_frac"]) >= 0.06

        normal_score = 2.0 * road_context + 0.8 * vehicle + 0.5 * cov.get("sky", 0.0) - 0.4 * stats["dark_frac"]
        thin_score = 8.0 * dynamic + 0.8 * pred_change + 0.2 * road_context
        camera_score = 0.01 * camera_stage + 0.4 * road_context + 0.2 * vehicle
        lowlight_score = 4.0 * stats["dark_frac"] + 2.0 * stats["bright_frac"] + 0.3 * road_context

        if normal_ok:
            scores.append((
                "normal_driving_context",
                normal_score,
                row_idx,
                class_text,
                f"Row {row_idx}: broad road/sidewalk context with vehicles and sky/building cues.",
            ))
        if thin_ok:
            scores.append((
                "thin_dynamic_objects",
                thin_score,
                row_idx,
                class_text,
                f"Row {row_idx}: contains thin or dynamic classes ({dynamic_names}).",
            ))
        if camera_ok:
            scores.append((
                "camera_stage_evidence",
                camera_score,
                row_idx,
                class_text,
                f"Row {row_idx}: camera-stage panels have structured residuals instead of random artifacts.",
            ))
        if lighting_ok:
            scores.append((
                "hard_lighting_case",
                lowlight_score,
                row_idx,
                class_text,
                f"Row {row_idx}: visible shadow/highlight stress for exposure and sensor discussion.",
            ))
    return scores


def _caption(story: str, exp: str, seed: str, classes: str, rationale: str) -> str:
    pretty = STORY_LABELS.get(story, story.replace("_", " "))
    intent = STORY_INTENT.get(story, "")
    return (
        f"{pretty}. {intent} Experiment={exp}, seed={seed}. "
        f"Visible classes: {classes or 'not detected from palette'}. "
        f"{rationale} Columns show input RGB, false-color PSF residual, "
        f"exposure image, CFA/RAW view, false-color noise/ADC residual, ground truth, and prediction."
    )


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current: List[str] = []
    for word in words:
        trial = " ".join(current + [word])
        try:
            width = draw.textbbox((0, 0), trial, font=font)[2]
        except Exception:
            width = len(trial) * 8
        if width <= max_width or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines[:3]


def _crop_story_row(src: Path, row_idx: int, caption: str, dst: Path) -> None:
    img = Image.open(src).convert("RGB")
    arr = np.array(img)
    ncols = 7
    footer_h = 34
    padding = 4
    # Recompute from canvas dimensions to keep this independent of tile content.
    h, w = arr.shape[:2]
    tile_w = (w - padding * (ncols + 1)) // ncols
    expected_tile_h = max(1, int(round(tile_w * 0.30)))
    nrows = max(1, int((h - padding) // max(expected_tile_h + footer_h + padding, 1)))
    tile_h = expected_tile_h
    row_idx = max(0, min(int(row_idx), nrows - 1))
    y0 = padding + row_idx * (tile_h + footer_h + padding)
    y1 = min(img.height, y0 + tile_h + footer_h)
    row_img = img.crop((0, y0, img.width, y1))

    caption_h = 76
    canvas = Image.new("RGB", (row_img.width, row_img.height + caption_h), (18, 18, 18))
    canvas.paste(row_img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    except Exception:
        font = None
    for i, line in enumerate(_wrap_text(draw, caption, font, row_img.width - 28)):
        draw.text((14, row_img.height + 10 + i * 21), line, fill=(245, 245, 245), font=font)
    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst)


def curate(root: Path, out_dir: Path, max_per_story: int) -> List[FigureStory]:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates: Dict[str, List[Tuple[float, Path, int, str, str]]] = {}
    for path in _iter_viz(root):
        for story, score, row_idx, classes, rationale in _score_image(path):
            candidates.setdefault(story, []).append((score, path, row_idx, classes, rationale))

    rows: List[FigureStory] = []
    for story, vals in sorted(candidates.items()):
        picked_paths = set()
        vals = sorted(vals, key=lambda x: x[0], reverse=True)
        rank = 0
        for score, src, row_idx, classes, rationale in vals:
            dedupe_key = (src, row_idx)
            if dedupe_key in picked_paths:
                continue
            picked_paths.add(dedupe_key)
            rank += 1
            exp, seed = _parse_exp_seed(src)
            stem = f"{story}_{rank:02d}_{exp}_seed{seed}_row{row_idx}_{src.stem}"
            dst = out_dir / f"{stem}.png"
            caption_txt = out_dir / f"{stem}.txt"
            caption_json = out_dir / f"{stem}.json"
            shutil.copy2(src, dst)
            caption = _caption(story, exp, seed, classes, rationale)
            _crop_story_row(src, row_idx=row_idx, caption=caption, dst=dst)
            caption_txt.write_text(caption + "\n")
            caption_json.write_text(json.dumps({
                "story_type": story,
                "source": str(src),
                "artifact": str(dst),
                "experiment": exp,
                "seed": seed,
                "row": row_idx,
                "score": score,
                "classes": classes,
                "caption": caption,
                "rationale": rationale,
            }, indent=2) + "\n")
            rows.append(FigureStory(
                source=str(src),
                artifact=str(dst),
                caption_txt=str(caption_txt),
                caption_json=str(caption_json),
                experiment=exp,
                seed=seed,
                story_type=story,
                score=float(score),
                classes=classes,
                rationale=rationale,
            ))
            if rank >= max_per_story:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/industry_paper_matrix")
    parser.add_argument("--out-dir", default="runs/paper_story_figures")
    parser.add_argument("--max-per-story", type=int, default=4)
    args = parser.parse_args()

    rows = curate(Path(args.root), Path(args.out_dir), max_per_story=args.max_per_story)
    manifest = Path(args.out_dir) / "manifest.csv"
    with manifest.open("w", newline="") as f:
        fields = list(FigureStory.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    print(f"Curated {len(rows)} story figures into {args.out_dir}")
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
