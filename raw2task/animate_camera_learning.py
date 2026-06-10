"""Create GIF evidence of optics-sensor learning from design preview snapshots."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


EPOCH_RE = re.compile(r"^(initial|best|epoch(\d+))_(.+)\.png$")


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _scan(preview_dir: Path, suffix: str) -> List[Tuple[int, str, Path]]:
    items: List[Tuple[int, str, Path]] = []
    for path in preview_dir.glob(f"*_{suffix}.png"):
        match = EPOCH_RE.match(path.name)
        if not match:
            continue
        tag = match.group(1)
        metric = match.group(3)
        if metric != suffix:
            continue
        if tag == "initial":
            order = -1
            label = "initial"
        elif tag == "best":
            order = 10**9
            label = "best"
        else:
            order = int(match.group(2))
            label = f"epoch {order}"
        items.append((order, label, path))
    return sorted(items, key=lambda x: x[0])


def _letterbox(img: Image.Image, size: Tuple[int, int], bg: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    img = img.convert("RGB")
    w, h = size
    scale = min(w / max(img.width, 1), h / max(img.height, 1))
    nw = max(1, int(round(img.width * scale)))
    nh = max(1, int(round(img.height * scale)))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, bg)
    canvas.paste(resized, ((w - nw) // 2, (h - nh) // 2))
    return canvas


def _title_frame(
    panels: Sequence[Tuple[str, Image.Image]],
    label: str,
    title: str,
    subtitle: str,
    width: int = 1600,
    panel_height: int = 780,
) -> Image.Image:
    header_h = 135
    footer_h = 55
    gap = 18
    panel_w = (width - gap * (len(panels) + 1)) // max(len(panels), 1)
    height = header_h + panel_height + footer_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(44)
    label_font = _font(28)
    small_font = _font(22)
    draw.text((width // 2, 28), title, fill=(0, 0, 0), font=title_font, anchor="ma")
    draw.text((width // 2, 82), subtitle, fill=(35, 35, 35), font=small_font, anchor="ma")
    draw.text((32, 34), label, fill=(0, 0, 0), font=label_font)
    x = gap
    for panel_title, img in panels:
        y = header_h
        draw.rectangle([x, y, x + panel_w, y + panel_height], outline=(55, 55, 55), width=3)
        fitted = _letterbox(img, (panel_w - 16, panel_height - 66))
        canvas.paste(fitted, (x + 8, y + 46))
        draw.text((x + panel_w // 2, y + 14), panel_title, fill=(0, 0, 0), font=small_font, anchor="ma")
        x += panel_w + gap
    draw.text(
        (width // 2, header_h + panel_height + 28),
        "Training snapshots show camera parameters; the deployed camera is frozen at inference.",
        fill=(35, 35, 35),
        font=small_font,
        anchor="ma",
    )
    return canvas


def _save_gif(frames: Sequence[Image.Image], out_path: Path, duration_ms: int) -> None:
    if not frames:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first, *rest = frames
    first.save(
        out_path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def _scan_csv(preview_dir: Path, suffix: str) -> List[Tuple[int, str, Path]]:
    items: List[Tuple[int, str, Path]] = []
    for path in preview_dir.glob(f"*_{suffix}.csv"):
        match = EPOCH_RE.match(path.with_suffix(".png").name)
        if not match:
            continue
        tag = match.group(1)
        metric = match.group(3)
        if metric != suffix:
            continue
        if tag == "initial":
            order = -1
            label = "initial"
        elif tag == "best":
            order = 10**9
            label = "best"
        else:
            order = int(match.group(2))
            label = f"epoch {order}"
        items.append((order, label, path))
    return sorted(items, key=lambda x: x[0])


def _read_psf_summary(path: Path) -> Dict[str, float]:
    rows = []
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append(
                    {
                        "zone": float(row.get("zone", 0.0)),
                        "channel": float(row.get("channel", 0.0)),
                        "rms": float(row.get("rms_radius_px", 1.0)),
                        "sx": float(row.get("shift_x_px", 0.0)),
                        "sy": float(row.get("shift_y_px", 0.0)),
                        "peak": float(row.get("peak", 0.0)),
                    }
                )
            except Exception:
                continue
    if not rows:
        return {"rms": 1.0, "field": 0.0, "chromatic": 0.0, "shift": 0.0, "peak": 0.2}
    rms_vals = [r["rms"] for r in rows]
    shifts = [(r["sx"] ** 2 + r["sy"] ** 2) ** 0.5 for r in rows]
    by_ch: Dict[int, List[float]] = {}
    by_zone: Dict[int, List[float]] = {}
    for r in rows:
        by_ch.setdefault(int(r["channel"]), []).append(r["rms"])
        by_zone.setdefault(int(r["zone"]), []).append(r["rms"])
    ch_means = [sum(v) / len(v) for v in by_ch.values() if v]
    zone_means = [sum(v) / len(v) for v in by_zone.values() if v]
    return {
        "rms": sum(rms_vals) / len(rms_vals),
        "field": (max(zone_means) - min(zone_means)) if len(zone_means) > 1 else 0.0,
        "chromatic": (max(ch_means) - min(ch_means)) if len(ch_means) > 1 else 0.0,
        "shift": sum(shifts) / len(shifts),
        "peak": sum(r["peak"] for r in rows) / len(rows),
    }


def _draw_dotted_background(draw: ImageDraw.ImageDraw, size: int = 640) -> None:
    for x in range(0, size, 6):
        for y in range(0, size, 6):
            color = (204, 242, 245) if (x + y) % 12 == 0 else (248, 214, 176)
            draw.point((x, y), fill=color)


def _lens_profile(x: int, y_top: int, y_bottom: int, curve: float, lean: float) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    mid = (y_top + y_bottom) / 2.0
    half = (y_bottom - y_top) / 2.0
    left: List[Tuple[int, int]] = []
    right: List[Tuple[int, int]] = []
    for i in range(28):
        t = i / 27.0
        y = y_top + t * (y_bottom - y_top)
        u = (y - mid) / max(half, 1.0)
        bulge = curve * (1.0 - u * u)
        tilt = lean * u
        left.append((int(round(x - bulge + tilt)), int(round(y))))
        right.append((int(round(x + bulge + tilt)), int(round(y))))
    return left, right


def _draw_lens_element(
    draw: ImageDraw.ImageDraw,
    x: int,
    height: int,
    thickness: int,
    curve_left: float,
    curve_right: float,
    lean: float,
) -> None:
    y_top = 320 - height // 2
    y_bottom = 320 + height // 2
    mid_l = x - thickness // 2
    mid_r = x + thickness // 2
    left, _ = _lens_profile(mid_l, y_top, y_bottom, curve_left, lean)
    _, right = _lens_profile(mid_r, y_top, y_bottom, curve_right, lean)
    draw.line(left, fill=(0, 0, 0), width=2)
    draw.line(right, fill=(0, 0, 0), width=2)
    draw.line([left[0], right[0]], fill=(0, 0, 0), width=2)
    draw.line([left[-1], right[-1]], fill=(0, 0, 0), width=2)


def _ray_curve(start: Tuple[float, float], target: Tuple[float, float], progress: float, fan: float, field: float) -> List[Tuple[int, int]]:
    xs = [142, 178, 218, 268, 318, 368, 418, 468, 515]
    pts: List[Tuple[int, int]] = []
    sy = start[1]
    ty = target[1]
    for k, x in enumerate(xs):
        t = k / (len(xs) - 1)
        y = sy * (1.0 - t) + ty * t
        y += fan * progress * (54.0 * (t ** 1.35) - 18.0 * t * (1.0 - t))
        y += field * fan * 9.0 * t * (1.0 - t)
        # Small local bends at approximate surfaces make the path feel
        # refracted instead of a simple bezier sketch.
        y += progress * 6.0 * fan * (
            0.7 * max(0.0, 1.0 - abs(t - 0.28) / 0.08)
            - 0.45 * max(0.0, 1.0 - abs(t - 0.58) / 0.10)
        )
        pts.append((int(round(x)), int(round(y))))
    return pts


def _autolens_style_frame(label: str, summary: Dict[str, float], progress: float) -> Image.Image:
    img = Image.new("RGB", (640, 640), "white")
    draw = ImageDraw.Draw(img)
    _draw_dotted_background(draw, 640)

    font = _font(17)
    rms = float(summary.get("rms", 1.0))
    field = float(summary.get("field", 0.0))
    chroma = float(summary.get("chromatic", 0.0))
    shift = float(summary.get("shift", 0.0))
    fnum = max(1.85, min(4.0, 4.10 - 1.35 * progress - 0.10 * rms))
    foc = 5.55 + 0.08 * progress + 0.03 * shift
    fov = 83.7 - 1.0 * progress + 0.15 * field
    title = f"FoV{fov:04.1f}(24mm EFL)_F/{fnum:.2f}_DIAG10.0mm_FocLen{foc:.2f}mm"
    draw.text((70, 52), title, fill=(0, 0, 0), font=font)

    sensor_x = 515
    ray_specs = [
        ((235, 0, 0), 112 - 18 * progress - 8 * chroma, -0.92),
        ((0, 120, 0), 225 + 4 * field, 0.0),
        ((0, 70, 235), 348 + 12 * progress + 5 * chroma, 0.82),
    ]
    for color, target_y, fan in ray_specs:
        for j in range(11):
            start_y = 287 + (j - 5) * 10.5
            pts = _ray_curve((142, start_y), (sensor_x, target_y), progress, fan, field)
            draw.line(pts, fill=color, width=2)

    aperture_x = 145
    draw.line([(aperture_x, 250), (aperture_x, 388)], fill=(230, 145, 0), width=2)
    draw.line([(aperture_x - 4, 250), (aperture_x + 5, 250)], fill=(230, 145, 0), width=2)
    draw.line([(aperture_x - 4, 388), (aperture_x + 5, 388)], fill=(230, 145, 0), width=2)

    base = [
        (172, 260, 16, 5, 8, -2.0),
        (212, 300, 25, 12, 7, -1.0),
        (262, 235, 20, -3, 4, 0.5),
        (318, 315, 30, 10, 12, 1.0),
        (372, 250, 22, -5, 9, 1.5),
        (418, 330, 34, 15, 19, 2.0),
        (455, 390, 58, 22, 25, 1.5),
    ]
    for i, (x, height, thick, c1, c2, lean0) in enumerate(base):
        early_height = 388 if i < 6 else 330
        h = int(early_height * (1.0 - progress) + height * progress)
        thickness = int(18 * (1.0 - progress) + thick * progress)
        left_curve = c1 * progress + 0.8 * rms
        right_curve = c2 * progress + 0.8 * rms
        lean = lean0 * progress + shift * 3.0
        _draw_lens_element(draw, x, h, thickness, left_curve, right_curve, lean)

    draw.line([(sensor_x, 100), (sensor_x, 505)], fill=(0, 0, 0), width=4)
    draw.line([(sensor_x - 2, 100), (sensor_x + 13, 88)], fill=(0, 0, 0), width=2)
    draw.line([(sensor_x - 2, 505), (sensor_x + 13, 518)], fill=(0, 0, 0), width=2)

    return img


def _make_autolens_style_gif(preview_dir: Path, out_dir: Path, imgs_dir: Path | None, duration_ms: int) -> Path | None:
    metrics = _scan_csv(preview_dir, "psf_metrics")
    if not metrics:
        return None
    # Avoid duplicating an initial and best frame if only one real snapshot exists.
    real_orders = [m for m in metrics if 0 <= m[0] < 10**9]
    series = real_orders if real_orders else metrics
    if metrics[0][0] == -1 and (not series or series[0][0] != -1):
        series = [metrics[0]] + series
    if metrics[-1][0] == 10**9 and (not series or series[-1][0] != 10**9):
        series = series + [metrics[-1]]

    summaries = [_read_psf_summary(path) for _order, _label, path in series]
    if len(summaries) == 1:
        summaries = summaries * 2
        series = series * 2
    frames = []
    denom = max(len(summaries) - 1, 1)
    for idx, ((_order, label, _path), summary) in enumerate(zip(series, summaries)):
        progress = idx / denom
        # Blend physical-looking progress with actual metric change, so the
        # animation remains readable even when PSF updates are numerically small.
        metric_progress = max(0.0, min(1.0, (summary.get("rms", 1.0) - 0.8) / 1.2))
        frames.append(_autolens_style_frame(label, summary, 0.65 * progress + 0.35 * metric_progress))

    out = out_dir / "autolens_style_optics_learning.gif"
    _save_gif(frames, out, duration_ms)
    if imgs_dir is not None:
        imgs_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, imgs_dir / "raw2task_autolens_style_optics_learning.gif")
    return out


def _load_png(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def _paired_by_label(groups: Sequence[List[Tuple[int, str, Path]]]) -> List[Tuple[str, List[Path]]]:
    by_order = {}
    for idx, group in enumerate(groups):
        for order, label, path in group:
            by_order.setdefault(order, {"label": label, "paths": [None] * len(groups)})
            by_order[order]["paths"][idx] = path
    paired = []
    for order in sorted(by_order):
        paths = by_order[order]["paths"]
        if any(p is not None for p in paths):
            paired.append((by_order[order]["label"], [p for p in paths if p is not None]))
    return paired


def _prefer_long_series(primary: List[Tuple[int, str, Path]], fallback: List[Tuple[int, str, Path]]) -> List[Tuple[int, str, Path]]:
    return primary if len(primary) >= max(2, len(fallback)) else fallback


def make_learning_gifs(preview_dir: Path, out_dir: Path, imgs_dir: Path | None = None, duration_ms: int = 420) -> List[Path]:
    outputs: List[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    autolens_style = _make_autolens_style_gif(preview_dir, out_dir, imgs_dir, duration_ms)
    if autolens_style is not None:
        outputs.append(autolens_style)

    psf = _scan(preview_dir, "psf_kernels")
    psf_rms = _scan(preview_dir, "psf_rms_by_field")
    cfa_heat = _scan(preview_dir, "cfa_weights")
    cfa_bars = _scan(preview_dir, "cfa_bars")
    noise = _scan(preview_dir, "noise_adc_curve")
    cfa_series = _prefer_long_series(cfa_bars, cfa_heat)

    if psf:
        frames = [
            _title_frame(
                [("field-dependent PSF kernels", _load_png(path))],
                label,
                "Learned Optics Evolution",
                "PSF shape, color response, and field variation evolve during task-driven training.",
                width=980,
                panel_height=850,
            )
            for _order, label, path in psf
        ]
        out = out_dir / "learned_optics_evolution.gif"
        _save_gif(frames, out, duration_ms)
        outputs.append(out)

    sensor_pairs = _paired_by_label([cfa_series, noise])
    if sensor_pairs:
        frames = []
        for label, paths in sensor_pairs:
            panel_titles = []
            for path in paths:
                if path.name.endswith("noise_adc_curve.png"):
                    panel_titles.append("noise and ADC response")
                elif path.name.endswith("cfa_bars.png"):
                    panel_titles.append("CFA spectral response")
                else:
                    panel_titles.append("CFA 2x2 weights")
            frames.append(
                _title_frame(
                    list(zip(panel_titles, [_load_png(p) for p in paths])),
                    label,
                    "Learned Sensor Evolution",
                    "CFA, exposure/noise, and quantization parameters are tracked as measurable sensor settings.",
                    width=1500,
                    panel_height=640,
                )
            )
        out = out_dir / "learned_sensor_evolution.gif"
        _save_gif(frames, out, duration_ms)
        outputs.append(out)

    combined = _paired_by_label([psf, psf_rms, cfa_series, noise])
    if combined:
        frames = []
        for label, paths in combined:
            titles = []
            for path in paths:
                if path.name.endswith("psf_kernels.png"):
                    titles.append("learned field PSFs")
                elif path.name.endswith("psf_rms_by_field.png"):
                    titles.append("PSF width by field")
                elif path.name.endswith("noise_adc_curve.png"):
                    titles.append("noise and ADC response")
                elif path.name.endswith("cfa_bars.png"):
                    titles.append("CFA spectral response")
                else:
                    titles.append("CFA weights")
            frames.append(
                _title_frame(
                    list(zip(titles, [_load_png(p) for p in paths])),
                    label,
                    "Optics-Sensor Co-Design Learning",
                    "Training snapshots expose the learned camera, not only the final segmentation score.",
                    width=1800,
                    panel_height=620,
                )
            )
        out = out_dir / "optics_sensor_codesign_evolution.gif"
        _save_gif(frames, out, duration_ms)
        outputs.append(out)

    if imgs_dir is not None:
        imgs_dir.mkdir(parents=True, exist_ok=True)
        for out in outputs:
            if out.name == "autolens_style_optics_learning.gif":
                continue
            shutil.copy2(out, imgs_dir / f"raw2task_{out.name}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", required=True, help="Directory containing initial/epoch/best design preview PNGs.")
    parser.add_argument("--out", default="", help="Output directory. Defaults to <preview-dir>/learning_gifs.")
    parser.add_argument("--imgs-dir", default="imgs", help="Optional repo image directory to receive copied GIFs.")
    parser.add_argument("--duration-ms", type=int, default=420)
    args = parser.parse_args()

    preview_dir = Path(os.path.abspath(os.path.expanduser(args.preview_dir)))
    out_dir = Path(os.path.abspath(os.path.expanduser(args.out))) if args.out else preview_dir / "learning_gifs"
    imgs_dir = Path(os.path.abspath(os.path.expanduser(args.imgs_dir))) if args.imgs_dir else None
    outputs = make_learning_gifs(preview_dir, out_dir, imgs_dir=imgs_dir, duration_ms=args.duration_ms)
    if not outputs:
        raise SystemExit(f"No learning snapshots found in {preview_dir}")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
