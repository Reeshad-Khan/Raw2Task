#!/usr/bin/env python3
"""Summarize hybrid camera validation/control matrices.

The hybrid search produces several follow-up runs per dataset: learned
co-design, the same physical camera held fixed, and an RGB control. This
utility reads partially-complete or complete run directories and reports the
decision-relevant deltas without requiring artifact generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


METRICS_LOG = "metrics_log.csv"


@dataclass
class RunSummary:
    dataset: str
    experiment: str
    role: str
    candidate: str
    model: str
    run_dir: Path
    status: str
    epoch: int
    target_epochs: int
    best_miou: float
    best_epoch: int
    latest_miou: float
    latest_acc: float
    avg_bestk_miou: float
    avg_bestk_acc: float
    train_probe_miou: float
    trend_last3: float
    plateau: bool


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _as_float(value: Any, default: float = math.nan) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _dataset_from_matrix(path: Path, matrix: Dict[str, Any]) -> str:
    out_root = str(matrix.get("out_root", ""))
    text = " ".join([path.as_posix(), out_root]).lower()
    if "city" in text:
        return "cityscapes"
    if "kitti" in text:
        return "kitti360"
    return path.stem.replace("_validation_matrix", "")


def _role_and_candidate(name: str) -> Tuple[str, str]:
    role = "codesign"
    if "_fixed_" in name or name.endswith("_fixed"):
        role = "fixed"
    elif "_rgb_" in name or name.endswith("_rgb"):
        role = "rgb"
    match = re.search(r"cand(\d+)", name)
    candidate = match.group(1) if match else "rgb"
    return role, candidate


def _experiment_model(exp: Dict[str, Any]) -> str:
    return str(exp.get("overrides", {}).get("model", {}).get("name", "unknown"))


def _target_epochs(exp: Dict[str, Any]) -> int:
    return _as_int(exp.get("overrides", {}).get("train", {}).get("epochs"), 0)


def _metrics_rows(run_dir: Path) -> List[Dict[str, str]]:
    path = run_dir / METRICS_LOG
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _avg_bestk_metrics(run_dir: Path) -> Tuple[float, float]:
    best_miou = math.nan
    best_acc = math.nan
    for path in sorted(run_dir.glob("metrics_avg_*k*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        miou = _as_float(data.get("mIoU"))
        if math.isnan(miou):
            continue
        if math.isnan(best_miou) or miou > best_miou:
            best_miou = miou
            best_acc = _as_float(data.get("pixel_acc"))
    return best_miou, best_acc


def _run_completed(run_dir: Path) -> bool:
    log_path = run_dir / "train.log"
    if not log_path.exists():
        return False
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "TRAINING COMPLETE" in text or "[EarlyStop]" in text


def _trend(values: Sequence[float], n: int = 3) -> float:
    finite = [v for v in values if not math.isnan(v)]
    if len(finite) < 2:
        return math.nan
    window = finite[-n:]
    if len(window) < 2:
        return math.nan
    return window[-1] - window[0]


def _summarize_experiment(dataset: str, out_root: Path, exp: Dict[str, Any], seed: int) -> RunSummary:
    name = str(exp["name"])
    role, candidate = _role_and_candidate(name)
    run_dir = out_root / f"{name}_seed{seed}"
    rows = _metrics_rows(run_dir)
    target_epochs = _target_epochs(exp)
    model = _experiment_model(exp)

    if not rows:
        return RunSummary(
            dataset=dataset,
            experiment=name,
            role=role,
            candidate=candidate,
            model=model,
            run_dir=run_dir,
            status="waiting",
            epoch=0,
            target_epochs=target_epochs,
            best_miou=math.nan,
            best_epoch=0,
            latest_miou=math.nan,
            latest_acc=math.nan,
            avg_bestk_miou=math.nan,
            avg_bestk_acc=math.nan,
            train_probe_miou=math.nan,
            trend_last3=math.nan,
            plateau=False,
        )

    miou_values = [_as_float(row.get("mIoU")) for row in rows]
    acc_values = [_as_float(row.get("pixel_acc")) for row in rows]
    epochs = [_as_int(row.get("epoch")) for row in rows]
    probe_values = [_as_float(row.get("train_deploy_probe_mIoU")) for row in rows]
    best_idx = max(range(len(rows)), key=lambda idx: (-math.inf if math.isnan(miou_values[idx]) else miou_values[idx]))
    latest_epoch = epochs[-1] if epochs else len(rows)
    trend_last3 = _trend(miou_values, n=3)
    avg_miou, avg_acc = _avg_bestk_metrics(run_dir)
    status = "done" if (target_epochs > 0 and latest_epoch >= target_epochs) or _run_completed(run_dir) else "running"
    plateau = (not math.isnan(trend_last3)) and abs(trend_last3) < 0.003 and latest_epoch >= 4

    return RunSummary(
        dataset=dataset,
        experiment=name,
        role=role,
        candidate=candidate,
        model=model,
        run_dir=run_dir,
        status=status,
        epoch=latest_epoch,
        target_epochs=target_epochs,
        best_miou=miou_values[best_idx],
        best_epoch=epochs[best_idx],
        latest_miou=miou_values[-1],
        latest_acc=acc_values[-1] if acc_values else math.nan,
        avg_bestk_miou=avg_miou,
        avg_bestk_acc=avg_acc,
        train_probe_miou=probe_values[-1] if probe_values else math.nan,
        trend_last3=trend_last3,
        plateau=plateau,
    )


def _read_matrix(path: Path) -> List[RunSummary]:
    matrix = _load_yaml(path)
    dataset = _dataset_from_matrix(path, matrix)
    out_root = Path(matrix["out_root"])
    seeds = matrix.get("seeds") or [0]
    summaries: List[RunSummary] = []
    for exp in matrix.get("experiments", []):
        for seed in seeds:
            summaries.append(_summarize_experiment(dataset, out_root, exp, int(seed)))
    return summaries


def _fmt(value: float, digits: int = 4) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _print_table(rows: Sequence[RunSummary]) -> None:
    headers = [
        "dataset",
        "role",
        "cand",
        "model",
        "status",
        "epoch",
        "best",
        "best_ep",
        "latest",
        "avg_bestk",
        "probe",
        "trend3",
    ]
    data = []
    for row in rows:
        data.append(
            [
                row.dataset,
                row.role,
                row.candidate,
                row.model,
                row.status,
                f"{row.epoch}/{row.target_epochs}",
                _fmt(row.best_miou),
                str(row.best_epoch),
                _fmt(row.latest_miou),
                _fmt(row.avg_bestk_miou),
                _fmt(row.train_probe_miou),
                _fmt(row.trend_last3),
            ]
        )
    widths = [len(h) for h in headers]
    for record in data:
        widths = [max(width, len(cell)) for width, cell in zip(widths, record)]
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for record in data:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(record)))


def _best_by_key(rows: Iterable[RunSummary], role: str, candidate: str) -> Optional[RunSummary]:
    matches = [row for row in rows if row.role == role and row.candidate == candidate and not math.isnan(row.best_miou)]
    if not matches:
        return None
    return max(matches, key=lambda row: row.best_miou)


def _print_decisions(rows: Sequence[RunSummary]) -> None:
    print("\nDecision deltas")
    by_dataset: Dict[str, List[RunSummary]] = {}
    for row in rows:
        by_dataset.setdefault(row.dataset, []).append(row)
    for dataset, dataset_rows in sorted(by_dataset.items()):
        rgb = _best_by_key(dataset_rows, "rgb", "rgb")
        candidates = sorted({row.candidate for row in dataset_rows if row.candidate != "rgb"})
        if not candidates:
            continue
        print(f"\n{dataset}")
        for cand in candidates:
            codesign = _best_by_key(dataset_rows, "codesign", cand)
            fixed = _best_by_key(dataset_rows, "fixed", cand)
            if codesign is None:
                continue
            fixed_delta = math.nan if fixed is None else codesign.best_miou - fixed.best_miou
            rgb_delta = math.nan if rgb is None else codesign.best_miou - rgb.best_miou
            fixed_text = "pending" if fixed is None else f"{_fmt(fixed_delta, 4)} vs fixed"
            rgb_text = "pending" if rgb is None else f"{_fmt(rgb_delta, 4)} vs RGB"
            note = "wait"
            if fixed is not None and rgb is not None:
                if fixed_delta > 0.01 and rgb_delta > -0.02:
                    note = "promising"
                elif fixed_delta <= 0.003:
                    note = "not enough co-design delta"
                else:
                    note = "mixed"
            print(
                f"  cand{cand}: codesign best {_fmt(codesign.best_miou)} "
                f"({codesign.status}, ep {codesign.best_epoch}); {fixed_text}; {rgb_text}; {note}"
            )


def _write_csv(rows: Sequence[RunSummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(RunSummary.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = dict(row.__dict__)
            record["run_dir"] = str(record["run_dir"])
            writer.writerow(record)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", action="append", required=True, help="Validation/control matrix YAML. Can be repeated.")
    parser.add_argument("--csv", default="", help="Optional path for a machine-readable summary CSV.")
    parser.add_argument(
        "--watch",
        type=float,
        default=0.0,
        help="Refresh every N seconds. Metrics update after each eval epoch, not every train step.",
    )
    return parser.parse_args()


def _run_once(args: argparse.Namespace) -> None:
    rows: List[RunSummary] = []
    for matrix_path in args.matrix:
        rows.extend(_read_matrix(Path(matrix_path)))
    rows.sort(key=lambda row: (row.dataset, row.candidate, row.role))
    print(time.strftime("Validation Control Summary | %Y-%m-%d %H:%M:%S"))
    _print_table(rows)
    _print_decisions(rows)
    if args.csv:
        _write_csv(rows, Path(args.csv))
        print(f"\nWrote {args.csv}")


def main() -> None:
    args = parse_args()
    if args.watch and args.watch > 0:
        while True:
            if os.isatty(1):
                print("\033[2J\033[H", end="")
            _run_once(args)
            print(f"\nRefreshing every {args.watch:g}s. Values change when an eval epoch writes metrics_log.csv.")
            time.sleep(args.watch)
    else:
        _run_once(args)


if __name__ == "__main__":
    main()
