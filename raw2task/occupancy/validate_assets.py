"""Validate real occupancy dataset assets before training."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict

import numpy as np

from raw2task.occupancy.dataset import DEFAULT_MASK_KEYS, DEFAULT_OCC_KEYS, _load_array
from raw2task.occupancy.schema import load_manifest, missing_files


def _shape(path: str, keys) -> tuple[int, ...]:
    return tuple(_load_array(path, keys).shape)


def validate(root: str, manifest: str, max_samples: int = 200) -> Dict[str, Any]:
    records = load_manifest(manifest, root=root)
    missing = missing_files(records)
    report: Dict[str, Any] = {
        "root": os.path.abspath(os.path.expanduser(root)),
        "manifest": os.path.abspath(os.path.expanduser(manifest)),
        "num_records": len(records),
        "missing_count": len(missing),
        "missing_preview": missing[:20],
        "camera_count_hist": dict(Counter(len(r.cameras) for r in records)),
        "occupancy_shapes": {},
        "mask_shapes": {},
        "calibration_missing_samples": [],
    }
    if missing:
        return report

    occ_shapes: Counter = Counter()
    mask_shapes: Counter = Counter()
    calibration_missing = []
    for rec in records[:max_samples]:
        occ_shapes[str(_shape(rec.occupancy, DEFAULT_OCC_KEYS))] += 1
        mask_path = rec.camera_mask or rec.lidar_mask
        if mask_path:
            mask_shapes[str(_shape(mask_path, DEFAULT_MASK_KEYS))] += 1
        if any(cam.intrinsics is None or cam.extrinsics is None for cam in rec.cameras):
            calibration_missing.append(rec.sample_id)

    report["occupancy_shapes"] = dict(occ_shapes)
    report["mask_shapes"] = dict(mask_shapes)
    report["calibration_missing_samples"] = calibration_missing[:20]
    report["status"] = "ok" if not missing and not calibration_missing else "needs_attention"
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--max-samples", type=int, default=200)
    args = parser.parse_args()

    report = validate(args.root, args.manifest, max_samples=args.max_samples)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text + "\n")
    if report.get("missing_count", 0) or report.get("calibration_missing_samples"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

