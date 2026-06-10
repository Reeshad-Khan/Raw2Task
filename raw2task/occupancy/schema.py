"""Manifest schema for real 3D occupancy experiments.

The loader intentionally uses a dataset-neutral JSONL manifest. This keeps the
training code independent of whether samples come from Occ3D-nuScenes,
SSCBench-KITTI-360, OpenOccupancy, or a future internal calibrated dataset.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence


REQUIRED_CAMERA_KEYS = ("image",)
REQUIRED_RECORD_KEYS = ("sample_id", "cameras", "occupancy")


@dataclass(frozen=True)
class CameraRecord:
    name: str
    image: str
    intrinsics: List[List[float]] | None = None
    extrinsics: List[List[float]] | None = None


@dataclass(frozen=True)
class OccupancyRecord:
    sample_id: str
    cameras: List[CameraRecord]
    occupancy: str
    lidar_mask: str | None = None
    camera_mask: str | None = None
    meta: Dict[str, Any] | None = None


def resolve_path(root: str, path: str | None) -> str | None:
    if not path:
        return None
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.expanduser(root), path)


def _matrix_or_none(value: Any, name: str) -> List[List[float]] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a numeric matrix, got {type(value).__name__}")
    out: List[List[float]] = []
    for row in value:
        if not isinstance(row, Sequence):
            raise ValueError(f"{name} rows must be sequences")
        out.append([float(x) for x in row])
    return out


def parse_record(row: Mapping[str, Any], root: str, line_no: int = 0) -> OccupancyRecord:
    missing = [k for k in REQUIRED_RECORD_KEYS if k not in row]
    if missing:
        raise ValueError(f"manifest line {line_no}: missing required keys {missing}")

    cameras_raw = row["cameras"]
    if isinstance(cameras_raw, Mapping):
        cameras_iter = [{"name": name, **dict(cam)} for name, cam in cameras_raw.items()]
    elif isinstance(cameras_raw, Sequence):
        cameras_iter = list(cameras_raw)
    else:
        raise ValueError(f"manifest line {line_no}: cameras must be a list or dict")

    cameras: List[CameraRecord] = []
    for idx, cam in enumerate(cameras_iter):
        if not isinstance(cam, Mapping):
            raise ValueError(f"manifest line {line_no}: camera {idx} must be an object")
        missing_cam = [k for k in REQUIRED_CAMERA_KEYS if k not in cam]
        if missing_cam:
            raise ValueError(f"manifest line {line_no}: camera {idx} missing {missing_cam}")
        name = str(cam.get("name", f"cam{idx}"))
        cameras.append(
            CameraRecord(
                name=name,
                image=resolve_path(root, str(cam["image"])) or "",
                intrinsics=_matrix_or_none(cam.get("intrinsics"), f"{name}.intrinsics"),
                extrinsics=_matrix_or_none(cam.get("extrinsics"), f"{name}.extrinsics"),
            )
        )

    if not cameras:
        raise ValueError(f"manifest line {line_no}: at least one camera is required")

    return OccupancyRecord(
        sample_id=str(row["sample_id"]),
        cameras=cameras,
        occupancy=resolve_path(root, str(row["occupancy"])) or "",
        lidar_mask=resolve_path(root, row.get("lidar_mask")),
        camera_mask=resolve_path(root, row.get("camera_mask")),
        meta=dict(row.get("meta", {})),
    )


def load_manifest(path: str, root: str) -> List[OccupancyRecord]:
    path = os.path.expanduser(path)
    records: List[OccupancyRecord] = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            records.append(parse_record(json.loads(line), root=root, line_no=line_no))
    if not records:
        raise RuntimeError(f"No occupancy samples found in manifest: {path}")
    return records


def missing_files(records: Iterable[OccupancyRecord]) -> List[str]:
    missing: List[str] = []
    for rec in records:
        if not os.path.isfile(rec.occupancy):
            missing.append(rec.occupancy)
        for optional in (rec.lidar_mask, rec.camera_mask):
            if optional and not os.path.isfile(optional):
                missing.append(optional)
        for cam in rec.cameras:
            if not os.path.isfile(cam.image):
                missing.append(cam.image)
    return missing

