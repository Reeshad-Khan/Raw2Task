"""Build KITTI-360 observed semantic occupancy labels.

This preprocessing stage consumes the official KITTI-360 3D semantic PLY
windows and produces per-frame voxel labels in the camera-0 coordinate frame.
The output is real 3D supervision from KITTI-360 annotations, but it is observed
semantic occupancy, not semantic scene completion of unobserved space.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import re
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from deeplens.projects.raw2task.data.kitti360_seg import CITYSCAPES_TRAINID


PLY_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("semantic", "<i4"),
        ("instance", "<i4"),
        ("visible", "u1"),
        ("confidence", "<f4"),
    ]
)


@dataclass(frozen=True)
class Segment:
    seq: str
    start: int
    end: int
    static_ply: Path
    dynamic_ply: Path


def parse_perspective(path: Path) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, rest = line.split(":", 1)
            if key.startswith("P_rect_"):
                vals = [float(x) for x in rest.strip().split()]
                if len(vals) != 12:
                    continue
                out[key[-2:]] = np.asarray(vals, dtype=np.float32).reshape(3, 4)
    if "00" not in out:
        raise RuntimeError(f"Missing P_rect_00 in {path}")
    return out


def load_cam0_to_world(path: Path) -> Dict[int, np.ndarray]:
    poses: Dict[int, np.ndarray] = {}
    with open(path, "r") as f:
        for line in f:
            vals = line.strip().split()
            if not vals:
                continue
            frame = int(vals[0])
            nums = [float(x) for x in vals[1:]]
            if len(nums) != 16:
                continue
            poses[frame] = np.asarray(nums, dtype=np.float64).reshape(4, 4)
    return poses


def nearest_pose(poses: Dict[int, np.ndarray], frame: int) -> np.ndarray:
    if frame in poses:
        return poses[frame]
    keys = sorted(poses)
    if not keys:
        raise RuntimeError("No poses available")
    idx = bisect.bisect_left(keys, frame)
    candidates = []
    if idx < len(keys):
        candidates.append(keys[idx])
    if idx > 0:
        candidates.append(keys[idx - 1])
    best = min(candidates, key=lambda k: abs(k - frame))
    return poses[best]


def discover_segments(root: Path) -> List[Segment]:
    base = root / "data_3d_semantics" / "train"
    segments: List[Segment] = []
    for static_ply in sorted(base.glob("2013_05_28_drive_*_sync/static/*.ply")):
        m = re.match(r"(\d{10})_(\d{10})\.ply$", static_ply.name)
        if not m:
            continue
        dynamic_ply = static_ply.parents[1] / "dynamic" / static_ply.name
        if not dynamic_ply.is_file():
            continue
        segments.append(
            Segment(
                seq=static_ply.parents[1].name,
                start=int(m.group(1)),
                end=int(m.group(2)),
                static_ply=static_ply,
                dynamic_ply=dynamic_ply,
            )
        )
    return segments


def index_segments(segments: Iterable[Segment]) -> Dict[str, List[Segment]]:
    out: Dict[str, List[Segment]] = {}
    for seg in segments:
        out.setdefault(seg.seq, []).append(seg)
    for seq in out:
        out[seq].sort(key=lambda s: (s.start, s.end))
    return out


def segment_for_frame(index: Dict[str, List[Segment]], seq: str, frame: int) -> Segment | None:
    for seg in index.get(seq, []):
        if seg.start <= frame <= seg.end:
            return seg
    return None


def read_ply_vertices(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"Unexpected EOF in PLY header: {path}")
            header.append(line.decode("latin1").strip())
            if header[-1] == "end_header":
                break
        fmt = next((x for x in header if x.startswith("format ")), "")
        if "binary_little_endian" not in fmt:
            raise RuntimeError(f"Unsupported PLY format in {path}: {fmt}")
        vertex_line = next((x for x in header if x.startswith("element vertex ")), "")
        n = int(vertex_line.split()[-1])
        data = np.frombuffer(f.read(n * PLY_DTYPE.itemsize), dtype=PLY_DTYPE, count=n)
    return data


@lru_cache(maxsize=8)
def load_segment_points(static_path: str, dynamic_path: str) -> tuple[np.ndarray, np.ndarray]:
    arrays = [read_ply_vertices(Path(static_path)), read_ply_vertices(Path(dynamic_path))]
    pts = []
    sem = []
    for arr in arrays:
        if arr.size == 0:
            continue
        visible = arr["visible"].astype(bool)
        xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
        semantic = arr["semantic"].astype(np.int32)
        pts.append(xyz[visible])
        sem.append(semantic[visible])
    if not pts:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.int32)
    return np.concatenate(pts, axis=0), np.concatenate(sem, axis=0)


def remap_semantic_to_trainid(semantic: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    out = np.full(semantic.shape, ignore_index, dtype=np.uint8)
    for original_id, train_id in CITYSCAPES_TRAINID.items():
        out[semantic == original_id] = train_id
    return out


def voxelize_points(
    points_cam: np.ndarray,
    labels: np.ndarray,
    grid_shape: Tuple[int, int, int],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Tuple[float, float],
    ignore_index: int = 255,
) -> tuple[np.ndarray, np.ndarray]:
    # Camera-0 convention: x right, y down, z forward. Output axes are D,H,W:
    # z bins, y bins, x bins.
    d, h, w = grid_shape
    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    keep = (
        (x >= x_range[0]) & (x < x_range[1])
        & (y >= y_range[0]) & (y < y_range[1])
        & (z >= z_range[0]) & (z < z_range[1])
        & (labels != ignore_index)
    )
    occ = np.full((d, h, w), ignore_index, dtype=np.uint8)
    valid = np.zeros((d, h, w), dtype=bool)
    if not keep.any():
        return occ, valid

    x = x[keep]
    y = y[keep]
    z = z[keep]
    lab = labels[keep].astype(np.int64)
    ix = ((x - x_range[0]) / (x_range[1] - x_range[0]) * w).astype(np.int64)
    iy = ((y - y_range[0]) / (y_range[1] - y_range[0]) * h).astype(np.int64)
    iz = ((z - z_range[0]) / (z_range[1] - z_range[0]) * d).astype(np.int64)
    flat = iz * h * w + iy * w + ix

    # Majority vote per voxel. Number of trainIds is 19, so bincount per occupied
    # voxel is compact and deterministic.
    order = np.argsort(flat)
    flat = flat[order]
    lab = lab[order]
    starts = np.r_[0, np.flatnonzero(flat[1:] != flat[:-1]) + 1]
    ends = np.r_[starts[1:], flat.size]
    for s, e in zip(starts, ends):
        counts = np.bincount(lab[s:e], minlength=19)
        chosen = int(counts.argmax())
        f = int(flat[s])
        iz0 = f // (h * w)
        rem = f % (h * w)
        iy0 = rem // w
        ix0 = rem % w
        occ[iz0, iy0, ix0] = chosen
        valid[iz0, iy0, ix0] = True
    return occ, valid


def parse_split_line(root: Path, line: str) -> tuple[str, int, Path]:
    image_rel = line.strip().split()[0]
    parts = Path(image_rel).parts
    seq = next(p for p in parts if p.startswith("2013_05_28_drive_"))
    frame = int(Path(image_rel).stem)
    return seq, frame, root / image_rel


def write_manifest_and_voxels(args: argparse.Namespace, split_name: str, split_file: Path, segments_idx, p_rect) -> int:
    root = Path(args.root)
    out_root = Path(args.out_root)
    vox_dir = out_root / "voxels" / split_name
    manifest_path = out_root / "manifests" / f"{split_name}.jsonl"
    vox_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    pose_cache: Dict[str, Dict[int, np.ndarray]] = {}
    written = 0
    skipped = 0
    with open(split_file, "r") as f, open(manifest_path, "w") as mf:
        for line_idx, line in enumerate(f):
            if args.max_samples > 0 and written >= args.max_samples:
                break
            if line_idx % args.stride != 0:
                continue
            if not line.strip():
                continue
            seq, frame, image_path = parse_split_line(root, line)
            seg = segment_for_frame(segments_idx, seq, frame)
            if seg is None or not image_path.is_file():
                skipped += 1
                continue
            if seq not in pose_cache:
                pose_cache[seq] = load_cam0_to_world(root / "data_poses" / seq / "cam0_to_world.txt")
            cam0_to_world = nearest_pose(pose_cache[seq], frame)
            world_to_cam0 = np.linalg.inv(cam0_to_world)
            pts_world, sem_orig = load_segment_points(str(seg.static_ply), str(seg.dynamic_ply))
            if pts_world.size == 0:
                skipped += 1
                continue
            pts_h = np.concatenate([pts_world, np.ones((pts_world.shape[0], 1), dtype=np.float64)], axis=1)
            pts_cam = (world_to_cam0 @ pts_h.T).T[:, :3]
            labels = remap_semantic_to_trainid(sem_orig, ignore_index=args.ignore_index)
            occ, valid = voxelize_points(
                pts_cam,
                labels,
                grid_shape=tuple(args.grid_shape),
                x_range=tuple(args.x_range),
                y_range=tuple(args.y_range),
                z_range=tuple(args.z_range),
                ignore_index=args.ignore_index,
            )
            if valid.sum() < args.min_valid_voxels:
                skipped += 1
                continue
            rel_npz = Path("voxels") / split_name / seq / f"{frame:010d}.npz"
            out_npz = out_root / rel_npz
            out_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_npz,
                semantics=occ,
                valid_mask=valid,
                mask_camera=valid,
                grid_shape=np.asarray(args.grid_shape, dtype=np.int32),
                x_range=np.asarray(args.x_range, dtype=np.float32),
                y_range=np.asarray(args.y_range, dtype=np.float32),
                z_range=np.asarray(args.z_range, dtype=np.float32),
                source_segment=np.asarray([seg.start, seg.end], dtype=np.int32),
            )
            rec = {
                "sample_id": f"{seq}_{frame:010d}",
                "cameras": {
                    "image_00": {
                        "image": os.path.relpath(image_path, out_root),
                        "intrinsics": p_rect["00"][:, :3].tolist(),
                        "extrinsics": np.eye(4, dtype=np.float32).tolist(),
                    }
                },
                "occupancy": str(rel_npz),
                "camera_mask": str(rel_npz),
                "meta": {
                    "dataset": "kitti360",
                    "task": "observed_semantic_occupancy",
                    "sequence": seq,
                    "frame": frame,
                    "coordinate_frame": "camera_00_rectified_proxy",
                    "note": "Voxelized from official KITTI-360 3D semantic PLY annotations; not hidden-space semantic scene completion.",
                },
            }
            mf.write(json.dumps(rec) + "\n")
            written += 1
    print(f"[{split_name}] wrote {written} samples to {manifest_path}; skipped={skipped}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/rk010/Desktop/Research/NurIPS/KITTI-360")
    parser.add_argument("--out-root", default="data_external/kitti360_occupancy")
    parser.add_argument("--train-split", default="data_2d_semantics/train/2013_05_28_drive_train_frames.txt")
    parser.add_argument("--val-split", default="data_2d_semantics/train/2013_05_28_drive_val_frames.txt")
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--min-valid-voxels", type=int, default=20)
    parser.add_argument("--grid-shape", type=int, nargs=3, default=[16, 64, 128], metavar=("D", "H", "W"))
    parser.add_argument("--x-range", type=float, nargs=2, default=[-20.0, 20.0])
    parser.add_argument("--y-range", type=float, nargs=2, default=[-3.0, 5.0])
    parser.add_argument("--z-range", type=float, nargs=2, default=[0.0, 50.0])
    parser.add_argument("--ignore-index", type=int, default=255)
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    out_root = Path(args.out_root).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    segments = discover_segments(root)
    if not segments:
        raise RuntimeError(f"No KITTI-360 3D semantic PLY segments found under {root}")
    p_rect = parse_perspective(root / "calibration" / "perspective.txt")
    segments_idx = index_segments(segments)

    metadata = {
        "source_root": str(root),
        "task": "observed_semantic_occupancy",
        "grid_shape": args.grid_shape,
        "x_range": args.x_range,
        "y_range": args.y_range,
        "z_range": args.z_range,
        "num_segments": len(segments),
        "class_space": "Cityscapes/KITTI trainId 19 occupied classes, 255 ignored/unobserved",
    }
    with open(out_root / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    train_n = write_manifest_and_voxels(args, "train", root / args.train_split, segments_idx, p_rect)
    val_n = write_manifest_and_voxels(args, "val", root / args.val_split, segments_idx, p_rect)
    if train_n == 0 or val_n == 0:
        raise RuntimeError("Generated empty train or val manifest; check split/segment overlap.")


if __name__ == "__main__":
    main()
