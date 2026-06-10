"""Run segmentation matrices plus KITTI-360 observed occupancy experiments."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml

from raw2task.run_paper_experiments import _detect_gpus


OCC_CONFIGS = [
    "deeplens/projects/raw2task/configs/occupancy/kitti360_observed_codesign.yaml",
    "deeplens/projects/raw2task/configs/occupancy/kitti360_observed_fixed_camera.yaml",
    "deeplens/projects/raw2task/configs/occupancy/kitti360_observed_rgb.yaml",
]

SEG3D_CONFIGS = [
    "deeplens/projects/raw2task/configs/seg3d/kitti360_observed_3dseg_codesign.yaml",
    "deeplens/projects/raw2task/configs/seg3d/kitti360_observed_3dseg_fixed_camera.yaml",
    "deeplens/projects/raw2task/configs/seg3d/kitti360_observed_3dseg_rgb.yaml",
]


@dataclass
class OccJob:
    name: str
    config_path: str
    ckpt_dir: str
    log_path: str
    epochs: int


@dataclass
class RunningOccJob:
    job: OccJob
    gpu: str
    process: subprocess.Popen
    log_file: object
    start_time: float
    output_thread: threading.Thread


def _run(cmd: List[str], dry_run: bool = False) -> None:
    print("[cmd] " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def _read_yaml(path: str) -> Dict:
    with open(os.path.expanduser(path), "r") as f:
        return yaml.safe_load(f)


def _write_yaml(path: str, cfg: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _last_epoch(metrics_csv: str) -> int:
    if not os.path.isfile(metrics_csv):
        return 0
    with open(metrics_csv, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0
    try:
        return int(float(rows[-1].get("epoch", 0)))
    except Exception:
        return 0


def _best_occ_metrics(ckpt_dir: str) -> Dict[str, str]:
    metrics_csv = os.path.join(ckpt_dir, "metrics_log.csv")
    if not os.path.isfile(metrics_csv):
        return {}
    with open(metrics_csv, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    best = max(rows, key=lambda r: float(r.get("mIoU", -1) or -1))
    return dict(best)


def _best_occ_ckpt(ckpt_dir: str) -> str:
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "best_ep*_miou*.pt")))
    return paths[-1] if paths else ""


def _write_voxel_paper_artifacts(
    jobs: List[OccJob],
    summary_path: str,
    run_root: str,
    title: str,
    note: str,
    guidance: str,
) -> None:
    tables_dir = os.path.join(run_root, "paper_tables")
    fig_dir = os.path.join(tables_dir, "figures")
    qual_dir = os.path.join(tables_dir, "qualitative")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(qual_dir, exist_ok=True)

    rows: List[Dict[str, str]] = []
    for job in jobs:
        metrics = _best_occ_metrics(job.ckpt_dir)
        if not metrics:
            continue
        row = {
            "name": job.name,
            "ckpt_dir": job.ckpt_dir,
            "best_ckpt": _best_occ_ckpt(job.ckpt_dir),
            "epoch": metrics.get("epoch", ""),
            "train_loss": metrics.get("train_loss", ""),
            "train_feedback": metrics.get("train_feedback", ""),
            "voxel_acc": metrics.get("voxel_acc", ""),
            "mIoU": metrics.get("mIoU", ""),
            "occupied_IoU": metrics.get("occupied_IoU", ""),
        }
        rows.append(row)

        viz = sorted(glob.glob(os.path.join(job.ckpt_dir, "viz", "*occupancy_bev.png")))
        if viz:
            shutil.copy2(viz[-1], os.path.join(qual_dir, f"{job.name}_bev_panel.png"))
        latest_controller = os.path.join(job.ckpt_dir, "monitoring", "task_feedback_controller_latest.json")
        if os.path.isfile(latest_controller):
            shutil.copy2(latest_controller, os.path.join(tables_dir, f"{job.name}_task_feedback_controller.json"))

    if not rows:
        return

    summary_name = "summary"
    with open(os.path.join(tables_dir, f"{summary_name}.csv"), "w", newline="") as f:
        fields = ["name", "epoch", "train_loss", "train_feedback", "voxel_acc", "mIoU", "occupied_IoU", "ckpt_dir", "best_ckpt"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    def _fmt(x: str) -> str:
        try:
            return f"{float(x):.4f}"
        except Exception:
            return str(x)

    md = [
        f"# {title}",
        "",
        note,
        "",
        "| experiment | epoch | voxel acc | voxel mIoU | occupied IoU | train feedback |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['name']} | {row.get('epoch', '')} | {_fmt(row.get('voxel_acc', ''))} | "
            f"{_fmt(row.get('mIoU', ''))} | {_fmt(row.get('occupied_IoU', ''))} | {_fmt(row.get('train_feedback', ''))} |"
        )
    md.extend(
        [
            "",
            "Claim guidance:",
            guidance,
            "",
            f"Raw summary CSV: `{summary_path}`",
        ]
    )
    with open(os.path.join(tables_dir, f"{summary_name}.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        labels = [r["name"].replace("kitti360_observed_occupancy_", "") for r in rows]
        miou = [float(r.get("mIoU", 0.0) or 0.0) for r in rows]
        occ = [float(r.get("occupied_IoU", 0.0) or 0.0) for r in rows]
        x = np.arange(len(rows))
        width = 0.38
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        ax.bar(x - width / 2, miou, width, label="voxel mIoU", color="#4c78a8")
        ax.bar(x + width / 2, occ, width, label="occupied IoU", color="#f58518")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("score")
        ax.set_title(title + ": matched camera-design comparison")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "matched_comparison.png"), dpi=200)
        plt.close(fig)
    except Exception as exc:
        print(f"[voxel] plot artifact skipped: {exc}")


def _prepare_occupancy_assets(args: argparse.Namespace) -> None:
    out_root = Path(args.occupancy_out_root)
    train_manifest = out_root / "manifests" / "train.jsonl"
    val_manifest = out_root / "manifests" / "val.jsonl"
    if args.skip_occupancy_build and train_manifest.is_file() and val_manifest.is_file():
        print(f"[occupancy] Reusing existing manifests under {out_root}")
    else:
        build_cmd = [
            args.python,
            "-u",
            "-m",
            "raw2task.occupancy.kitti360_builder",
            "--root",
            args.kitti_root,
            "--out-root",
            args.occupancy_out_root,
            "--stride",
            str(args.occupancy_stride),
            "--grid-shape",
            *[str(x) for x in args.occupancy_grid_shape],
            "--min-valid-voxels",
            str(args.occupancy_min_valid_voxels),
        ]
        if args.occupancy_max_samples > 0:
            build_cmd.extend(["--max-samples", str(args.occupancy_max_samples)])
        _run(build_cmd, dry_run=args.dry_run)

    val_report = os.path.join("runs", "occupancy", "kitti360_asset_report_val.json")
    validate_cmd = [
        args.python,
        "-u",
        "-m",
        "raw2task.occupancy.validate_assets",
        "--root",
        args.occupancy_out_root,
        "--manifest",
        str(val_manifest),
        "--out",
        val_report,
    ]
    _run(validate_cmd, dry_run=args.dry_run)


def _expand_occ_jobs(configs: List[str], args: argparse.Namespace) -> List[OccJob]:
    jobs: List[OccJob] = []
    for cfg_path in configs:
        cfg = _read_yaml(cfg_path)
        cfg.setdefault("data", {})
        cfg["data"]["root"] = args.occupancy_out_root
        cfg["data"]["train_manifest"] = os.path.join(args.occupancy_out_root, "manifests", "train.jsonl")
        cfg["data"]["val_manifest"] = os.path.join(args.occupancy_out_root, "manifests", "val.jsonl")
        cfg["data"]["grid_shape"] = list(args.occupancy_grid_shape)
        cfg.setdefault("train", {})
        if args.occupancy_epochs > 0:
            cfg["train"]["epochs"] = int(args.occupancy_epochs)
        ckpt_dir = os.path.expanduser(cfg["train"]["ckpt_dir"])
        os.makedirs(ckpt_dir, exist_ok=True)
        resolved = os.path.join(ckpt_dir, "resolved_config.yaml")
        last_pt = os.path.join(ckpt_dir, "last.pt")
        if os.path.isfile(last_pt) and _last_epoch(os.path.join(ckpt_dir, "metrics_log.csv")) < int(cfg["train"]["epochs"]):
            cfg["train"]["resume"] = True
            cfg["train"]["resume_path"] = last_pt
        else:
            cfg["train"]["resume"] = False
            cfg["train"]["resume_path"] = ""
        _write_yaml(resolved, cfg)
        jobs.append(
            OccJob(
                name=str(cfg.get("name", Path(cfg_path).stem)),
                config_path=resolved,
                ckpt_dir=ckpt_dir,
                log_path=os.path.join(ckpt_dir, "train.log"),
                epochs=int(cfg["train"]["epochs"]),
            )
        )
    return jobs


def _is_occ_complete(job: OccJob) -> bool:
    return os.path.isfile(os.path.join(job.ckpt_dir, "last.pt")) and _last_epoch(os.path.join(job.ckpt_dir, "metrics_log.csv")) >= job.epochs


def _launch_occ(job: OccJob, gpu: str, python_bin: str, label: str = "occupancy") -> RunningOccJob:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = "Agg"
    if gpu != "cpu":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_f = open(job.log_path, "a", buffering=1)
    log_f.write(f"\n\n=== Launch {label} {job.name} gpu={gpu} ===\n")
    cmd = [
        python_bin,
        "-u",
        "-m",
        "raw2task.occupancy.train_occupancy",
        "--config",
        job.config_path,
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
    )
    output_thread = threading.Thread(
        target=_tee_process_output,
        args=(proc, log_f, f"{label}/{job.name}"),
        daemon=True,
    )
    output_thread.start()
    return RunningOccJob(job=job, gpu=gpu, process=proc, log_file=log_f, start_time=time.time(), output_thread=output_thread)


def _tee_process_output(proc: subprocess.Popen, log_f: object, prefix: str) -> None:
    """Mirror child process output to both the per-run log and this terminal."""
    if proc.stdout is None:
        return
    for line in proc.stdout:
        log_f.write(line)
        if line.endswith("\n"):
            print(f"[{prefix}] {line}", end="", flush=True)
        else:
            print(f"[{prefix}] {line}", flush=True)


def _run_voxel_jobs(
    args: argparse.Namespace,
    configs: List[str],
    run_root: str,
    label: str,
    title: str,
    note: str,
    guidance: str,
) -> None:
    jobs = _expand_occ_jobs(configs, args)
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()] if args.gpus else _detect_gpus()
    if not gpus:
        gpus = ["cpu"]
    max_parallel = args.max_parallel if args.max_parallel > 0 else len(gpus)
    slots = gpus[: max(1, min(max_parallel, len(gpus)))]
    print(f"[{label}] Jobs={len(jobs)} GPU slots={slots}")
    if args.dry_run:
        for job in jobs:
            state = "complete" if _is_occ_complete(job) else f"epoch {_last_epoch(os.path.join(job.ckpt_dir, 'metrics_log.csv'))}/{job.epochs}"
            print(f"[dry-run {label}] {job.name} [{state}] -> {job.ckpt_dir}")
        return

    pending = list(jobs)
    available = list(slots)
    running: List[RunningOccJob] = []
    failures: List[tuple[OccJob, int]] = []
    while pending or running:
        while pending and available:
            job = pending.pop(0)
            if args.skip_existing and _is_occ_complete(job):
                print(f"[{label} skip-complete] {job.name}")
                continue
            if os.path.isfile(os.path.join(job.ckpt_dir, "last.pt")) and not _is_occ_complete(job):
                print(f"[{label} resume] {job.name} from epoch {_last_epoch(os.path.join(job.ckpt_dir, 'metrics_log.csv'))}/{job.epochs}")
            gpu = available.pop(0)
            print(f"[{label} launch] {job.name} gpu={gpu}")
            print(f"        log: {job.log_path}")
            running.append(_launch_occ(job, gpu, args.python, label=label))

        time.sleep(5 if running else 0)
        still = []
        for rj in running:
            code = rj.process.poll()
            if code is None:
                still.append(rj)
                continue
            rj.log_file.write(f"\n=== Exit code {code} after {time.time() - rj.start_time:.1f}s ===\n")
            rj.output_thread.join(timeout=2.0)
            rj.log_file.close()
            available.append(rj.gpu)
            if code != 0:
                failures.append((rj.job, int(code)))
                print(f"[{label} failed] {rj.job.name} code={code}")
            else:
                print(f"[{label} done] {rj.job.name}")
        running = still

    if failures:
        for job, code in failures:
            print(f"  {label} failure: {job.name} code={code} log={job.log_path}")
        raise SystemExit(1)

    summary_path = os.path.join(run_root, "summary.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        fields = ["name", "ckpt_dir", "best_ckpt", "epoch", "train_loss", "train_feedback", "voxel_acc", "mIoU", "occupied_IoU"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for job in jobs:
            metrics = _best_occ_metrics(job.ckpt_dir)
            writer.writerow(
                {
                    "name": job.name,
                    "ckpt_dir": job.ckpt_dir,
                    "best_ckpt": _best_occ_ckpt(job.ckpt_dir),
                    **{k: metrics.get(k, "") for k in fields if k not in ("name", "ckpt_dir", "best_ckpt")},
                }
            )
    print(f"[{label}] summary: {summary_path}")
    _write_voxel_paper_artifacts(jobs, summary_path, run_root, title, note, guidance)
    print(f"[{label}] paper artifacts: {run_root}/paper_tables")


def _run_occupancy_jobs(args: argparse.Namespace) -> None:
    _run_voxel_jobs(
        args,
        OCC_CONFIGS,
        "runs/occupancy",
        "occupancy",
        "Observed 3D Occupancy Results",
        "This table reports camera-to-voxel observed semantic occupancy on KITTI-360-derived voxel labels. It is an observed-occupancy experiment, not amodal semantic scene completion.",
        "- Use this as 3D evidence that the same fixed-at-inference optics/sensor design can be optimized for voxel perception.\n- Do not claim state-of-the-art occupancy: this reference model is intentionally small and camera-only.\n- The fair claim is matched RGB vs fixed camera vs learnable optics/sensor under the same occupancy architecture and feedback controller.",
    )


def _run_3d_segmentation_jobs(args: argparse.Namespace) -> None:
    _run_voxel_jobs(
        args,
        SEG3D_CONFIGS,
        "runs/seg3d",
        "seg3d",
        "Observed 3D Semantic Segmentation Results",
        "This table reports camera-to-voxel 3D semantic segmentation on visible KITTI-360 semantic voxels derived from the official 3D semantic PLY annotations.",
        "- Use this as the direct 3D semantic segmentation evidence requested by reviewers.\n- The target is visible/observed 3D semantic segmentation, not hidden-surface completion.\n- The fair claim is matched RGB vs fixed camera vs learnable optics/sensor under the same voxel segmentation architecture and feedback controller.",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--segmentation-matrix", default="deeplens/projects/raw2task/configs/industry_paper_matrix.yaml")
    parser.add_argument("--segmentation-seeds", default="")
    parser.add_argument("--segmentation-only", default="")
    parser.add_argument("--skip-segmentation", action="store_true")
    parser.add_argument("--skip-occupancy", action="store_true")
    parser.add_argument("--skip-3d-segmentation", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=0)
    parser.add_argument("--gpus", default="")
    parser.add_argument("--kitti-root", default="/home/rk010/Desktop/Research/NurIPS/KITTI-360")
    parser.add_argument("--occupancy-out-root", default="data_external/kitti360_occupancy")
    parser.add_argument("--occupancy-stride", type=int, default=20)
    parser.add_argument("--occupancy-max-samples", type=int, default=0)
    parser.add_argument("--occupancy-min-valid-voxels", type=int, default=20)
    parser.add_argument("--occupancy-grid-shape", type=int, nargs=3, default=[16, 64, 128])
    parser.add_argument("--occupancy-epochs", type=int, default=0)
    parser.add_argument("--skip-occupancy-build", action="store_true")
    args = parser.parse_args()

    if not args.skip_segmentation:
        cmd = [
            args.python,
            "-u",
            "-m",
            "raw2task.run_paper_experiments",
            "--matrix",
            args.segmentation_matrix,
        ]
        if args.segmentation_only:
            cmd.extend(["--only", args.segmentation_only])
        if args.segmentation_seeds:
            cmd.extend(["--seeds", args.segmentation_seeds])
        if args.skip_existing:
            cmd.append("--skip-existing")
        if args.dry_run:
            cmd.append("--dry-run")
        if args.max_parallel > 0:
            cmd.extend(["--max-parallel", str(args.max_parallel)])
        if args.gpus:
            cmd.extend(["--gpus", args.gpus])
        _run(cmd, dry_run=False)

    needs_voxel_assets = (not args.skip_occupancy) or (not args.skip_3d_segmentation)
    if needs_voxel_assets:
        _prepare_occupancy_assets(args)
    if not args.skip_3d_segmentation:
        _run_3d_segmentation_jobs(args)
    if not args.skip_occupancy:
        _run_occupancy_jobs(args)


if __name__ == "__main__":
    main()
