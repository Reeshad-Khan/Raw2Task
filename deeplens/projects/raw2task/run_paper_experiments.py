"""One-command paper experiment orchestrator.

This is the production runner for the optics-sensor co-design paper. It expands
the YAML matrix into named experiment/seed jobs, schedules them sequentially or
one-per-GPU, and refreshes paper-ready artifacts after every completed job:

- summary.csv plus main_results.csv/.md
- validation/inference visual panels from each run
- camera design inventory and initial-to-final camera deltas
- robustness tables/plots for selected experiments
- paper figures and pipeline diagram

The runner is resumable. If a run directory already contains ``last.pt`` and
``--skip-existing`` is set, it summarizes that run without retraining.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import yaml

from deeplens.projects.raw2task.run_end2end_pipeline import (
    _matrix_out_root,
    _parse_csv_list,
    aggregate_main_results,
    collect_camera_measurements,
    collect_claim_readiness,
    collect_design_inventory,
    collect_qualitative_outputs,
    collect_robustness,
    plot_paper_figures,
    run_robustness,
    write_run_manifest,
)
from deeplens.projects.raw2task.run_review_matrix import (
    _avg_checkpoint_metrics,
    _best_avg_ckpt_path,
    _best_ckpt_path,
    _deep_set,
    _deep_update,
    _latest_best_metrics,
)


SUMMARY_FIELDS = [
    "experiment",
    "seed",
    "ckpt_dir",
    "best_ckpt",
    "avg_best_ckpt",
    "design_json",
    "best_val",
    "pixel_acc",
    "mIoU",
    "avg_bestk_mIoU",
    "avg_bestk_pixel_acc",
    "avg_bestk_metrics",
    "params_model_m",
    "latency_chain_ms",
]


@dataclass
class Job:
    experiment: str
    seed: int
    ckpt_dir: str
    config_path: str
    log_path: str
    cfg: Dict[str, Any]


@dataclass
class RunningJob:
    job: Job
    gpu: str
    process: subprocess.Popen
    log_file: Any
    start_time: float
    output_thread: threading.Thread


def _detect_gpus() -> List[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        vals = [x.strip() for x in visible.split(",") if x.strip()]
        return vals or ["0"]
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        vals = [x.strip() for x in out.splitlines() if x.strip()]
        return vals or ["0"]
    except Exception:
        return ["cpu"]


def _load_matrix(matrix_path: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    with open(os.path.expanduser(matrix_path), "r") as f:
        matrix = yaml.safe_load(f)
    with open(os.path.expanduser(matrix["base_config"]), "r") as f:
        base_cfg = yaml.safe_load(f)
    return matrix, base_cfg


def _expand_jobs(
    matrix_path: str,
    only: Iterable[str],
    seeds_override: Optional[Iterable[int]],
    order_policy: str = "codesign_first",
) -> List[Job]:
    matrix, base_cfg = _load_matrix(matrix_path)
    out_root = os.path.abspath(os.path.expanduser(matrix.get("out_root", "./runs/raw2task_review_matrix")))
    os.makedirs(out_root, exist_ok=True)

    only_set = set(only or [])
    seeds = list(seeds_override) if seeds_override is not None else list(matrix.get("seeds", [base_cfg.get("seed", 0)]))
    experiments = list(matrix.get("experiments", []))
    if only_set:
        experiments = [exp for exp in experiments if exp["name"] in only_set]
    if order_policy and order_policy != "matrix":
        experiments = sorted(
            enumerate(experiments),
            key=lambda item: (_experiment_priority(str(item[1].get("name", "")), order_policy), item[0]),
        )
        experiments = [exp for _, exp in experiments]

    jobs: List[Job] = []
    for exp in experiments:
        exp_name = exp["name"]
        overrides = exp.get("overrides", {})
        for seed in seeds:
            cfg = copy.deepcopy(base_cfg)
            cfg["seed"] = int(seed)
            _deep_update(cfg, overrides)
            ckpt_dir = os.path.join(out_root, f"{exp_name}_seed{seed}")
            os.makedirs(ckpt_dir, exist_ok=True)
            _deep_set(cfg, "train.ckpt_dir", ckpt_dir)
            _deep_set(cfg, "train.resume", False)
            _deep_set(cfg, "train.resume_path", "")
            cfg_path = os.path.join(ckpt_dir, "resolved_config.yaml")
            log_path = os.path.join(ckpt_dir, "train.log")
            jobs.append(
                Job(
                    experiment=exp_name,
                    seed=int(seed),
                    ckpt_dir=ckpt_dir,
                    config_path=cfg_path,
                    log_path=log_path,
                    cfg=cfg,
                )
            )
    return jobs


def _experiment_priority(name: str, order_policy: str = "codesign_first") -> int:
    """Stable launch priority for paper matrices.

    We keep YAML anchors intact and apply the user's preferred scheduling here:
    learned optics/sensor rows run first, then raw/learned-sensor ablations,
    then fixed/RGB references.
    """
    n = name.lower()
    if order_policy == "new_idea_first":
        is_codesign = "codesign" in n or "co_design" in n
        is_rgb_or_fixed = "fixed" in n or "rgb" in n or "reference" in n or "baseline" in n
        if is_codesign and ("segformer_b1" in n or "segformer_b2" in n):
            return 0
        if is_codesign and ("ddrlite" in n or "ddrnet" in n or "pidnet" in n or "biselite" in n or "bisenet" in n):
            return 1
        if is_codesign and ("tasktokens" in n or "task_tokens" in n):
            return 2
        if is_codesign:
            return 3
        if is_rgb_or_fixed and ("segformer_b1" in n or "segformer_b2" in n):
            return 4
        if is_rgb_or_fixed and ("ddrlite" in n or "ddrnet" in n or "pidnet" in n or "biselite" in n or "bisenet" in n):
            return 5
        if "sensor_only" in n or "optics_only" in n or "learned" in n or "raw" in n:
            return 6
        if is_rgb_or_fixed:
            return 7
        return 8
    if order_policy != "codesign_first":
        return 0
    if "codesign" in n or "co_design" in n:
        return 0
    if "sensor_only" in n or "optics_only" in n or "learned" in n or "raw" in n:
        return 1
    if "fixed" in n or "rgb" in n or "reference" in n or "baseline" in n:
        return 2
    return 3


def _load_batch_plan(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    full = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(full):
        raise FileNotFoundError(f"Batch plan not found: {full}")
    with open(full, "r") as f:
        if full.endswith((".yaml", ".yml")):
            return yaml.safe_load(f) or {}
        return json.load(f)


def _apply_batch_plan(jobs: List[Job], plan: Dict[str, Any]) -> None:
    """Apply a fixed preflight batch-size plan to expanded jobs."""
    if not plan:
        return
    entries = plan.get("experiments", plan.get("jobs", plan))
    if not isinstance(entries, dict):
        raise ValueError("Batch plan must be a mapping keyed by experiment name.")
    for job in jobs:
        item = entries.get(job.experiment)
        if item is None:
            item = entries.get(f"{job.experiment}_seed{job.seed}")
        if item is None:
            continue
        if not isinstance(item, dict):
            raise ValueError(f"Batch plan entry for {job.experiment} must be a mapping.")
        data_cfg = job.cfg.setdefault("data", {})
        train_cfg = job.cfg.setdefault("train", {})
        if "batch_size" in item:
            data_cfg["batch_size"] = int(item["batch_size"])
        if "accum_steps" in item:
            train_cfg["accum_steps"] = int(item["accum_steps"])
        train_cfg["batch_plan_source"] = str(plan.get("path", "batch_plan"))


def _write_job_config(job: Job) -> None:
    os.makedirs(job.ckpt_dir, exist_ok=True)
    _prepare_resume(job)
    with open(job.config_path, "w") as f:
        yaml.safe_dump(job.cfg, f, sort_keys=False)


def _read_summary(summary_path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(summary_path):
        return []
    with open(summary_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _write_summary(summary_path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    latest: Dict[tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        latest[(str(row.get("experiment", "")), str(row.get("seed", "")))] = row
    out = sorted(latest.values(), key=lambda r: (str(r.get("experiment", "")), int(r.get("seed", 0) or 0)))
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in out:
            writer.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})


def _summarize_job(job: Job) -> Dict[str, Any]:
    metrics = _latest_best_metrics(job.ckpt_dir)
    avg_metrics = _avg_checkpoint_metrics(job.ckpt_dir)
    eff_chain: Dict[str, Any] = {}
    eff_model: Dict[str, Any] = {}
    for filename, target in [("efficiency_chain.json", eff_chain), ("efficiency_model.json", eff_model)]:
        path = os.path.join(job.ckpt_dir, filename)
        if os.path.isfile(path):
            with open(path, "r") as f:
                target.update(json.load(f))
    best_ckpt = _best_ckpt_path(job.ckpt_dir)
    avg_best_ckpt = _best_avg_ckpt_path(job.ckpt_dir)
    design_json = os.path.join(job.ckpt_dir, "camera_design_best.json")
    if not os.path.isfile(design_json):
        design_json = ""
    return {
        "experiment": job.experiment,
        "seed": job.seed,
        "ckpt_dir": job.ckpt_dir,
        "best_ckpt": best_ckpt,
        "avg_best_ckpt": avg_best_ckpt,
        "design_json": design_json,
        "best_val": metrics.get("best_val", ""),
        "pixel_acc": metrics.get("pixel_acc", ""),
        "mIoU": metrics.get("mIoU", ""),
        "avg_bestk_mIoU": avg_metrics.get("mIoU", ""),
        "avg_bestk_pixel_acc": avg_metrics.get("pixel_acc", ""),
        "avg_bestk_metrics": avg_metrics.get("metrics_path", ""),
        "params_model_m": eff_model.get("params_m", ""),
        "latency_chain_ms": eff_chain.get("latency_ms", ""),
    }


def _refresh_artifacts(
    matrix_path: str,
    summary_path: str,
    out_root: str,
    tables_dir: str,
    robustness_experiments: Iterable[str],
    robustness_sweep: str,
    max_robustness_batches: int,
    force_robustness: bool,
) -> None:
    summary_rows = aggregate_main_results(summary_path, tables_dir)
    run_robustness(
        summary_rows,
        experiments=robustness_experiments,
        sweep=robustness_sweep,
        max_batches=max_robustness_batches,
        force=force_robustness,
    )
    collect_robustness(summary_rows, tables_dir)
    collect_design_inventory(summary_rows, tables_dir)
    collect_camera_measurements(summary_rows, tables_dir)
    collect_claim_readiness(summary_rows, tables_dir)
    collect_qualitative_outputs(summary_rows, tables_dir)
    write_run_manifest(matrix_path, out_root, tables_dir)
    plot_paper_figures(tables_dir)


def _launch(job: Job, gpu: str, python_bin: str) -> RunningJob:
    _write_job_config(job)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = "Agg"
    if gpu != "cpu":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_f = open(job.log_path, "a", buffering=1)
    log_f.write(f"\n\n=== Launch {job.experiment} seed={job.seed} gpu={gpu} ===\n")
    bs = (job.cfg.get("data") or {}).get("batch_size", "")
    accum = (job.cfg.get("train") or {}).get("accum_steps", "")
    plan_source = (job.cfg.get("train") or {}).get("batch_plan_source", "")
    batch_msg = f"[batch] batch_size={bs} accum_steps={accum}"
    if plan_source:
        batch_msg += f" source={plan_source}"
    log_f.write(batch_msg + "\n")
    print(f"[batch] {job.experiment}/s{job.seed}: batch_size={bs} accum_steps={accum}", flush=True)
    cmd = [
        python_bin,
        "-u",
        "-m",
        "deeplens.projects.raw2task.train_extended",
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
    prefix = f"{job.experiment}/s{job.seed}"
    output_thread = threading.Thread(
        target=_tee_process_output,
        args=(proc, log_f, prefix),
        daemon=True,
    )
    output_thread.start()
    return RunningJob(job=job, gpu=gpu, process=proc, log_file=log_f, start_time=time.time(), output_thread=output_thread)


def _tee_process_output(proc: subprocess.Popen, log_f: Any, prefix: str) -> None:
    """Mirror child process output to both the per-run log and this terminal."""
    if proc.stdout is None:
        return
    for line in proc.stdout:
        log_f.write(line)
        if line.endswith("\n"):
            print(f"[{prefix}] {line}", end="", flush=True)
        else:
            print(f"[{prefix}] {line}", flush=True)


def _last_logged_epoch(job: Job) -> int:
    rows = _read_csv(os.path.join(job.ckpt_dir, "epoch_summary.csv"))
    if not rows:
        return 0
    try:
        return int(float(rows[-1].get("epoch", 0)))
    except Exception:
        return 0


def _target_epochs(job: Job) -> int:
    try:
        return int(((job.cfg.get("train") or {}).get("epochs", 0)))
    except Exception:
        return 0


def _is_complete(job: Job) -> bool:
    target = _target_epochs(job)
    if target <= 0:
        return False
    if not os.path.isfile(os.path.join(job.ckpt_dir, "last.pt")):
        return False
    if _last_logged_epoch(job) >= target:
        return True
    complete_json = os.path.join(job.ckpt_dir, "train_complete.json")
    if os.path.isfile(complete_json):
        return True
    log_path = os.path.join(job.ckpt_dir, "train.log")
    if os.path.isfile(log_path):
        try:
            tail = open(log_path, "r", errors="ignore").read()[-20000:]
            if "[EarlyStop]" in tail or "TRAINING COMPLETE" in tail:
                return True
        except Exception:
            pass
    return False


def _has_resume_checkpoint(job: Job) -> bool:
    return os.path.isfile(os.path.join(job.ckpt_dir, "last.pt"))


def _prepare_resume(job: Job) -> None:
    """Resume partial jobs from last.pt; leave empty jobs as fresh runs."""
    last_path = os.path.join(job.ckpt_dir, "last.pt")
    if _has_resume_checkpoint(job) and not _is_complete(job):
        _deep_set(job.cfg, "train.resume", True)
        _deep_set(job.cfg, "train.resume_path", last_path)
    else:
        _deep_set(job.cfg, "train.resume", False)
        _deep_set(job.cfg, "train.resume_path", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="deeplens/projects/raw2task/configs/industry_paper_matrix.yaml")
    parser.add_argument("--only", default="", help="Comma-separated experiment names.")
    parser.add_argument("--seeds", default="", help="Comma-separated seed override.")
    parser.add_argument("--tables-dir", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fresh-summary", action="store_true")
    parser.add_argument("--summary-only", action="store_true", help="Refresh summary.csv only; skip paper artifact and figure generation.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=0, help="0 means one job per detected GPU.")
    parser.add_argument(
        "--order-policy",
        default="codesign_first",
        choices=["new_idea_first", "codesign_first", "matrix"],
        help="Launch order inside the matrix.",
    )
    parser.add_argument(
        "--gpus",
        default="",
        help="Comma-separated GPU ids. Default: CUDA_VISIBLE_DEVICES or nvidia-smi. Use 'cpu' for CPU sequential.",
    )
    parser.add_argument(
        "--robustness-experiments",
        default=(
            "kitti360_codesign_liteseg,kitti360_rgb_liteseg,kitti360_no_optics,"
            "kitti360_fixed_camera_frontend,codesign_full_liteseg,rgb_liteseg,"
            "kitti360_occupancy_codesign_liteseg_aspp,"
            "kitti360_occupancy_fixed_camera_liteseg_aspp,"
            "kitti360_occupancy_rgb_liteseg_aspp"
        ),
    )
    parser.add_argument("--robustness-sweep", default="")
    parser.add_argument("--max-robustness-batches", type=int, default=0)
    parser.add_argument("--force-robustness", action="store_true")
    parser.add_argument(
        "--batch-plan",
        default="",
        help="YAML/JSON produced by probe_batch_size.py. Applies fixed batch_size/accum_steps before launch.",
    )
    args = parser.parse_args()

    matrix_path = os.path.abspath(os.path.expanduser(args.matrix))
    out_root = _matrix_out_root(matrix_path)
    tables_dir = os.path.abspath(os.path.expanduser(args.tables_dir or os.path.join(out_root, "paper_tables")))
    summary_path = os.path.join(out_root, "summary.csv")
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    seeds = [int(x) for x in _parse_csv_list(args.seeds)] if args.seeds else None
    jobs = _expand_jobs(
        matrix_path,
        only=_parse_csv_list(args.only),
        seeds_override=seeds,
        order_policy=args.order_policy,
    )
    batch_plan = _load_batch_plan(args.batch_plan)
    if batch_plan:
        batch_plan["path"] = os.path.abspath(os.path.expanduser(args.batch_plan))
        _apply_batch_plan(jobs, batch_plan)
    gpus = _parse_csv_list(args.gpus) if args.gpus else _detect_gpus()
    if not gpus:
        gpus = ["cpu"]
    max_parallel = int(args.max_parallel) if args.max_parallel > 0 else len(gpus)
    max_parallel = max(1, min(max_parallel, len(gpus)))
    gpu_slots = gpus[:max_parallel]

    print(f"Matrix: {matrix_path}")
    print(f"Out root: {out_root}")
    print(f"Tables: {tables_dir}")
    print(f"Jobs: {len(jobs)}")
    print(f"GPU slots: {gpu_slots}")
    if args.dry_run:
        for job in jobs:
            if _is_complete(job):
                state = f"complete epoch {_last_logged_epoch(job)}/{_target_epochs(job)}"
            elif _has_resume_checkpoint(job):
                state = f"resume epoch {_last_logged_epoch(job)}/{_target_epochs(job)}"
            else:
                state = "fresh"
            bs = (job.cfg.get("data") or {}).get("batch_size", "")
            accum = (job.cfg.get("train") or {}).get("accum_steps", "")
            print(
                f"[dry-run] {job.experiment} seed={job.seed} [{state}] "
                f"batch_size={bs} accum_steps={accum} -> {job.ckpt_dir}"
            )
        return

    if args.fresh_summary and os.path.exists(summary_path):
        os.remove(summary_path)
    summary_rows = _read_summary(summary_path)
    available = list(gpu_slots)
    pending = list(jobs)
    running: List[RunningJob] = []
    failures: List[tuple[Job, int]] = []
    robustness_experiments = _parse_csv_list(args.robustness_experiments)

    try:
        while pending or running:
            while pending and available:
                job = pending.pop(0)
                if args.skip_existing and _is_complete(job):
                    print(f"[skip-complete] {job.experiment} seed={job.seed}")
                    summary_rows.append(_summarize_job(job))
                    _write_summary(summary_path, summary_rows)
                    if not args.summary_only:
                        _refresh_artifacts(
                            matrix_path,
                            summary_path,
                            out_root,
                            tables_dir,
                            [],
                            args.robustness_sweep,
                            args.max_robustness_batches,
                            args.force_robustness,
                        )
                    continue
                if _has_resume_checkpoint(job) and not _is_complete(job):
                    print(
                        f"[resume] {job.experiment} seed={job.seed} "
                        f"from epoch {_last_logged_epoch(job)} / {_target_epochs(job)}"
                    )
                gpu = available.pop(0)
                print(f"[launch] {job.experiment} seed={job.seed} gpu={gpu}")
                print(f"        log: {job.log_path}")
                running.append(_launch(job, gpu, args.python))

            time.sleep(5 if running else 0)
            still_running: List[RunningJob] = []
            for rj in running:
                code = rj.process.poll()
                if code is None:
                    still_running.append(rj)
                    continue
                rj.log_file.write(f"\n=== Exit code {code} after {time.time() - rj.start_time:.1f}s ===\n")
                rj.output_thread.join(timeout=2.0)
                rj.log_file.close()
                available.append(rj.gpu)
                if code != 0:
                    failures.append((rj.job, int(code)))
                    print(f"[failed] {rj.job.experiment} seed={rj.job.seed} code={code}")
                    continue
                print(f"[done] {rj.job.experiment} seed={rj.job.seed}")
                summary_rows.append(_summarize_job(rj.job))
                _write_summary(summary_path, summary_rows)
                if not args.summary_only:
                    _refresh_artifacts(
                        matrix_path,
                        summary_path,
                        out_root,
                        tables_dir,
                        [],
                        args.robustness_sweep,
                        args.max_robustness_batches,
                        args.force_robustness,
                    )
            running = still_running
    except KeyboardInterrupt:
        print("Interrupted; terminating running jobs...")
        for rj in running:
            rj.process.terminate()
            rj.output_thread.join(timeout=2.0)
            rj.log_file.close()
        raise

    if failures:
        print("Failures:")
        for job, code in failures:
            print(f"  {job.experiment} seed={job.seed} code={code} log={job.log_path}")
        raise SystemExit(1)

    if not args.summary_only:
        _refresh_artifacts(
            matrix_path,
            summary_path,
            out_root,
            tables_dir,
            robustness_experiments,
            args.robustness_sweep,
            args.max_robustness_batches,
            args.force_robustness,
        )
    print(f"All experiments completed. Paper artifacts: {tables_dir}")


if __name__ == "__main__":
    main()
