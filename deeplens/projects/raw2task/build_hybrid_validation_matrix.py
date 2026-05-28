"""Build validation/control matrices from hybrid camera-search results.

The search stage answers "which camera candidates look promising?"  This helper
turns that answer into a fair follow-up matrix:

- learned co-design initialized from a selected candidate
- fixed camera using the same physical candidate
- RGB baseline under the same data/model/training budget

It intentionally writes a normal run_paper_experiments matrix so the downstream
runner, status script, plots, and paper tables stay unchanged.
"""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from deeplens.projects.raw2task.hybrid_camera_search import _find_template
from deeplens.projects.raw2task.run_review_matrix import _deep_update


SENSOR_FIELDS = [
    "exposure_init",
    "bit_depth",
    "read_noise_std",
    "shot_noise_scale",
    "cfa_init_floor",
]

PSF_FIELDS = [
    "base_sigma_px",
    "max_sigma_px",
    "max_shift_px",
    "field_sigma",
]

MODEL_IDS = {
    "segformer_b0": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    "segformer_b1": "nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
    "segformer_b2": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
}


def _read_csv(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open("r", newline="") as f:
        return list(csv.DictReader(f))


def _write_yaml(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _parse_csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _candidate_rows(ranked_csv: Path, candidate_ids: Iterable[str], top_k: int) -> List[Dict[str, str]]:
    rows = _read_csv(ranked_csv)
    wanted = set(str(x) for x in candidate_ids)
    selected: List[Dict[str, str]] = []
    seen = set()
    if wanted:
        for row in rows:
            cid = str(row.get("candidate", ""))
            if cid in wanted and cid not in seen:
                selected.append(row)
                seen.add(cid)
    else:
        selected = rows[: max(1, int(top_k))]
    if not selected:
        raise ValueError(f"No candidate rows selected from {ranked_csv}")
    return selected


def _set_candidate_camera(overrides: Dict[str, Any], row: Dict[str, str], learn: bool) -> None:
    sensor = overrides.setdefault("sensor", {})
    lens = overrides.setdefault("lens", {})
    trainable_psf = lens.setdefault("trainable_psf", {})
    for field in SENSOR_FIELDS:
        if row.get(field, "") != "":
            value: Any = float(row[field])
            if field == "bit_depth":
                value = int(round(value))
            sensor[field] = value
    for field in PSF_FIELDS:
        if row.get(field, "") != "":
            trainable_psf[field] = float(row[field])

    lens["learn_optics"] = bool(learn)
    sensor["learn_cfa"] = bool(learn)
    sensor["learn_exposure"] = bool(learn)
    sensor["learn_noise"] = bool(learn)
    sensor["learn_bit_depth"] = bool(learn)
    if not learn:
        sensor["regularization"] = {}
        trainable_psf.setdefault("regularization", {})


def _set_train_budget(overrides: Dict[str, Any], args: argparse.Namespace, learn: bool) -> None:
    data = overrides.setdefault("data", {})
    train = overrides.setdefault("train", {})
    train["epochs"] = int(args.epochs)
    train["early_stop_patience"] = int(args.early_stop_patience)
    train["checkpoint_average_k"] = int(args.checkpoint_average_k)
    train["keep_last_n"] = max(int(args.checkpoint_average_k) + 1, int(train.get("keep_last_n", 3)))
    train["sensor_warmup_epochs"] = int(args.sensor_warmup_epochs) if learn else 0
    train["camera_curriculum_epochs"] = int(args.camera_curriculum_epochs) if learn else 0
    if args.deployable_task_weight >= 0:
        train["deployable_task_weight"] = float(args.deployable_task_weight) if learn else 0.0
    train["aux_losses_update_adapter"] = bool(learn)
    train["freeze_backbone_during_sensor_stage"] = bool(learn)
    train["sensor_freeze_after_best_patience"] = int(args.sensor_freeze_after_best_patience) if learn else 0
    train["sensor_freeze_restore_best"] = bool(learn)
    if args.img_height > 0 and args.img_width > 0:
        data["img_size"] = [int(args.img_height), int(args.img_width)]
    if args.max_train_samples >= 0:
        if args.max_train_samples == 0:
            data.pop("max_train_samples", None)
        else:
            data["max_train_samples"] = int(args.max_train_samples)
    if args.max_val_samples >= 0:
        if args.max_val_samples == 0:
            data.pop("max_val_samples", None)
        else:
            data["max_val_samples"] = int(args.max_val_samples)
    if args.batch_size > 0:
        data["batch_size"] = int(args.batch_size)
    if args.accum_steps > 0:
        train["accum_steps"] = int(args.accum_steps)


def _set_model(overrides: Dict[str, Any], model_name: str) -> None:
    if not model_name:
        return
    model = overrides.setdefault("model", {})
    model["name"] = model_name
    if model_name in MODEL_IDS:
        model["hf_model_id"] = MODEL_IDS[model_name]


def build_matrix(args: argparse.Namespace) -> Path:
    with Path(args.base_matrix).open("r") as f:
        matrix = yaml.safe_load(f) or {}
    code_t = _find_template(matrix, args.codesign_template)
    fixed_t = _find_template(matrix, args.fixed_template)
    rgb_t = _find_template(matrix, args.rgb_template)
    candidates = _candidate_rows(Path(args.ranked_csv), _parse_csv(args.candidates), args.top_k)

    experiments = []
    for row in candidates:
        cid = int(float(row["candidate"]))
        suffix = f"cand{cid:03d}"

        if "codesign" in args.modes:
            overrides = copy.deepcopy(code_t.get("overrides", {}))
            _set_candidate_camera(overrides, row, learn=True)
            _set_train_budget(overrides, args, learn=True)
            _set_model(overrides, args.model)
            experiments.append({"name": f"{args.prefix}_{suffix}_codesign_{args.model}", "overrides": overrides})

        if "fixed" in args.modes:
            overrides = copy.deepcopy(fixed_t.get("overrides", {}))
            _set_candidate_camera(overrides, row, learn=False)
            _set_train_budget(overrides, args, learn=False)
            _set_model(overrides, args.model)
            experiments.append({"name": f"{args.prefix}_{suffix}_fixed_{args.model}", "overrides": overrides})

    if "rgb" in args.modes:
        overrides = copy.deepcopy(rgb_t.get("overrides", {}))
        _set_train_budget(overrides, args, learn=False)
        _set_model(overrides, args.rgb_model or args.model)
        experiments.append({"name": f"{args.prefix}_rgb_{args.rgb_model or args.model}", "overrides": overrides})

    payload = {
        "base_config": matrix["base_config"],
        "out_root": str(Path(args.out_root).expanduser().resolve()),
        "seeds": [int(x) for x in _parse_csv(args.seeds)] if args.seeds else [0],
        "experiments": experiments,
    }
    out_path = Path(args.output).expanduser().resolve()
    _write_yaml(out_path, payload)
    print(f"Wrote validation matrix: {out_path}")
    print(f"Experiments: {len(experiments)}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-matrix", required=True)
    parser.add_argument("--ranked-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--codesign-template", required=True)
    parser.add_argument("--fixed-template", required=True)
    parser.add_argument("--rgb-template", required=True)
    parser.add_argument("--candidates", default="", help="Comma-separated candidate ids. Default: top-k from ranked CSV.")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--modes", default="codesign,fixed,rgb")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--model", default="segformer_b1")
    parser.add_argument("--rgb-model", default="")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--checkpoint-average-k", type=int, default=3)
    parser.add_argument("--sensor-warmup-epochs", type=int, default=2)
    parser.add_argument("--camera-curriculum-epochs", type=int, default=4)
    parser.add_argument("--deployable-task-weight", type=float, default=0.12)
    parser.add_argument("--sensor-freeze-after-best-patience", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=-1, help="-1 keep template, 0 full dataset, >0 cap.")
    parser.add_argument("--max-val-samples", type=int, default=-1, help="-1 keep template, 0 full val, >0 cap.")
    parser.add_argument("--img-height", type=int, default=0)
    parser.add_argument("--img-width", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--accum-steps", type=int, default=0)
    args = parser.parse_args()
    args.modes = set(_parse_csv(args.modes))
    build_matrix(args)


if __name__ == "__main__":
    main()
