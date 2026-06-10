#!/usr/bin/env bash
set -euo pipefail

# One-command constrained-sensing workflow:
#   1. Probe fixed batch sizes on each requested GPU.
#   2. Select the conservative plan across GPUs.
#   3. Run the stress matrix with live progress and paper artifact refresh.
#
# This is separate from run_preflight_then_experiments.sh so stress evidence
# does not contaminate the clean/full KITTI-360 matrix folders.

MODE="fast"
GPUS="${GPUS:-${GPU_ID:-0,1}}"
MAX_BATCH="${MAX_BATCH:-16}"
FAST_SEEDS="${FAST_SEEDS:-0}"
FULL_SEEDS="${FULL_SEEDS:-0,1,2}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
SKIP_EXISTING=1
PROBE_STEPS="${PROBE_STEPS:-1}"
PROBE_MARGIN_MB="${PROBE_MARGIN_MB:-4096}"
PROBE_SAFETY_FACTOR="${PROBE_SAFETY_FACTOR:-0.8}"

FAST_MATRIX="deeplens/projects/raw2task/configs/sensor_stress_matrix_fast.yaml"
FULL_MATRIX="deeplens/projects/raw2task/configs/sensor_stress_matrix.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --gpu|--gpus)
      GPUS="$2"
      shift 2
      ;;
    --max-batch)
      MAX_BATCH="$2"
      shift 2
      ;;
    --fast-seeds)
      FAST_SEEDS="$2"
      shift 2
      ;;
    --full-seeds)
      FULL_SEEDS="$2"
      shift 2
      ;;
    --probe-steps)
      PROBE_STEPS="$2"
      shift 2
      ;;
    --probe-margin-mb)
      PROBE_MARGIN_MB="$2"
      shift 2
      ;;
    --probe-safety-factor)
      PROBE_SAFETY_FACTOR="$2"
      shift 2
      ;;
    --no-skip-existing)
      SKIP_EXISTING=0
      shift
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --help|-h)
      cat <<'EOF'
Usage: deeplens/projects/raw2task/scripts/run_sensor_stress_experiments.sh [options]

Options:
  --mode fast|full|both          Stress workflow to run (default: fast)
  --gpus <csv>                   GPUs for probing and training slots (default: 0,1)
  --max-batch <n>                Highest batch size to test (default: 16)
  --fast-seeds <csv>             Seeds for fast stress matrix (default: 0)
  --full-seeds <csv>             Seeds for full stress matrix (default: 0,1,2)
  --probe-steps <n>              Synthetic train steps per tested batch (default: 1)
  --probe-margin-mb <n>          Required free VRAM after probe (default: 4096)
  --probe-safety-factor <float>  Fraction of max fitting batch to use (default: 0.8)
  --no-skip-existing             Re-run completed jobs
  --python <path>                Python executable

Environment overrides:
  GPUS, MAX_BATCH, FAST_SEEDS, FULL_SEEDS, PYTHON_BIN,
  PROBE_STEPS, PROBE_MARGIN_MB, PROBE_SAFETY_FACTOR
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"
cd "${REPO_ROOT}"

mkdir -p runs

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONUNBUFFERED=1
export PYTHON_BIN

probe_fixed_plan() {
  local matrix="$1"
  local seeds="$2"
  local plan="$3"
  local probe_log="$4"
  local label="$5"

  local plan_dir
  plan_dir="$(dirname "${plan}")"
  local plan_base
  plan_base="$(basename "${plan}" .yaml)"
  local best_plan=""
  local best_batch=""
  local first=1

  IFS=',' read -r -a gpu_array <<< "${GPUS}"
  : > "${probe_log}"
  for gpu in "${gpu_array[@]}"; do
    gpu="$(echo "${gpu}" | xargs)"
    [[ -z "${gpu}" ]] && continue
    local gpu_plan="${plan_dir}/${plan_base}_gpu${gpu}.yaml"
    echo "[stress-workflow] ${label} preflight on GPU ${gpu} -> ${gpu_plan}" | tee -a "${probe_log}"
    deeplens/projects/raw2task/scripts/probe_batch_size.sh \
      --matrix "${matrix}" \
      --seeds "${seeds}" \
      --gpu "${gpu}" \
      --max-batch "${MAX_BATCH}" \
      --steps "${PROBE_STEPS}" \
      --margin-mb "${PROBE_MARGIN_MB}" \
      --safety-factor "${PROBE_SAFETY_FACTOR}" \
      --out "${gpu_plan}" \
      2>&1 | tee -a "${probe_log}"

    local min_batch
    min_batch="$("${PYTHON_BIN}" - "${gpu_plan}" <<'PY'
import sys, yaml
with open(sys.argv[1]) as f:
    plan = yaml.safe_load(f) or {}
vals = [int(v.get("batch_size", 0)) for v in (plan.get("experiments") or {}).values()]
print(min(vals) if vals else 0)
PY
)"
    if [[ ${first} -eq 1 || ${min_batch} -lt ${best_batch} ]]; then
      best_batch="${min_batch}"
      best_plan="${gpu_plan}"
      first=0
    fi
  done

  if [[ -z "${best_plan}" ]]; then
    echo "No GPU probe plan was produced. Check --gpus '${GPUS}'." >&2
    exit 1
  fi
  cp "${best_plan}" "${plan}"
  "${PYTHON_BIN}" - "${plan}" "${GPUS}" <<'PY'
import sys, yaml
path, gpus = sys.argv[1], sys.argv[2]
with open(path) as f:
    plan = yaml.safe_load(f) or {}
plan["training_gpus"] = gpus
plan["selection"] = "conservative_min_batch_across_probed_gpus"
with open(path, "w") as f:
    yaml.safe_dump(plan, f, sort_keys=False)
PY
  echo "[stress-workflow] selected fixed ${label} batch plan: ${plan}" | tee -a "${probe_log}"
}

run_matrix() {
  local matrix="$1"
  local seeds="$2"
  local plan="$3"
  local label="$4"
  local extra=()
  if [[ ${SKIP_EXISTING} -eq 1 ]]; then
    extra+=(--skip-existing)
  fi
  echo "[stress-workflow] running ${label} matrix on GPUs ${GPUS}"
  deeplens/projects/raw2task/scripts/run_paper_experiments.sh \
    --matrix "${matrix}" \
    --batch-plan "${plan}" \
    --gpus "${GPUS}" \
    --seeds "${seeds}" \
    --robustness-experiments "" \
    "${extra[@]}"
}

case "${MODE}" in
  fast)
    probe_fixed_plan "${FAST_MATRIX}" "${FAST_SEEDS}" "runs/batch_size_plan_stress_fast.yaml" "runs/stress_fast_preflight.log" "fast stress"
    run_matrix "${FAST_MATRIX}" "${FAST_SEEDS}" "runs/batch_size_plan_stress_fast.yaml" "fast stress"
    ;;
  full)
    probe_fixed_plan "${FULL_MATRIX}" "${FULL_SEEDS}" "runs/batch_size_plan_stress_full.yaml" "runs/stress_full_preflight.log" "full stress"
    run_matrix "${FULL_MATRIX}" "${FULL_SEEDS}" "runs/batch_size_plan_stress_full.yaml" "full stress"
    ;;
  both)
    probe_fixed_plan "${FAST_MATRIX}" "${FAST_SEEDS}" "runs/batch_size_plan_stress_fast.yaml" "runs/stress_fast_preflight.log" "fast stress"
    run_matrix "${FAST_MATRIX}" "${FAST_SEEDS}" "runs/batch_size_plan_stress_fast.yaml" "fast stress"
    probe_fixed_plan "${FULL_MATRIX}" "${FULL_SEEDS}" "runs/batch_size_plan_stress_full.yaml" "runs/stress_full_preflight.log" "full stress"
    run_matrix "${FULL_MATRIX}" "${FULL_SEEDS}" "runs/batch_size_plan_stress_full.yaml" "full stress"
    ;;
  *)
    echo "Unknown --mode '${MODE}'. Expected fast, full, or both." >&2
    exit 2
    ;;
esac
