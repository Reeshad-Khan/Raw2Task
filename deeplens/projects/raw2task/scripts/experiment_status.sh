#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

exec "${PYTHON_BIN}" -m deeplens.projects.raw2task.experiment_status "$@"
