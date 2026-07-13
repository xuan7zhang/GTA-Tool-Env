#!/bin/bash
# Run the GTA-Atomic OpenCompass eval. Usage:
#   run_eval.sh <results_subdir> [infer|eval|all] [reuse_timestamp]
# Env knobs consumed by configs/gta_atomic_env.py:
#   GTA_LLM_URL GTA_MODEL_NAME GTA_TOOLSERVER GTA_TOOLMETA GTA_EVAL_MODES GTA_MAX_TURN
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
conda activate $GTA_BIG/envs/opencompass

OUT=${1:?usage: run_eval.sh <results_subdir> [mode] [reuse_ts]}
MODE=${2:-all}
REUSE=${3:-}

WORKDIR=$GTA_BIG/results/$OUT
mkdir -p "$WORKDIR"
# record run metadata
env | grep -E '^GTA_' | sort > "$WORKDIR/run_env.txt" || true

cd $GTA_REPO/opencompass
EXTRA=()
[ "$MODE" != "all" ] && EXTRA+=(--mode "$MODE")
[ -n "$REUSE" ] && EXTRA+=(--reuse "$REUSE")

exec python run.py configs/gta_atomic_env.py \
    --max-num-workers "${GTA_INFER_WORKERS:-1}" --debug \
    -w "$WORKDIR" "${EXTRA[@]}"
