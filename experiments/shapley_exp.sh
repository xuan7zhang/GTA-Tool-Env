#!/usr/bin/env bash
# Exact tool Shapley experiment: generate and run every one of the 2^n masks.
#
# Usage:
#   experiments/shapley_exp.sh [run_prefix] [toolmeta.json] [seed_csv]
#
# Optional environment variables:
#   SHAPLEY_TOOLS=OCR,Calculator       candidates; default is every metadata tool
#   SHAPLEY_FIXED_TOOLS=GoogleSearch   always available, not attributed
#   SHAPLEY_NATURAL_MENUS=1            use per-task natural-menu intersection
#   SHAPLEY_WORK=/path/to/work         manifest + analysis output directory
#   SHAPLEY_METRIC=answer_acc          GTA end-evaluator metric
set -euo pipefail

LAB=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$LAB/scripts/common_env.sh"
# The checked-out branch is authoritative even when common_env uses a legacy
# cluster default.
export GTA_LAB="$LAB"
export GTA_REPO="$LAB/GTA"

PREFIX=${1:-shapley}
TOOLMETA=${2:-$GTA_BIG/data/gta_dataset/toolmeta.json}
SEED_CSV=${3:-0}
WORK=${SHAPLEY_WORK:-$GTA_BIG/results/${PREFIX}_work}
MANIFEST=$WORK/manifest.jsonl
ANALYSIS=$WORK/analysis
METRIC=${SHAPLEY_METRIC:-answer_acc}

IFS=',' read -r -a SEEDS <<< "$SEED_CSV"
GEN_ARGS=(
  --toolmeta "$TOOLMETA"
  --out "$MANIFEST"
  --run-prefix "$PREFIX"
  --seeds "${SEEDS[@]}"
)
if [[ -n "${SHAPLEY_TOOLS:-}" ]]; then
  GEN_ARGS+=(--tools "$SHAPLEY_TOOLS")
fi
if [[ -n "${SHAPLEY_FIXED_TOOLS:-}" ]]; then
  GEN_ARGS+=(--fixed-tools "$SHAPLEY_FIXED_TOOLS")
fi
if [[ "${SHAPLEY_NATURAL_MENUS:-0}" == "1" ]]; then
  GEN_ARGS+=(--natural-task-menus)
fi

mkdir -p "$WORK"
unset GTA_PREDICTED_MASK
python "$LAB/envgen/gen_shapley.py" "${GEN_ARGS[@]}"
bash "$LAB/scripts/sweep.sh" "$MANIFEST"
python "$LAB/analysis/compute_shapley.py" \
  --manifest "$MANIFEST" \
  --results "$GTA_BIG/results" \
  --out "$ANALYSIS" \
  --metric "$METRIC" \
  --require-complete
