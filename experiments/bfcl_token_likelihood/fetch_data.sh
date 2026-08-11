#!/usr/bin/env bash
# Pull the BFCL v1 "multiple" category (single ground-truth function per task,
# 2-4 candidate functions in the prompt -- the category that most directly
# matches "given a task, pick the right tool out of several plausible ones").
#
# The HF repo has renamed the original BFCL v1 files to their v3 filenames in
# place (same category semantics: non-executable, single-turn, AST-scored);
# there is no separately versioned "v1" file anymore. `multiple` /
# `simple` / `parallel` / `parallel_multiple` are the four that correspond to
# the original v1 categories.
set -euo pipefail
OUT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/runtime/bfcl_v1_pilot_data}"
mkdir -p "$OUT"
BASE="https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main"
curl -sS -L "$BASE/BFCL_v3_multiple.json" -o "$OUT/questions.json"
curl -sS -L "$BASE/possible_answer/BFCL_v3_multiple.json" -o "$OUT/answers.json"
echo "wrote $(wc -l < "$OUT/questions.json") tasks -> $OUT"
