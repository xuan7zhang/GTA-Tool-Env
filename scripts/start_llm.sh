#!/bin/bash
# Serve the baseline LLM with LMDeploy (OpenAI-compatible endpoint).
# Run ON the GPU node (e.g. inside tmux, or via srun --overlap).
# Env knobs: GTA_LLM_GPUS (e.g. "0" or "0,1"), GTA_LLM_PORT, GTA_MODEL_PATH,
#            GTA_MODEL_NAME, GTA_KV_FRAC (kv-cache fraction, default 0.5).
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
conda activate $GTA_BIG/envs/lmdeploy

MODEL_PATH=${GTA_MODEL_PATH:-$GTA_BIG/models/Qwen2.5-7B-Instruct}
MODEL_NAME=${GTA_MODEL_NAME:-qwen2.5-7b-instruct}
NGPU=$(( $(echo "$GTA_LLM_GPUS" | tr -cd ',' | wc -c) + 1 ))

exec env CUDA_VISIBLE_DEVICES=$GTA_LLM_GPUS lmdeploy serve api_server "$MODEL_PATH" \
    --server-port "$GTA_LLM_PORT" \
    --model-name "$MODEL_NAME" \
    --tp "$NGPU" \
    --cache-max-entry-count "${GTA_KV_FRAC:-0.5}"
