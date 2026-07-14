#!/bin/bash
# Capability curve + cross-family for RegionRead composition. Serves each model on
# kn080 GPU3 sequentially (port 12591), runs regionread_curve.sh against the existing
# RegionRead tool stack (proxy 16282 -> tool server 16182), then frees the GPU.
# 7B is already served (12580) and its 5-seed run (rr_*) is reused as that point.
# 32B needs tp=2 (2 free GPUs) — not run here.
set -uo pipefail
source /project/6101776/xzhan576/gta2-envlab/scripts/common_env.sh
LAB=/project/6101776/xzhan576/gta2-envlab
SE=/datasets/omni_pretraining/gta2/scripts_extra
LOGDIR=/datasets/omni_pretraining/gta2/results/rr_stack
PORT=12591
NSEED=${NSEED:-5}

# path | model_name | prefix | tp
MODELS=(
  "$GTA_BIG/models/Qwen2.5-3B-Instruct|qwen2.5-3b-instruct|q3b|1"
  "$GTA_BIG/models/Qwen2.5-14B-Instruct|qwen2.5-14b-instruct|q14b|1"
  "$GTA_BIG/models/Llama-3.2-3B-Instruct|llama-3.2-3b-instruct|l3b|1"
  "$GTA_BIG/models/Llama-3.1-8B-Instruct|llama-3.1-8b-instruct|l8b|1"
)

gpu3_free_mb(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 2>/dev/null | tr -d ' '; }

for entry in "${MODELS[@]}"; do
  IFS='|' read -r MPATH MNAME PFX TP <<< "$entry"
  echo "==================== $MNAME ($PFX) ===================="
  # serve on GPU3
  CUDA_VISIBLE_DEVICES=3 GTA_LLM_GPUS=3 GTA_LLM_PORT=$PORT GTA_MODEL_PATH="$MPATH" \
    GTA_MODEL_NAME="$MNAME" GTA_KV_FRAC=0.4 \
    bash "$LAB/scripts/start_llm.sh" > "$LOGDIR/llm_${PFX}.log" 2>&1 &
  LLM_PID=$!
  echo "[battery] serving $MNAME pid=$LLM_PID; waiting for readiness"
  ready=0
  for i in $(seq 1 60); do
    if curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 200; then
      ready=1; echo "[battery] $MNAME ready @ ${i}0s"; break
    fi
    kill -0 $LLM_PID 2>/dev/null || { echo "[battery] $MNAME DIED during load"; break; }
    sleep 10
  done
  if [ "$ready" = 1 ]; then
    bash "$SE/regionread_curve.sh" "$MNAME" "$PORT" "$PFX" "$NSEED"
  else
    echo "[battery] SKIP $MNAME (not ready)"; tail -5 "$LOGDIR/llm_${PFX}.log"
  fi
  # free GPU3 for the next model
  kill $LLM_PID 2>/dev/null || true
  echo "[battery] killed $MNAME; waiting for GPU3 to free"
  for i in $(seq 1 30); do
    mb=$(gpu3_free_mb); [ -z "$mb" ] && mb=0
    [ "$mb" -lt 2000 ] && { echo "[battery] GPU3 free (${mb}MB)"; break; }
    sleep 5
  done
done
echo "[battery] ALL DONE"
