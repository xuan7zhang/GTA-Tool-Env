#!/bin/bash
# Restart the LLM server if it dies. lmdeploy/turbomind has segfaulted in
# Sampling::Update() under long TACO lanes; without supervision that silently
# turns every remaining condition into a connection-error run.
source "$(dirname "$0")/common_env.sh"
n=0
while true; do
  n=$((n+1))
  echo "[llm-supervisor] start #$n $(date -Is) port=$GTA_LLM_PORT gpus=$GTA_LLM_GPUS"
  bash "$(dirname "$0")/start_llm.sh" || true
  echo "[llm-supervisor] EXITED (attempt $n) $(date -Is) — restarting in 10s"
  sleep 10
done
