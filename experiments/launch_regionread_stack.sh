#!/bin/bash
# Bring up a SECOND tool-server+proxy on free GPU1 with RegionRead registered,
# leaving the existing 16181/16281 stack untouched (no kills). The macro's
# internal calls route back through the NEW proxy (16282) via GTA_PROXY_URL_INTERNAL.
set -uo pipefail
LAB=/project/6101776/xzhan576/gta2-envlab
LOG=/datasets/omni_pretraining/gta2/results/rr_stack
mkdir -p "$LOG"

# 1) tool server #2 on GPU1:16182 with PerceiveAll + RegionRead macros
setsid nohup env PYTHONUNBUFFERED=1 \
  GTA_TOOL_GPU=1 GTA_TOOL_PORT=16182 \
  GTA_MACRO=1 GTA_MACRO_NAMES="PerceiveAll RegionRead" \
  GTA_PROXY_URL_INTERNAL=http://127.0.0.1:16282 \
  bash "$LAB/scripts/start_toolserver.sh" > "$LOG/toolserver.log" 2>&1 &
echo "[launch] tool server #2 -> GPU1:16182 (log $LOG/toolserver.log)"

# 2) proxy #2 on 16282 -> upstream 16182 (use agentlego env python: has httpx/fastapi)
sleep 3
PY=/datasets/omni_pretraining/gta2/envs/agentlego/bin/python
setsid nohup env PYTHONUNBUFFERED=1 "$PY" "$LAB/proxy/proxy.py" --port 16282 \
  --upstream http://127.0.0.1:16182 \
  --log "$LOG/proxy_calls.jsonl" > "$LOG/proxy.log" 2>&1 &
echo "[launch] proxy #2 -> 16282 (log $LOG/proxy.log)"
sleep 2
echo "[launch] done; tool server model load takes a few min"
