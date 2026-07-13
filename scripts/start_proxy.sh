#!/bin/bash
# Start the probe/noise proxy in front of the tool server (CPU only).
# Env knobs: GTA_PROXY_PORT, GTA_TOOL_PORT (upstream), GTA_PROXY_MODE,
#            GTA_PROXY_CONFIG (json), GTA_PROXY_LOG (jsonl path).
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
conda activate $GTA_BIG/envs/agentlego

LOG=${GTA_PROXY_LOG:-$GTA_BIG/results/proxy_calls.jsonl}
ARGS=(--port "$GTA_PROXY_PORT" --upstream "http://127.0.0.1:$GTA_TOOL_PORT" --log "$LOG")

# no-external-api mode: with keys absent, mark the two external tools
# unavailable at the proxy (they stay registered on the tool server).
if [ -z "${GTA_PROXY_CONFIG:-}" ]; then
    UNAVAIL=()
    [ -z "${SERPER_API_KEY:-}" ] || [ "${SERPER_API_KEY:-}" = "MISSING_KEY_NO_EXTERNAL_API" ] && UNAVAIL+=("GoogleSearch")
    [ -z "${MATHPIX_APP_ID:-}" ] || [ "${MATHPIX_APP_ID:-}" = "MISSING_KEY_NO_EXTERNAL_API" ] && UNAVAIL+=("MathOCR")
    if [ ${#UNAVAIL[@]} -gt 0 ]; then
        GTA_PROXY_CONFIG=$GTA_BIG/proxy_autoconfig.json
        python3 - "$GTA_PROXY_CONFIG" "${UNAVAIL[@]}" <<'EOF'
import json, sys
path, *tools = sys.argv[1:]
json.dump({"unavailable_tools": tools}, open(path, "w"))
print(f"[proxy] no-external-api: unavailable_tools={tools}")
EOF
    fi
fi
[ -n "${GTA_PROXY_CONFIG:-}" ] && ARGS+=(--config "$GTA_PROXY_CONFIG")
[ -n "${GTA_PROXY_MODE:-}" ] && ARGS+=(--mode "$GTA_PROXY_MODE")

exec python $GTA_LAB/proxy/proxy.py "${ARGS[@]}"
