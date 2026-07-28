#!/bin/bash
# Start the AgentLego tool server (perception models on GPU).
# Run ON the GPU node. Env knobs: GTA_TOOL_GPU, GTA_TOOL_PORT,
#   GTA_TOOLLIST (default benchmark_toollist.txt = the 14 GTA-Atomic tools),
#   SERPER_API_KEY / MATHPIX_APP_ID / MATHPIX_APP_KEY (optional; when absent
#   the proxy routes GoogleSearch/MathOCR to 'unavailable').
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
conda activate $GTA_BIG/envs/agentlego

cd $GTA_REPO/agentlego
TOOLLIST=${GTA_TOOLLIST:-benchmark_toollist.txt}

# Injection study: register poison tools (CPU-only) alongside the real pool when
# GTA_POISON=1. Dose (how many the agent sees) is controlled by the proxy mask,
# so all 8 are always registered here; the mask hides the rest.
POISON_ARGS=""
if [ "${GTA_POISON:-0}" = "1" ]; then
    PDIR=$GTA_LAB/envgen/variants/poison
    POISON_NAMES=$(python3 -c "import json;print(' '.join(json.load(open('$PDIR/names.json'))))")
    POISON_ARGS="--extra $PDIR/poison_tools.py $POISON_NAMES"
    echo "[toolserver] POISON injected: $POISON_NAMES"
fi

# Composition macros (C-axis): register PerceiveAll etc. via --extra when GTA_MACRO=1.
MACRO_ARGS=""
if [ "${GTA_MACRO:-0}" = "1" ]; then
    MF=/datasets/omni_pretraining/gta2/scripts_extra/macro_tools.py
    MACRO_NAMES="${GTA_MACRO_NAMES:-PerceiveAll}"
    MACRO_ARGS="--extra $MF $MACRO_NAMES"
    echo "[toolserver] MACRO registered: $MACRO_NAMES"
fi

# Escape-hatch meta-tool (dynamic mask policy): register RequestTool when GTA_HATCH=1.
HATCH_ARGS=""
if [ "${GTA_HATCH:-0}" = "1" ]; then
    HATCH_ARGS="--extra /datasets/omni_pretraining/gta2/scripts_extra/hatch_tools.py RequestTool"
    echo "[toolserver] HATCH registered: RequestTool"
fi

# Optional experiment-local tools. Defaults empty, so historical launchers are unchanged.
CUSTOM_ARGS=""
if [ -n "${GTA_CUSTOM_TOOL_FILE:-}" ] && [ -n "${GTA_CUSTOM_TOOL_NAMES:-}" ]; then
    CUSTOM_ARGS="--extra $GTA_CUSTOM_TOOL_FILE $GTA_CUSTOM_TOOL_NAMES"
    echo "[toolserver] CUSTOM registered: $GTA_CUSTOM_TOOL_NAMES from $GTA_CUSTOM_TOOL_FILE"
fi

# --no-external-api mode: agentlego refuses to construct GoogleSearch/MathOCR
# without keys. Register them with placeholder keys; the proxy routes them to
# the 'unavailable' response (never forwards), and logs which tasks touch them.
if [ -z "${SERPER_API_KEY:-}" ]; then
    export SERPER_API_KEY=MISSING_KEY_NO_EXTERNAL_API
    echo "[toolserver] WARN: no SERPER_API_KEY — GoogleSearch registered but proxy-unavailable"
fi
if [ -z "${MATHPIX_APP_ID:-}" ]; then
    export MATHPIX_APP_ID=MISSING_KEY_NO_EXTERNAL_API MATHPIX_APP_KEY=MISSING_KEY_NO_EXTERNAL_API
    echo "[toolserver] WARN: no MATHPIX keys — MathOCR registered but proxy-unavailable"
fi

exec env CUDA_VISIBLE_DEVICES=$GTA_TOOL_GPU agentlego-server start \
    --port "$GTA_TOOL_PORT" --host 0.0.0.0 \
    --extra ./benchmark.py $POISON_ARGS $MACRO_ARGS $HATCH_ARGS $CUSTOM_ARGS $(cat "$TOOLLIST")
