#!/bin/bash
# Corruption-attribution probes for the "ours" selector.
# For each candidate tool, run the eval with ONLY that tool's output corrupted
# (proxy per_tool_modes={tool: corrupt_output}, everything else passthrough) on a
# TASK SUBSET, and record AnsAcc. Attribution(tool) = clean_acc - corrupt_tool_acc.
# A poison tool's corruption barely moves accuracy (its output was already wrong)
# -> low attribution -> pruned by the method.
#
# Usage: run_corrupt_probes.sh <degraded_run_id> <tools_csv> <subset_n>
# Env: GTA_MODEL_NAME, GTA_PROXY_PORT, GTA_TASK_SUBSET (n tasks).
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
RUN_ID=${1:?degraded run_id}      # env whose openapi/mask defines the pool
TOOLS=${2:?comma tools to probe}
SUBSET=${3:-76}
PROXY=http://127.0.0.1:$GTA_PROXY_PORT
OUT=$GTA_BIG/results/probes_${RUN_ID}
mkdir -p "$OUT"

IFS=',' read -ra TARR <<< "$TOOLS"
for tool in "${TARR[@]}"; do
  if [ -e "$OUT/${tool}.DONE" ]; then echo "[probe] skip $tool"; continue; fi
  echo "[probe] corrupt-only $tool (subset $SUBSET)"
  # per-tool corrupt: passthrough globally, corrupt just this tool
  python3 - "$PROXY" "$tool" "$OUT/${tool}.jsonl" <<EOF
import json,urllib.request,sys
proxy,tool,log=sys.argv[1:4]
body=json.dumps({"mode":"passthrough","per_tool_modes":{tool:"corrupt_output"},
                 "seed":0,"log_path":log,"run_meta":{"probe":tool}}).encode()
urllib.request.urlopen(urllib.request.Request(proxy+"/proxy_config",data=body,
  method="POST",headers={"Content-Type":"application/json"}),timeout=30).read()
EOF
  GTA_EVAL_MODES=end GTA_TASK_SUBSET=$SUBSET GTA_TOOLSERVER=$PROXY \
    bash "$(dirname "$0")/run_eval.sh" "probe_${RUN_ID}_${tool}" >/dev/null 2>&1 || true
  touch "$OUT/${tool}.DONE"
done
# reset per_tool_modes
python3 - "$PROXY" <<EOF
import json,urllib.request,sys
urllib.request.urlopen(urllib.request.Request(sys.argv[1]+"/proxy_config",
  data=json.dumps({"per_tool_modes":{}}).encode(),method="POST",
  headers={"Content-Type":"application/json"}),timeout=30).read()
EOF
echo "[probe] all probes done -> $OUT"
