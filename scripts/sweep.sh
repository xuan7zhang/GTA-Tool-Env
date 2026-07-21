#!/bin/bash
# Manifest-driven sweep runner (one lane; run a second instance with lane env
# overrides + a manifest shard for parallel sweeps).
#
# Usage: sweep.sh <manifest.jsonl>
# Each manifest line:
# {"run_id":"loo_OCR_pass_s0", "probe_mode":"passthrough", "seed":0,
#  "toolmeta":"/abs/path/toolmeta.json"|null, "mask":["OCR",...]|null,
#  "hide_tools":["Calculator",...]|null,
#  "extra_tools":""|"LocateThenDescribe,...", "eval_modes":"end",
#  "p_fail":0.1}
#
# Assumes LLM + tool server + proxy for this lane are already up
# (start_llm.sh / start_toolserver.sh / start_proxy.sh). Mask + probe mode are
# switched per run through the proxy's /proxy_config endpoint — no tool-server
# restart needed for mask/phi variants. Compose variants (extra tools file)
# need the tool server started with GTA_EXTRA (documented in RUNBOOK).
set -euo pipefail
source "$(dirname "$0")/common_env.sh"

MANIFEST=${1:?usage: sweep.sh <manifest.jsonl>}
PROXY=http://127.0.0.1:$GTA_PROXY_PORT

while IFS= read -r line; do
  [ -z "$line" ] && continue
  eval "$(python3 - "$line" <<'EOF'
import json, shlex, sys
r = json.loads(sys.argv[1])
def sh(k, v): print(f'{k}={shlex.quote(str(v if v is not None else ""))}')
sh('RUN_ID', r['run_id']); sh('MODE', r.get('probe_mode', 'passthrough'))
sh('SEED', r.get('seed', 0)); sh('TOOLMETA', r.get('toolmeta') or '')
sh('MASK_JSON', json.dumps(r.get('mask'))); sh('EXTRA', r.get('extra_tools') or '')
sh('HIDE', ','.join(r.get('hide_tools') or []))
sh('PROXY_EXTRAS_JSON', json.dumps({
    k: r[k] for k in ('per_tool_modes', 'phi_toolmeta', 'unavailable_tools')
    if k in r
}))
sh('EVAL_MODES', r.get('eval_modes', 'end')); sh('P_FAIL', r.get('p_fail', 0.1))
EOF
)"
  RUN_DIR=$GTA_BIG/results/$RUN_ID
  if [ -e "$RUN_DIR/DONE" ]; then echo "[sweep] skip $RUN_ID (done)"; continue; fi
  mkdir -p "$RUN_DIR"
  echo "$line" > "$RUN_DIR/manifest_row.json"
  echo "[sweep] === $RUN_ID (mode=$MODE seed=$SEED) ==="

  # configure proxy for this run
  python3 - "$PROXY" "$MODE" "$SEED" "$P_FAIL" "$MASK_JSON" "$RUN_DIR" "$RUN_ID" "$PROXY_EXTRAS_JSON" <<'EOF'
import json, sys, urllib.request
proxy, mode, seed, p_fail, mask_json, run_dir, run_id, extras_json = sys.argv[1:9]
body = json.dumps({
    "mode": mode, "seed": int(seed), "p_fail": float(p_fail),
    "mask": json.loads(mask_json),
    "log_path": f"{run_dir}/proxy_calls.jsonl",
    "run_meta": {"run_id": run_id},
} | json.loads(extras_json)).encode()
req = urllib.request.Request(proxy + "/proxy_config", data=body, method="POST",
                             headers={"Content-Type": "application/json"})
print(urllib.request.urlopen(req, timeout=30).read().decode()[:200])
EOF

  export GTA_EVAL_MODES=$EVAL_MODES
  export GTA_SEED=$SEED
  export GTA_TOOLSERVER=$PROXY
  [ -n "$TOOLMETA" ] && export GTA_TOOLMETA=$TOOLMETA || unset GTA_TOOLMETA
  [ -n "$EXTRA" ] && export GTA_EXTRA_TOOLS=$EXTRA || unset GTA_EXTRA_TOOLS
  [ -n "$HIDE" ] && export GTA_HIDE_TOOLS=$HIDE || unset GTA_HIDE_TOOLS

  if bash "$(dirname "$0")/run_eval.sh" "$RUN_ID"; then
    gzip -f "$RUN_DIR/proxy_calls.jsonl" 2>/dev/null || true
    touch "$RUN_DIR/DONE"
  else
    echo "[sweep] RUN FAILED: $RUN_ID (continuing)"
  fi
done < "$MANIFEST"
echo "[sweep] manifest complete"
