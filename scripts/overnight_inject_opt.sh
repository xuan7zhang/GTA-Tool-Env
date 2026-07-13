#!/bin/bash
# Overnight injection->optimization->beat-baselines experiment (single lane).
# Operating point: dose8 (14 real + 8 poison tools). Runs on the kn066/l2 32B
# stack (proxy :16281 -> poison tool server :16182). Resumable via DONE markers.
#
# Phases:
#   1. selectors (non-ours) from the dose8 degraded trajectory  [fast, cpu]
#   2. clean-subset + per-tool corruption probes (subset)        [~3h]
#   3. ours selector from attribution                            [fast]
#   4. evaluate each method's pruned env (full 229)              [~5h]
#   5. bootstrap comparison + report                             [fast]
set -uo pipefail
source "$(dirname "$0")/common_env.sh"
LAB=$GTA_LAB; BIG=$GTA_BIG
PROXY=http://127.0.0.1:${GTA_PROXY_PORT:-16281}
WORK=$BIG/results/inject_opt; mkdir -p $WORK
FULL=$BIG/data/gta_dataset/toolmeta.json
POISON=$(python3 -c "import json;print(','.join(json.load(open('$LAB/envgen/variants/poison/names.json'))))")
BASE=$(python3 -c "import json;print(','.join(json.load(open('$FULL'))))")
POOL="$BASE,$POISON"                       # 22 tools at dose8
DEGRADED_RUN=$BIG/results/inject_dose8_32b
SUBSET=${GTA_PROBE_SUBSET:-60}
export GTA_MODEL_NAME=qwen2.5-32b-instruct GTA_TOOLSERVER=$PROXY

pcfg() { curl -s -m 8 -X POST $PROXY/proxy_config -H 'Content-Type: application/json' -d "$1" >/dev/null; }
log(){ echo "[overnight $(date +%H:%M)] $*"; }

# ---- Phase 0: wait for the degradation curve to release the lane ----
log "phase0 waiting for degradation curve (inject_dose0_32b) to finish"
while [ ! -e $BIG/results/inject_dose0_32b/DONE ]; do sleep 120; done
log "phase0 curve done; lane free"

# ---- Phase 1: non-ours selectors (need only degraded trajectory) ----
if [ ! -e $WORK/selectors_base.DONE ]; then
  log "phase1 selectors (keep_all/oracle/random/call_frequency/error_rate)"
  python3 $LAB/analysis/tool_selectors.py --degraded-run $DEGRADED_RUN \
    --pool "$POOL" --injected "$POISON" --k 8 --seed 0 \
    --out $WORK/selectors.json && touch $WORK/selectors_base.DONE
fi

# ---- Phase 2: clean-subset + per-tool corruption probes ----
# probe tools called >=5 times in the degraded run
CALLED=$(python3 - "$DEGRADED_RUN" "$SUBSET" <<'EOF'
import json,glob,sys
from collections import Counter
run=sys.argv[1]
f=sorted(glob.glob(f'{run}/*/predictions/*/gta_bench_end.json'))
c=Counter()
if f:
  for k,v in json.load(open(f[-1])).items():
    if not k.isdigit():continue
    pred=v.get('prediction') or []; rounds=pred if (pred and isinstance(pred[0],list)) else [pred]
    for s in (rounds[0] or []):
      if isinstance(s,dict) and 'tool_calls' in s: c[s['tool_calls'][0]['function']['name']]+=1
print(','.join(t for t,n in c.items() if n>=5))
EOF
)
log "probe tools (called>=5): $CALLED"

# clean subset baseline (dose8 env, no corruption)
if [ ! -e $WORK/probe_clean.DONE ]; then
  log "phase2 clean-subset baseline (n=$SUBSET)"
  pcfg "{\"mode\":\"passthrough\",\"mask\":$(python3 -c "import json;print(json.dumps('$POOL'.split(',')))"),\"per_tool_modes\":{},\"unavailable_tools\":[\"GoogleSearch\",\"MathOCR\"],\"log_path\":\"$WORK/probe_clean.jsonl\"}"
  GTA_EVAL_MODES=end GTA_TASK_SUBSET=$SUBSET GTA_TOOLMETA=$FULL GTA_EXTRA_TOOLS=$POISON \
    bash $LAB/scripts/run_eval.sh probe_clean_subset >/dev/null 2>&1
  touch $WORK/probe_clean.DONE
fi

# per-tool corrupt probes
IFS=',' read -ra TA <<< "$CALLED"
for tool in "${TA[@]}"; do
  [ -z "$tool" ] && continue
  if [ -e $WORK/probe_${tool}.DONE ]; then continue; fi
  log "phase2 corrupt-probe $tool"
  pcfg "{\"mode\":\"passthrough\",\"mask\":$(python3 -c "import json;print(json.dumps('$POOL'.split(',')))"),\"per_tool_modes\":{\"$tool\":\"corrupt_output\"},\"seed\":0,\"unavailable_tools\":[\"GoogleSearch\",\"MathOCR\"],\"log_path\":\"$WORK/probe_${tool}.jsonl\"}"
  GTA_EVAL_MODES=end GTA_TASK_SUBSET=$SUBSET GTA_TOOLMETA=$FULL GTA_EXTRA_TOOLS=$POISON \
    bash $LAB/scripts/run_eval.sh probe_${tool} >/dev/null 2>&1
  touch $WORK/probe_${tool}.DONE
done
pcfg "{\"per_tool_modes\":{}}"

# ---- Phase 3: attribution + ours selector ----
if [ ! -e $WORK/selectors_ours.DONE ]; then
  log "phase3 corruption attribution -> ours selector"
  python3 $LAB/analysis/compute_attribution.py --work $WORK --pool "$POOL" \
    --injected "$POISON" --k 8 --degraded-run $DEGRADED_RUN \
    --out-selectors $WORK/selectors.json && touch $WORK/selectors_ours.DONE
fi

# ---- Phase 4: evaluate each method's pruned env (full 229) ----
log "phase4 method evals"
python3 - "$WORK/selectors.json" "$POISON" "$FULL" "$WORK/method_manifest.jsonl" <<'EOF'
import json,sys
sel=json.load(open(sys.argv[1])); poison=set(sys.argv[2].split(',')); full=sys.argv[3]
rows=[]
for m,v in sel.items():
    keep=v['keep']; kept_poison=[t for t in keep if t in poison]
    rows.append({"run_id":f"inopt_{m}","probe_mode":"passthrough","seed":0,
                 "toolmeta":full,"mask":keep,"extra_tools":",".join(kept_poison),"eval_modes":"end"})
open(sys.argv[4],'w').write('\n'.join(json.dumps(r) for r in rows)+'\n')
print("methods:",[r['run_id'] for r in rows])
EOF
bash $LAB/scripts/sweep.sh $WORK/method_manifest.jsonl

# ---- Phase 5: bootstrap + report ----
log "phase5 bootstrap + report"
conda activate $BIG/envs/opencompass
RUNS=""
for m in keep_all oracle random call_frequency error_rate ours_corruption; do
  d=$BIG/results/inopt_$m; [ -d "$d" ] && RUNS="$RUNS ${m}=$d"
done
python3 $LAB/analysis/bootstrap_env.py --dataset $BIG/data/gta_dataset/dataset.json \
  --sim-threshold 0.5 --runs $RUNS --out $LAB/analysis/inject_opt_bootstrap.md 2>/dev/null | tail -20
touch $WORK/ALL_DONE
log "OVERNIGHT COMPLETE"
