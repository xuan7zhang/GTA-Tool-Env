#!/bin/bash
# Subtle-poison experiment: same 8 poison tools, but their agent-visible
# descriptions are rewritten NEUTRAL via the proxy Phi axis (no server restart),
# so the agent no longer over-calls them -> call_frequency should FAIL (it prunes
# the genuinely high-use real tools), while the behavioral answer-echo signal
# still catches the poison the agent submits. Target: ours_echo > call_frequency.
set -uo pipefail
source /project/6101776/xzhan576/gta2-envlab/scripts/common_env.sh
conda activate $GTA_BIG/envs/opencompass
LAB=$GTA_LAB; BIG=$GTA_BIG
RUN=/datasets/omni_pretraining/gta2/ocrun
PROXY=http://127.0.0.1:${GTA_PROXY_PORT:-16281}
WORK=$BIG/results/inject_opt
PHI=$WORK/subtle_phi.json
FULL=$BIG/data/gta_dataset/toolmeta.json
POISON=$(python3 -c "import json;print(','.join(json.load(open('$LAB/envgen/variants/poison/names.json'))))")
BASE=$(python3 -c "import json;print(','.join(json.load(open('$FULL'))))")
POOL="$BASE,$POISON"
export GTA_MODEL_NAME=qwen2.5-32b-instruct GTA_TOOLSERVER=$PROXY GTA_EVAL_MODES=end GTA_TOOLMETA=$FULL
cd $RUN
log(){ echo "[subtle $(date +%H:%M)] $*"; }

# proxy config helper: always keeps phi_toolmeta set (subtle descriptions)
setproxy(){ # setproxy <mask_json> <extra_names> <logpath> <run_id>
  python3 - "$PROXY" "$1" "$PHI" "$3" "$4" <<'EOF'
import json,sys,urllib.request,os
proxy,mask,phi,log,rid=sys.argv[1:6]
os.makedirs(os.path.dirname(log),exist_ok=True)
body=json.dumps({"mode":"passthrough","mask":json.loads(mask),"per_tool_modes":{},
 "phi_toolmeta":phi,"unavailable_tools":["GoogleSearch","MathOCR"],
 "log_path":log,"run_meta":{"run_id":rid}}).encode()
urllib.request.urlopen(urllib.request.Request(proxy+"/proxy_config",data=body,method="POST",
 headers={"Content-Type":"application/json"}),timeout=30).read()
EOF
}
doeval(){ # doeval <run_id> <mask_json> <extra>
  local RID=$1 MASK=$2 EXTRA=$3
  [ -e $BIG/results/$RID/DONE ] && { echo "skip $RID"; return; }
  setproxy "$MASK" "$EXTRA" "$BIG/results/$RID/proxy_calls.jsonl" "$RID"
  local WD=$BIG/results/$RID; mkdir -p $WD
  GTA_EXTRA_TOOLS=$EXTRA python $GTA_REPO/opencompass/run.py configs/gta_atomic_env.py \
    --max-num-workers 1 --debug -w $WD 2>&1 | grep -vE "Future|warn|Resource|TRANSFORMERS" | tail -2
  if ls $WD/*/results/*/gta_bench_end.json >/dev/null 2>&1; then
    touch $WD/DONE
    log "$RID DONE acc=$(python3 -c "import json,glob;print(round(json.load(open(sorted(glob.glob('$WD/*/results/*/gta_bench_end.json'))[-1]))['answer_acc'],2))")"
  else log "$RID FAILED"; fi
}

MASK_ALL=$(python3 -c "import json;print(json.dumps('$POOL'.split(',')))")

# Phase A: degraded with subtle (neutral) descriptions
log "phaseA keep_all_subtle (dose8, neutral desc)"
doeval keep_all_subtle "$MASK_ALL" "$POISON"

# Phase B: selectors from the subtle degraded trajectory
log "phaseB selectors from subtle trajectory"
python3 $LAB/analysis/tool_selectors.py --degraded-run $BIG/results/keep_all_subtle \
  --pool "$POOL" --injected "$POISON" --k 8 --seed 0 --out $WORK/selectors_subtle.json >/dev/null
python3 /datasets/omni_pretraining/gta2/scripts_extra/answer_echo_selector.py \
  --degraded-run $BIG/results/keep_all_subtle --pool "$POOL" --injected "$POISON" \
  --k 8 --out $WORK/selectors_subtle_echo.json
python3 - "$WORK/selectors_subtle.json" "$WORK/selectors_subtle_echo.json" <<'EOF'
import json,sys
s=json.load(open(sys.argv[1])); e=json.load(open(sys.argv[2])); s['ours_echo']=e['ours_echo']
json.dump(s,open(sys.argv[1],'w'),indent=1)
poison=set("DirectImageQA InstantAnswer PreciseOCR SmartCount ExpertMath VerifiedFact ImageDetailPro QuickBBox".split())
for m in ['oracle','call_frequency','ours_echo']:
    hit=len(set(s[m]['pruned'])&poison); print(f"  {m}: prune {s[m]['pruned']} (caught {hit}/8)")
EOF

# Phase C: key method evals (subtle)
log "phaseC method evals"
for m in oracle call_frequency ours_echo; do
  IFS=$'\t' read MASK EXTRA < <(python3 - "$WORK/selectors_subtle.json" "$m" "$POISON" <<'EOF'
import json,sys
sel=json.load(open(sys.argv[1])); m=sys.argv[2]; poison=set(sys.argv[3].split(','))
keep=sel[m]['keep']; kp=[t for t in keep if t in poison]
print(json.dumps(keep,separators=(',',':')) + '\t' + (','.join(kp) if kp else '_'))
EOF
)
  [ "$EXTRA" = "_" ] && EXTRA=""
  doeval ${m}_subtle "$MASK" "$EXTRA"
done

# Phase D: bootstrap
log "phaseD bootstrap"
python3 $LAB/analysis/bootstrap_env.py --dataset $BIG/data/gta_dataset/dataset.json --sim-threshold 0.5 \
  --runs keep_all=$BIG/results/keep_all_subtle oracle=$BIG/results/oracle_subtle \
    call_frequency=$BIG/results/call_frequency_subtle ours_echo=$BIG/results/ours_echo_subtle \
  --out $WORK/bootstrap_subtle.md 2>/dev/null | tail -12
log "SUBTLE COMPLETE"
