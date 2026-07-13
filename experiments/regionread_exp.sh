#!/bin/bash
# C-axis composition v2 (sequential-dependency): baseline (agent chains TextToBbox
# -> RegionAttributeDescription itself) vs composed (RegionRead macro does the chain
# in one call, hiding bbox-threading). Runs ONLY the 8 GTA tasks whose gt uses this
# chain (ids 9,10,11,40,84,164,165,166). 7B, N seeds. Uses the #2 stack (proxy 16282
# -> tool server 16182 with RegionRead). Macro internals call the tool server through
# proxy 16282, where the primitives stay in the mask; the agent's MENU hides them via
# GTA_HIDE_TOOLS so it must use RegionRead.
set -uo pipefail
source /project/6101776/xzhan576/gta2-envlab/scripts/common_env.sh; conda activate $GTA_BIG/envs/opencompass
RUN=/datasets/omni_pretraining/gta2/ocrun; PROXY=http://127.0.0.1:16282
FULL=$GTA_BIG/data/gta_dataset/toolmeta.json
SEL=$GTA_BIG/results/inject_opt/selectors_regionread.json
IDS="9,10,11,40,84,164,165,166"
NSEED=${NSEED:-5}
export GTA_MODEL_NAME=qwen2.5-7b-instruct GTA_LLM_URL=http://127.0.0.1:12580/v1/chat/completions \
       GTA_TOOLSERVER=$PROXY GTA_EVAL_MODES=end GTA_TOOLMETA=$FULL GTA_TASK_IDS=$IDS
cd $RUN

doeval(){ local COND=$1 RID=$2
  [ -e $GTA_BIG/results/$RID/DONE ] && { echo "skip $RID"; return; }
  local KEEP=$(python3 -c "import json;print(json.dumps(json.load(open('$SEL'))['$COND']['keep'],separators=(',',':')))")
  local EXTRA=$(python3 -c "import json;print(json.load(open('$SEL'))['$COND']['extra'])")
  local HIDE=$(python3 -c "import json;print(json.load(open('$SEL'))['$COND'].get('hide',''))")
  python3 - "$PROXY" "$KEEP" "$GTA_BIG/results/$RID/proxy_calls.jsonl" "$RID" <<PY
import json,sys,urllib.request,os
proxy,keep,log,rid=sys.argv[1:5]; os.makedirs(os.path.dirname(log),exist_ok=True)
body=json.dumps({"mode":"passthrough","mask":json.loads(keep),"per_tool_modes":{},"phi_toolmeta":None,
 "unavailable_tools":["GoogleSearch","MathOCR"],"log_path":log,"run_meta":{"run_id":rid}}).encode()
urllib.request.urlopen(urllib.request.Request(proxy+"/proxy_config",data=body,method="POST",headers={"Content-Type":"application/json"}),timeout=30).read()
PY
  local WD=$GTA_BIG/results/$RID; mkdir -p $WD
  GTA_EXTRA_TOOLS="$EXTRA" GTA_HIDE_TOOLS="$HIDE" \
    python $GTA_REPO/opencompass/run.py configs/gta_atomic_env.py --max-num-workers 1 --debug -w $WD 2>&1 | grep -vE "Future|warn|Resource|TRANSFORMERS" | tail -1
  if ls $WD/*/results/*/gta_bench_end.json >/dev/null 2>&1; then
    touch $WD/DONE
    echo "[rr] $RID acc=$(python3 -c "import json,glob;print(round(json.load(open(sorted(glob.glob('$WD/*/results/*/gta_bench_end.json'))[-1]))['answer_acc'],2))")"
  else echo "[rr] $RID FAILED"; fi
}

for s in $(seq 1 $NSEED); do
  doeval baseline_prim  rr_baseline_s$s
  doeval composed_rr    rr_composed_s$s
done
echo "[rr] complete"
