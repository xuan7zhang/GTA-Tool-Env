#!/bin/bash
# C-axis composition: baseline (ImageDescription + OCR as 2 primitives) vs
# composed (PerceiveAll macro replaces them). 7B, full 229; analysis on the 42
# tasks needing both perception tools. Tests whether composing 2 tools into 1
# macro beats using them separately (esp. for a weak model that orchestrates poorly).
set -uo pipefail
source /project/6101776/xzhan576/gta2-envlab/scripts/common_env.sh; conda activate $GTA_BIG/envs/opencompass
RUN=/datasets/omni_pretraining/gta2/ocrun; PROXY=http://127.0.0.1:16281
FULL=$GTA_BIG/data/gta_dataset/toolmeta.json
SEL=$GTA_BIG/results/inject_opt/selectors_compose.json
export GTA_MODEL_NAME=qwen2.5-7b-instruct GTA_LLM_URL=http://127.0.0.1:12580/v1/chat/completions \
       GTA_TOOLSERVER=$PROXY GTA_EVAL_MODES=end GTA_TOOLMETA=$FULL
cd $RUN
doeval(){ local COND=$1 RID=$2
  [ -e $GTA_BIG/results/$RID/DONE ] && { echo "skip $RID"; return; }
  local KEEP=$(python3 -c "import json;print(json.dumps(json.load(open('$SEL'))['$COND']['keep'],separators=(',',':')))")
  local EXTRA=$(python3 -c "import json;print(json.load(open('$SEL'))['$COND']['extra'])")
  python3 - "$PROXY" "$KEEP" "$GTA_BIG/results/$RID/proxy_calls.jsonl" "$RID" <<PY
import json,sys,urllib.request,os
proxy,keep,log,rid=sys.argv[1:5]; os.makedirs(os.path.dirname(log),exist_ok=True)
body=json.dumps({"mode":"passthrough","mask":json.loads(keep),"per_tool_modes":{},"phi_toolmeta":None,
 "unavailable_tools":["GoogleSearch","MathOCR"],"log_path":log,"run_meta":{"run_id":rid}}).encode()
urllib.request.urlopen(urllib.request.Request(proxy+"/proxy_config",data=body,method="POST",headers={"Content-Type":"application/json"}),timeout=30).read()
PY
  local WD=$GTA_BIG/results/$RID; mkdir -p $WD
  GTA_EXTRA_TOOLS="$EXTRA" python $GTA_REPO/opencompass/run.py configs/gta_atomic_env.py --max-num-workers 1 --debug -w $WD 2>&1 | grep -vE "Future|warn|Resource|TRANSFORMERS" | tail -1
  ls $WD/*/results/*/gta_bench_end.json >/dev/null 2>&1 && { touch $WD/DONE; echo "[compose] $RID acc=$(python3 -c "import json,glob;print(round(json.load(open(sorted(glob.glob('$WD/*/results/*/gta_bench_end.json'))[-1]))['answer_acc'],2))")"; } || echo "[compose] $RID FAILED"
}
for s in 1 2 3; do
  doeval baseline_prim  compose_baseline_s$s
  doeval composed_macro compose_macro_s$s
done
echo "[compose] complete"
