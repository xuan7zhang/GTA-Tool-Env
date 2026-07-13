#!/bin/bash
# call_frequency multi-seed, regime-correct selector: attractive uses the dose8
# selector (selectors.json), subtle uses the subtle-trajectory selector.
set -uo pipefail
source /project/6101776/xzhan576/gta2-envlab/scripts/common_env.sh
conda activate $GTA_BIG/envs/opencompass
RUN=/datasets/omni_pretraining/gta2/ocrun; PROXY=http://127.0.0.1:16281
PHI=$GTA_BIG/results/inject_opt/subtle_phi.json; FULL=$GTA_BIG/data/gta_dataset/toolmeta.json
POISON=$(python3 -c "import json;print(','.join(json.load(open('$GTA_LAB/envgen/variants/poison/names.json'))))")
export GTA_MODEL_NAME=qwen2.5-32b-instruct GTA_TOOLSERVER=$PROXY GTA_EVAL_MODES=end GTA_TOOLMETA=$FULL
cd $RUN
doeval(){ local RID=$1 SEL=$2 PH=$3
  [ -e $GTA_BIG/results/$RID/DONE ] && { echo "skip $RID"; return; }
  local KEEP=$(python3 -c "import json;print(json.dumps(json.load(open('$SEL'))['call_frequency']['keep'],separators=(',',':')))")
  local EXTRA=$(python3 -c "import json;k=json.load(open('$SEL'))['call_frequency']['keep'];p=set('$POISON'.split(','));print(','.join([t for t in k if t in p]))")
  python3 - "$PROXY" "$KEEP" "$PH" "$GTA_BIG/results/$RID/proxy_calls.jsonl" "$RID" <<PY
import json,sys,urllib.request,os
proxy,keep,phi,log,rid=sys.argv[1:6]; os.makedirs(os.path.dirname(log),exist_ok=True)
body=json.dumps({"mode":"passthrough","mask":json.loads(keep),"per_tool_modes":{},
 "phi_toolmeta":(None if phi=="null" else phi),"unavailable_tools":["GoogleSearch","MathOCR"],
 "log_path":log,"run_meta":{"run_id":rid}}).encode()
urllib.request.urlopen(urllib.request.Request(proxy+"/proxy_config",data=body,method="POST",headers={"Content-Type":"application/json"}),timeout=30).read()
PY
  local WD=$GTA_BIG/results/$RID; mkdir -p $WD
  GTA_EXTRA_TOOLS="$EXTRA" python $GTA_REPO/opencompass/run.py configs/gta_atomic_env.py --max-num-workers 1 --debug -w $WD 2>&1 | grep -vE "Future|warn|Resource|TRANSFORMERS" | tail -1
  ls $WD/*/results/*/gta_bench_end.json >/dev/null 2>&1 && { touch $WD/DONE; echo "[cf] $RID acc=$(python3 -c "import json,glob;print(round(json.load(open(sorted(glob.glob('$WD/*/results/*/gta_bench_end.json'))[-1]))['answer_acc'],2))")"; } || echo "[cf] $RID FAILED"
}
SELA=$GTA_BIG/results/inject_opt/selectors.json
SELS=$GTA_BIG/results/inject_opt/selectors_subtle.json
for s in 1 2 3; do
  doeval cf_attractive_s$s $SELA null
  doeval cf_subtle_s$s $SELS $PHI
done
echo "[cf] complete"
