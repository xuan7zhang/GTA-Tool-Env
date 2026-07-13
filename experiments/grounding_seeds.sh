#!/bin/bash
set -uo pipefail
source /project/6101776/xzhan576/gta2-envlab/scripts/common_env.sh
conda activate $GTA_BIG/envs/opencompass
RUN=/datasets/omni_pretraining/gta2/ocrun; PROXY=http://127.0.0.1:16281
SEL=$GTA_BIG/results/inject_opt/selectors.json
PHI=$GTA_BIG/results/inject_opt/subtle_phi.json
FULL=$GTA_BIG/data/gta_dataset/toolmeta.json
export GTA_MODEL_NAME=qwen2.5-32b-instruct GTA_TOOLSERVER=$PROXY GTA_EVAL_MODES=end GTA_TOOLMETA=$FULL GTA_EXTRA_TOOLS=""
cd $RUN
KEEP=$(python3 -c "import json;print(json.dumps(json.load(open('$SEL'))['ours_grounding']['keep'],separators=(',',':')))")
doeval(){ local RID=$1 PH=$2
  [ -e $GTA_BIG/results/$RID/DONE ] && { echo "skip $RID"; return; }
  python3 - "$PROXY" "$KEEP" "$PH" "$GTA_BIG/results/$RID/proxy_calls.jsonl" "$RID" <<PY
import json,sys,urllib.request,os
proxy,keep,phi,log,rid=sys.argv[1:6]; os.makedirs(os.path.dirname(log),exist_ok=True)
body=json.dumps({"mode":"passthrough","mask":json.loads(keep),"per_tool_modes":{},
 "phi_toolmeta":(None if phi=="null" else phi),"unavailable_tools":["GoogleSearch","MathOCR"],
 "log_path":log,"run_meta":{"run_id":rid}}).encode()
urllib.request.urlopen(urllib.request.Request(proxy+"/proxy_config",data=body,method="POST",headers={"Content-Type":"application/json"}),timeout=30).read()
PY
  local WD=$GTA_BIG/results/$RID; mkdir -p $WD
  python $GTA_REPO/opencompass/run.py configs/gta_atomic_env.py --max-num-workers 1 --debug -w $WD 2>&1 | grep -vE "Future|warn|Resource|TRANSFORMERS" | tail -1
  ls $WD/*/results/*/gta_bench_end.json >/dev/null 2>&1 && { touch $WD/DONE; echo "[gs] $RID acc=$(python3 -c "import json,glob;print(round(json.load(open(sorted(glob.glob('$WD/*/results/*/gta_bench_end.json'))[-1]))['answer_acc'],2))")"; } || echo "[gs] $RID FAILED"
}
doeval ours_grounding_attractive_s2 null
doeval ours_grounding_attractive_s3 null
doeval ours_grounding_subtle_s2 $PHI
doeval ours_grounding_subtle_s3 $PHI
echo "[gs] complete"
