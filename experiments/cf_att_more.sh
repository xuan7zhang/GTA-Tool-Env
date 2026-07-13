#!/bin/bash
set -uo pipefail
source /project/6101776/xzhan576/gta2-envlab/scripts/common_env.sh; conda activate $GTA_BIG/envs/opencompass
RUN=/datasets/omni_pretraining/gta2/ocrun; PROXY=http://127.0.0.1:16281
FULL=$GTA_BIG/data/gta_dataset/toolmeta.json
POISON=$(python3 -c "import json;print(','.join(json.load(open('$GTA_LAB/envgen/variants/poison/names.json'))))")
SEL=$GTA_BIG/results/inject_opt/selectors.json
export GTA_MODEL_NAME=qwen2.5-32b-instruct GTA_TOOLSERVER=$PROXY GTA_EVAL_MODES=end GTA_TOOLMETA=$FULL
cd $RUN
KEEP=$(python3 -c "import json;print(json.dumps(json.load(open('$SEL'))['call_frequency']['keep'],separators=(',',':')))")
EXTRA=$(python3 -c "import json;k=json.load(open('$SEL'))['call_frequency']['keep'];p=set('$POISON'.split(','));print(','.join([t for t in k if t in p]))")
for s in 4 5 6; do
  RID=cf_attractive_s$s; WD=$GTA_BIG/results/$RID
  [ -e $WD/DONE ] && { echo "skip $RID"; continue; }
  python3 - "$PROXY" "$KEEP" "$WD/proxy_calls.jsonl" "$RID" <<PY
import json,sys,urllib.request,os
proxy,keep,log,rid=sys.argv[1:5]; os.makedirs(os.path.dirname(log),exist_ok=True)
body=json.dumps({"mode":"passthrough","mask":json.loads(keep),"per_tool_modes":{},"phi_toolmeta":None,
 "unavailable_tools":["GoogleSearch","MathOCR"],"log_path":log,"run_meta":{"run_id":rid}}).encode()
urllib.request.urlopen(urllib.request.Request(proxy+"/proxy_config",data=body,method="POST",headers={"Content-Type":"application/json"}),timeout=30).read()
PY
  mkdir -p $WD
  GTA_EXTRA_TOOLS="$EXTRA" python $GTA_REPO/opencompass/run.py configs/gta_atomic_env.py --max-num-workers 1 --debug -w $WD 2>&1 | grep -vE "Future|warn|Resource|TRANSFORMERS" | tail -1
  ls $WD/*/results/*/gta_bench_end.json >/dev/null 2>&1 && { touch $WD/DONE; echo "[cfm] $RID acc=$(python3 -c "import json,glob;print(round(json.load(open(sorted(glob.glob('$WD/*/results/*/gta_bench_end.json'))[-1]))['answer_acc'],2))")"; } || echo "[cfm] $RID FAILED"
done
echo "[cfm] complete"
