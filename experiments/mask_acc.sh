#!/bin/bash
# Clean mask-accuracy experiment (no poison, no phi): does pruning the natural
# pool beat the full pool on AnsAcc? 32B, N seeds each of full14 vs lean9.
set -uo pipefail
source /project/6101776/xzhan576/gta2-envlab/scripts/common_env.sh; conda activate $GTA_BIG/envs/opencompass
RUN=/datasets/omni_pretraining/gta2/ocrun; PROXY=http://127.0.0.1:16281
FULL=$GTA_BIG/data/gta_dataset/toolmeta.json
SEL=$GTA_BIG/results/inject_opt/selectors_mask.json
export GTA_MODEL_NAME=qwen2.5-32b-instruct GTA_TOOLSERVER=$PROXY GTA_EVAL_MODES=end GTA_TOOLMETA=$FULL
cd $RUN
NSEED=${1:-3}
doeval(){ local MK=$1 RID=$2
  [ -e $GTA_BIG/results/$RID/DONE ] && { echo "skip $RID"; return; }
  local KEEP=$(python3 -c "import json;print(json.dumps(json.load(open('$SEL'))['$MK']['keep'],separators=(',',':')))")
  python3 - "$PROXY" "$KEEP" "$GTA_BIG/results/$RID/proxy_calls.jsonl" "$RID" <<PY
import json,sys,urllib.request,os
proxy,keep,log,rid=sys.argv[1:5]; os.makedirs(os.path.dirname(log),exist_ok=True)
body=json.dumps({"mode":"passthrough","mask":json.loads(keep),"per_tool_modes":{},"phi_toolmeta":None,
 "unavailable_tools":["GoogleSearch","MathOCR"],"log_path":log,"run_meta":{"run_id":rid}}).encode()
urllib.request.urlopen(urllib.request.Request(proxy+"/proxy_config",data=body,method="POST",headers={"Content-Type":"application/json"}),timeout=30).read()
PY
  local WD=$GTA_BIG/results/$RID; mkdir -p $WD
  GTA_EXTRA_TOOLS="" python $GTA_REPO/opencompass/run.py configs/gta_atomic_env.py --max-num-workers 1 --debug -w $WD 2>&1 | grep -vE "Future|warn|Resource|TRANSFORMERS" | tail -1
  ls $WD/*/results/*/gta_bench_end.json >/dev/null 2>&1 && { touch $WD/DONE; echo "[mask] $RID acc=$(python3 -c "import json,glob;print(round(json.load(open(sorted(glob.glob('$WD/*/results/*/gta_bench_end.json'))[-1]))['answer_acc'],2))")"; } || echo "[mask] $RID FAILED"
}
for s in $(seq 1 $NSEED); do
  doeval full14 maskacc_full14_s$s
  doeval lean9  maskacc_lean9_s$s
done
echo "[mask] complete"
