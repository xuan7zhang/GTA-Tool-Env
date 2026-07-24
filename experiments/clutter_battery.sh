#!/bin/bash
# Clutter-structure battery (CLEAN pool, no injection). 7B, greedy, full 229.
# Each config = per-task mask via GTA_PREDICTED_MASK (base menu forced to 14, then
# restricted to the config's per-task keep-set = GT + specified distractors).
set -uo pipefail
source /project/6101776/xzhan576/gta2-envlab/scripts/common_env.sh; conda activate $GTA_BIG/envs/opencompass
RUN=/datasets/omni_pretraining/gta2/ocrun; PROXY=http://127.0.0.1:16281
FULL=$GTA_BIG/data/gta_dataset/toolmeta.json; MDIR=$GTA_BIG/results/inject_opt/clutter_masks
NAT=$(python3 -c "import json;print(','.join(json.load(open('$FULL'))))")
export GTA_MODEL_NAME=qwen2.5-7b-instruct GTA_LLM_URL=http://127.0.0.1:12581/v1/chat/completions \
       GTA_TOOLSERVER=$PROXY GTA_EVAL_MODES=end GTA_TOOLMETA=$FULL GTA_TEMP=0 GTA_EXTRA_TOOLS=$NAT
cd $RUN
python3 -c "import json,urllib.request;urllib.request.urlopen(urllib.request.Request('$PROXY/proxy_config',data=json.dumps({'mode':'passthrough','mask':'$NAT'.split(','),'per_tool_modes':{},'phi_toolmeta':None,'unavailable_tools':['GoogleSearch','MathOCR']}).encode(),method='POST',headers={'Content-Type':'application/json'}),timeout=30).read()"
doeval(){ local CFG=$1; local RID=cl_$CFG
  [ -e $GTA_BIG/results/$RID/DONE ] && { echo "skip $RID"; return; }
  local WD=$GTA_BIG/results/$RID; mkdir -p $WD
  GTA_PREDICTED_MASK=$MDIR/$CFG.json python $GTA_REPO/opencompass/run.py configs/gta_atomic_env.py --max-num-workers 1 --debug -w $WD 2>&1 | grep -vE "Future|warn|Resource|TRANSFORMERS" | tail -1
  local f=$(ls $WD/*/results/*/gta_bench_end.json 2>/dev/null|tail -1)
  [ -n "$f" ] && { touch $WD/DONE; echo "[cl] $RID acc=$(python3 -c "import json;print(round(json.load(open('$f'))['answer_acc'],2))")"; } || echo "[cl] $RID FAILED"
}
for C in oracle dose2_s1 dose2_s2 dose4_s1 dose4_s2 dose6_s1 dose6_s2 dose8_s1 dose8_s2 full \
         cat_perception cat_logic cat_operation cat_creativity; do doeval $C; done
echo "[cl] complete"
