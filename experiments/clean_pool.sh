#!/bin/bash
# Clean-pool control (NO poison injected): does Ground+'s declutter help or harm when
# there is nothing to defend against? Ground itself is a provable no-op here (0/14
# genuine tools are probe-INVARIANT, so its keep-set == keep_all), so we only need:
#   keep_all_clean : mask = 14 real                      (natural per-task menu)
#   gplus_clean    : mask = 14 real minus 5 image-output (Ground+'s declutter)
# 32B, greedy, full 229.
set -uo pipefail
source /project/6101776/xzhan576/gta2-envlab/scripts/common_env.sh; conda activate $GTA_BIG/envs/opencompass
RUN=/datasets/omni_pretraining/gta2/ocrun; PROXY=http://127.0.0.1:16281
FULL=$GTA_BIG/data/gta_dataset/toolmeta.json
NAT=$(python3 -c "import json;print(','.join(json.load(open('$FULL'))))")
DECL=$(python3 -c "
import json
img={'AddText','DrawBox','ImageStylization','Plot','TextToImage'}
print(','.join([t for t in json.load(open('$FULL')) if t not in img]))")
export GTA_MODEL_NAME=qwen2.5-32b-instruct GTA_LLM_URL=http://127.0.0.1:12580/v1/chat/completions \
       GTA_TOOLSERVER=$PROXY GTA_EVAL_MODES=end GTA_TOOLMETA=$FULL GTA_TEMP=0
cd $RUN
doeval(){ local RID=$1 MASK=$2
  [ -e $GTA_BIG/results/$RID/DONE ] && { echo "skip $RID"; return; }
  local WD=$GTA_BIG/results/$RID; mkdir -p $WD
  python3 -c "
import json,urllib.request
body=json.dumps({'mode':'passthrough','mask':'$MASK'.split(','),'per_tool_modes':{},'phi_toolmeta':None,
 'unavailable_tools':['GoogleSearch','MathOCR'],'log_path':'$WD/proxy.jsonl','run_meta':{'run_id':'$RID'}}).encode()
urllib.request.urlopen(urllib.request.Request('$PROXY/proxy_config',data=body,method='POST',headers={'Content-Type':'application/json'}),timeout=30).read()"
  GTA_EXTRA_TOOLS="" python $GTA_REPO/opencompass/run.py configs/gta_atomic_env.py --max-num-workers 1 --debug -w $WD 2>&1 | grep -vE "Future|warn|Resource|TRANSFORMERS" | tail -1
  local f=$(ls $WD/*/results/*/gta_bench_end.json 2>/dev/null|tail -1)
  [ -n "$f" ] && { touch $WD/DONE; echo "[clean] $RID acc=$(python3 -c "import json;print(round(json.load(open('$f'))['answer_acc'],2))")"; } || echo "[clean] $RID FAILED"
}
doeval clean_gplus_s1    "$DECL"   # Ground+ declutter (9 tools)
doeval clean_keepall_s1  "$NAT"    # natural 14 (paired baseline, same stack)
echo "[clean] complete"
