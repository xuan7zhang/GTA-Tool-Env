#!/bin/bash
# Overnight signal battery: can any entropy/likelihood form guide tool-space opt?
# 7B, full 229, 3 configs spanning a clean accuracy gradient:
#   oracle  : per-task GT tools       (HIGH acc)
#   full    : all-14 forced (clutter) (LOW acc)
#   corrupt : tool outputs corrupted  (LOWEST acc)
# Phase A: greedy + enriched per-token capture (token-level signals).
# Phase B: temperature-0.8 sampling x5 (answer-level semantic entropy / self-consistency).
set -uo pipefail
source /project/6101776/xzhan576/gta2-envlab/scripts/common_env.sh; conda activate $GTA_BIG/envs/opencompass
RUN=/datasets/omni_pretraining/gta2/ocrun; PROXY=http://127.0.0.1:16281
FULL=$GTA_BIG/data/gta_dataset/toolmeta.json
NAT=$(python3 -c "import json;print(','.join(json.load(open('$FULL'))))")
export GTA_MODEL_NAME=qwen2.5-7b-instruct GTA_LLM_URL=http://127.0.0.1:12581/v1/chat/completions \
       GTA_TOOLSERVER=$PROXY GTA_EVAL_MODES=end GTA_TOOLMETA=$FULL
cd $RUN; LLDIR=$GTA_BIG/results/ll_signal; mkdir -p $LLDIR

setproxy(){ # <mask> <mode> <per_tool_modes_json>
  python3 -c "
import json,urllib.request
body=json.dumps({'mode':'$2','mask':'$1'.split(','),'per_tool_modes':$3,'phi_toolmeta':None,'unavailable_tools':['GoogleSearch','MathOCR'],'log_path':'/dev/null','run_meta':{'run_id':'batt'}}).encode()
urllib.request.urlopen(urllib.request.Request('$PROXY/proxy_config',data=body,method='POST',headers={'Content-Type':'application/json'}),timeout=30).read()"
}
run1(){ # <rid> <extra> <temp> <ll_on>
  local RID=$1 EXTRA=$2 TEMP=$3 LLON=$4
  [ -e $GTA_BIG/results/$RID/DONE ] && { echo "skip $RID"; return; }
  local WD=$GTA_BIG/results/$RID; mkdir -p $WD
  local LLENV=""; [ "$LLON" = 1 ] && LLENV="$LLDIR/${RID}.jsonl" && : > "$LLENV"
  GTA_TEMP="$TEMP" GTA_EXTRA_TOOLS="$EXTRA" GTA_LL_LOG="$LLENV" \
    python $GTA_REPO/opencompass/run.py configs/gta_atomic_env.py --max-num-workers 1 --debug -w $WD 2>&1 | grep -vE "Future|warn|Resource|TRANSFORMERS" | tail -1
  local f=$(ls $WD/*/results/*/gta_bench_end.json 2>/dev/null|tail -1)
  [ -n "$f" ] && { touch $WD/DONE; echo "[batt] $RID acc=$(python3 -c "import json;print(round(json.load(open('$f'))['answer_acc'],2))")"; } || echo "[batt] $RID FAILED"
}
CORRUPT=$(python3 -c "import json;print(json.dumps({t:'corrupt_output' for t in json.load(open('$FULL'))}))")

echo "===== Phase A: greedy + enriched capture ====="
setproxy "$NAT" passthrough "{}";        run1 ll7b_oracle  ""     0 1
setproxy "$NAT" passthrough "{}";        run1 ll7b_full    "$NAT" 0 1
setproxy "$NAT" passthrough "$CORRUPT";  run1 ll7b_corrupt ""     0 1

echo "===== Phase B: temp 0.8 sampling x5 (semantic entropy) ====="
for k in 1 2 3 4 5; do
  setproxy "$NAT" passthrough "{}";       run1 sc_oracle_k$k  ""     0.8 0
  setproxy "$NAT" passthrough "{}";       run1 sc_full_k$k    "$NAT" 0.8 0
  setproxy "$NAT" passthrough "$CORRUPT"; run1 sc_corrupt_k$k ""     0.8 0
done
echo "[batt] complete"
