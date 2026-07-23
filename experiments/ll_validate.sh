#!/bin/bash
# Answer log-likelihood as a label-free signal for tool-space optimization.
# Run 3 tool configs on a task subset, capturing per-generation token logprobs
# (GTA_LL_LOG). Post-hoc we match the final answer to recover its LL and test
# whether answer-LL tracks accuracy across configs and predicts correctness.
#   full   : mask off (all 14 forced)          -- lower acc
#   oracle : per-task GT tools                  -- higher acc
#   poison : 14 real + 8 injected (attractive)  -- lowest acc
set -uo pipefail
source /project/6101776/xzhan576/gta2-envlab/scripts/common_env.sh; conda activate $GTA_BIG/envs/opencompass
RUN=/datasets/omni_pretraining/gta2/ocrun; PROXY=http://127.0.0.1:16281
FULL=$GTA_BIG/data/gta_dataset/toolmeta.json; IO=$GTA_BIG/results/inject_opt
NAT=$(python3 -c "import json;print(','.join(json.load(open('$FULL'))))")
POISON=$(python3 -c "import json;print(','.join(json.load(open('$GTA_LAB/envgen/variants/poison/names.json')))")
SUB=${1:-60}
export GTA_MODEL_NAME=qwen2.5-32b-instruct GTA_LLM_URL=http://127.0.0.1:12580/v1/chat/completions \
       GTA_TOOLSERVER=$PROXY GTA_EVAL_MODES=end GTA_TOOLMETA=$FULL GTA_TEMP=0 GTA_TASK_SUBSET=$SUB
cd $RUN
LLDIR=$GTA_BIG/results/ll_signal; mkdir -p $LLDIR
doeval(){ local RID=$1 MASK=$2 EXTRA=$3
  [ -e $GTA_BIG/results/$RID/DONE ] && { echo "skip $RID"; return; }
  local WD=$GTA_BIG/results/$RID; mkdir -p $WD; local LL=$LLDIR/${RID}.jsonl; : > $LL
  python3 -c "
import json,urllib.request
body=json.dumps({'mode':'passthrough','mask':'$MASK'.split(','),'per_tool_modes':{},'phi_toolmeta':None,
 'unavailable_tools':['GoogleSearch','MathOCR'],'log_path':'$WD/proxy.jsonl','run_meta':{'run_id':'$RID'}}).encode()
urllib.request.urlopen(urllib.request.Request('$PROXY/proxy_config',data=body,method='POST',headers={'Content-Type':'application/json'}),timeout=30).read()"
  GTA_EXTRA_TOOLS="$EXTRA" GTA_LL_LOG="$LL" python $GTA_REPO/opencompass/run.py configs/gta_atomic_env.py --max-num-workers 1 --debug -w $WD 2>&1 | grep -vE "Future|warn|Resource|TRANSFORMERS" | tail -1
  local f=$(ls $WD/*/results/*/gta_bench_end.json 2>/dev/null|tail -1)
  [ -n "$f" ] && { touch $WD/DONE; echo "[ll] $RID acc=$(python3 -c "import json;print(round(json.load(open('$f'))['answer_acc'],2))") ll_lines=$(wc -l <$LL)"; } || echo "[ll] $RID FAILED"
}
doeval ll_oracle "$NAT"           ""
doeval ll_full   "$NAT"           "$NAT"
doeval ll_poison "$NAT,$POISON"   "$POISON"
echo "[ll] complete"
