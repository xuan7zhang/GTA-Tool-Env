#!/bin/bash
# No-GT per-task mask, ONLINE test. Three conditions on the SAME model/session
# (fair, all under the current JsonParser-fixed harness), full 229:
#   full      GTA_EXTRA_TOOLS=<all14>                    -> menu = 14 (mask OFF)
#   pertask   GTA_EXTRA_TOOLS=""                         -> menu = GT resources (ORACLE)
#   predicted GTA_EXTRA_TOOLS=<all14> + GTA_PREDICTED_MASK=<json> -> menu = predicted set
# Does the predicted (no-GT) mask recover the oracle gain over full?
#   predmask_exp.sh <model_name> <llm_port> <prefix> <mask_json> [nseed]
set -uo pipefail
source /project/6101776/xzhan576/gta2-envlab/scripts/common_env.sh; conda activate $GTA_BIG/envs/opencompass
RUN=/datasets/omni_pretraining/gta2/ocrun; PROXY=http://127.0.0.1:16281
FULL=$GTA_BIG/data/gta_dataset/toolmeta.json
NATURAL=$(python3 -c "import json;print(','.join(json.load(open('$FULL'))))")
MODEL=$1; PORT=$2; PFX=$3; MASKJSON=$4; NSEED=${5:-3}
export GTA_MODEL_NAME=$MODEL GTA_LLM_URL=http://127.0.0.1:$PORT/v1/chat/completions \
       GTA_TOOLSERVER=$PROXY GTA_EVAL_MODES=end GTA_TOOLMETA=$FULL
cd $RUN
# proxy: keep all 14 in the mask (all forwardable + in self.tools)
python3 -c "import json,urllib.request;urllib.request.urlopen(urllib.request.Request('$PROXY/proxy_config',data=json.dumps({'mode':'passthrough','mask':'$NATURAL'.split(','),'per_tool_modes':{},'phi_toolmeta':None,'unavailable_tools':['GoogleSearch','MathOCR']}).encode(),method='POST',headers={'Content-Type':'application/json'}),timeout=30).read()"

doeval(){ local RID=$1 EXTRA=$2 PMASK=$3
  [ -e $GTA_BIG/results/$RID/DONE ] && { echo "skip $RID"; return; }
  local WD=$GTA_BIG/results/$RID; mkdir -p $WD
  GTA_EXTRA_TOOLS="$EXTRA" GTA_PREDICTED_MASK="$PMASK" \
    python $GTA_REPO/opencompass/run.py configs/gta_atomic_env.py --max-num-workers 1 --debug -w $WD 2>&1 | grep -vE "Future|warn|Resource|TRANSFORMERS" | tail -1
  if ls $WD/*/results/*/gta_bench_end.json >/dev/null 2>&1; then touch $WD/DONE
    echo "[$PFX] $RID acc=$(python3 -c "import json,glob;print(round(json.load(open(sorted(glob.glob('$WD/*/results/*/gta_bench_end.json'))[-1]))['answer_acc'],2))")"
  else echo "[$PFX] $RID FAILED"; fi
}
for s in $(seq 1 $NSEED); do
  doeval ${PFX}_full_s$s      "$NATURAL" ""
  doeval ${PFX}_pertask_s$s   ""         ""
  doeval ${PFX}_predicted_s$s "$NATURAL" "$MASKJSON"
done
echo "[$PFX] complete"
