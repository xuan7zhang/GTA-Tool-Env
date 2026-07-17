#!/bin/bash
# Signal ablation for the no-GT mask selector, ONLINE. Leave-one-out over the three
# signals (router / emb / freq) + singles, vs full & oracle anchors. Runs on the
# SEPARATE RegionRead stack (proxy 16282 -> tool server 16182, 7B on 12580) so it can
# run in parallel with the overnight recovery battery (which uses 16281).
#   abl_exp.sh [nseed]
set -uo pipefail
source /project/6101776/xzhan576/gta2-envlab/scripts/common_env.sh; conda activate $GTA_BIG/envs/opencompass
RUN=/datasets/omni_pretraining/gta2/ocrun; PROXY=http://127.0.0.1:16282
FULL=$GTA_BIG/data/gta_dataset/toolmeta.json; IO=$GTA_BIG/results/inject_opt
NATURAL=$(python3 -c "import json;print(','.join(json.load(open('$FULL'))))")
NSEED=${1:-2}
export GTA_MODEL_NAME=qwen2.5-7b-instruct GTA_LLM_URL=http://127.0.0.1:12580/v1/chat/completions \
       GTA_TOOLSERVER=$PROXY GTA_EVAL_MODES=end GTA_TOOLMETA=$FULL
cd $RUN
python3 -c "import json,urllib.request;urllib.request.urlopen(urllib.request.Request('$PROXY/proxy_config',data=json.dumps({'mode':'passthrough','mask':'$NATURAL'.split(','),'per_tool_modes':{},'phi_toolmeta':None,'unavailable_tools':['GoogleSearch','MathOCR']}).encode(),method='POST',headers={'Content-Type':'application/json'}),timeout=30).read()"

doeval(){ local RID=$1 EXTRA=$2 PMASK=$3
  [ -e $GTA_BIG/results/$RID/DONE ] && { echo "skip $RID"; return; }
  local WD=$GTA_BIG/results/$RID; mkdir -p $WD
  GTA_EXTRA_TOOLS="$EXTRA" GTA_PREDICTED_MASK="$PMASK" \
    python $GTA_REPO/opencompass/run.py configs/gta_atomic_env.py --max-num-workers 1 --debug -w $WD 2>&1 | grep -vE "Future|warn|Resource|TRANSFORMERS" | tail -1
  if ls $WD/*/results/*/gta_bench_end.json >/dev/null 2>&1; then touch $WD/DONE
    echo "[abl] $RID acc=$(python3 -c "import json,glob;print(round(json.load(open(sorted(glob.glob('$WD/*/results/*/gta_bench_end.json'))[-1]))['answer_acc'],2))")"
  else echo "[abl] $RID FAILED"; fi
}
for s in $(seq 1 $NSEED); do
  doeval abl_full_s$s       "$NATURAL" ""
  doeval abl_oracle_s$s     ""         ""
  doeval abl_all3_s$s       "$NATURAL" "$IO/mask_hybridA.json"
  doeval abl_norouter_s$s   "$NATURAL" "$IO/abl_emb_freq.json"
  doeval abl_noemb_s$s      "$NATURAL" "$IO/abl_router_freq.json"
  doeval abl_nofreq_s$s     "$NATURAL" "$IO/abl_router_emb.json"
  doeval abl_freqonly_s$s   "$NATURAL" "$IO/abl_freq.json"
  doeval abl_routeronly_s$s "$NATURAL" "$IO/abl_router.json"
  doeval abl_embonly_s$s    "$NATURAL" "$IO/abl_emb.json"
done
echo "[abl] complete"
