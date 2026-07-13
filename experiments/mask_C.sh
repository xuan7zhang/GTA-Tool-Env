#!/bin/bash
# Injection-FREE mask gain: per-task relevance. Same 14 natural tools, no poison.
#   full_everywhere: every task sees ALL 14 tools (GTA_EXTRA_TOOLS=all14) -> the
#     agent must choose among mostly-irrelevant tools -> misfire/exploration.
#   per_task_mask: every task sees only its 1-4 resources (GTA default) -> focused.
# If per_task_mask > full_everywhere, mask (per-task relevance) improves accuracy
# with no injected harm.
set -uo pipefail
source /project/6101776/xzhan576/gta2-envlab/scripts/common_env.sh; conda activate $GTA_BIG/envs/opencompass
RUN=/datasets/omni_pretraining/gta2/ocrun; PROXY=http://127.0.0.1:16281
FULL=$GTA_BIG/data/gta_dataset/toolmeta.json
NATURAL="Calculator,OCR,CountGivenObject,ImageDescription,GoogleSearch,RegionAttributeDescription,TextToBbox,Plot,MathOCR,Solver,DrawBox,AddText,TextToImage,ImageStylization"
export GTA_MODEL_NAME=qwen2.5-32b-instruct GTA_TOOLSERVER=$PROXY GTA_EVAL_MODES=end GTA_TOOLMETA=$FULL
cd $RUN
# proxy: mask to 14 natural (hide poison), passthrough, no phi
python3 -c "import json,urllib.request; urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:16281/proxy_config',data=json.dumps({'mode':'passthrough','mask':'Calculator,OCR,CountGivenObject,ImageDescription,GoogleSearch,RegionAttributeDescription,TextToBbox,Plot,MathOCR,Solver,DrawBox,AddText,TextToImage,ImageStylization'.split(','),'per_tool_modes':{},'phi_toolmeta':None,'unavailable_tools':['GoogleSearch','MathOCR']}).encode(),method='POST',headers={'Content-Type':'application/json'}),timeout=30).read()"
doeval(){ local COND=$1 RID=$2 EXTRA=$3
  [ -e $GTA_BIG/results/$RID/DONE ] && { echo "skip $RID"; return; }
  local WD=$GTA_BIG/results/$RID; mkdir -p $WD
  GTA_EXTRA_TOOLS="$EXTRA" python $GTA_REPO/opencompass/run.py configs/gta_atomic_env.py --max-num-workers 1 --debug -w $WD 2>&1 | grep -vE "Future|warn|Resource|TRANSFORMERS" | tail -1
  ls $WD/*/results/*/gta_bench_end.json >/dev/null 2>&1 && { touch $WD/DONE; echo "[maskC] $RID acc=$(python3 -c "import json,glob;print(round(json.load(open(sorted(glob.glob('$WD/*/results/*/gta_bench_end.json'))[-1]))['answer_acc'],2))")"; } || echo "[maskC] $RID FAILED"
}
for s in 1 2 3; do
  doeval full_everywhere maskC_fulleverywhere_s$s "$NATURAL"
  doeval per_task_mask   maskC_pertask_s$s        ""
done
echo "[maskC] complete"
