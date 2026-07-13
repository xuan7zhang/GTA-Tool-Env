#!/bin/bash
# Injection-free mask gain on a WEAKER model (7B). Weak tool-selection -> full
# pool per task may genuinely derail it (misfire -> waste turns -> fail), so
# per-task mask could improve AnsAcc where 32B (which ignores clutter) showed null.
set -uo pipefail
source /project/6101776/xzhan576/gta2-envlab/scripts/common_env.sh; conda activate $GTA_BIG/envs/opencompass
RUN=/datasets/omni_pretraining/gta2/ocrun; PROXY=http://127.0.0.1:16281
FULL=$GTA_BIG/data/gta_dataset/toolmeta.json
NATURAL=$(python3 -c "import json;print(','.join(json.load(open('$FULL'))))")
export GTA_MODEL_NAME=qwen2.5-7b-instruct GTA_LLM_URL=http://127.0.0.1:12581/v1/chat/completions \
       GTA_TOOLSERVER=$PROXY GTA_EVAL_MODES=end GTA_TOOLMETA=$FULL
cd $RUN
# proxy already at mask=14 natural (set by maskC); re-assert to be safe (same config, no conflict)
python3 -c "import json,urllib.request; urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:16281/proxy_config',data=json.dumps({'mode':'passthrough','mask':'$NATURAL'.split(','),'per_tool_modes':{},'phi_toolmeta':None,'unavailable_tools':['GoogleSearch','MathOCR']}).encode(),method='POST',headers={'Content-Type':'application/json'}),timeout=30).read()"
doeval(){ local RID=$1 EXTRA=$2
  [ -e $GTA_BIG/results/$RID/DONE ] && { echo "skip $RID"; return; }
  local WD=$GTA_BIG/results/$RID; mkdir -p $WD
  GTA_EXTRA_TOOLS="$EXTRA" python $GTA_REPO/opencompass/run.py configs/gta_atomic_env.py --max-num-workers 1 --debug -w $WD 2>&1 | grep -vE "Future|warn|Resource|TRANSFORMERS" | tail -1
  ls $WD/*/results/*/gta_bench_end.json >/dev/null 2>&1 && { touch $WD/DONE; echo "[mask7b] $RID acc=$(python3 -c "import json,glob;print(round(json.load(open(sorted(glob.glob('$WD/*/results/*/gta_bench_end.json'))[-1]))['answer_acc'],2))")"; } || echo "[mask7b] $RID FAILED"
}
for s in 1 2 3; do
  doeval mask7b_fulleverywhere_s$s "$NATURAL"
  doeval mask7b_pertask_s$s ""
done
echo "[mask7b] complete"
