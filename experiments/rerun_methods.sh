#!/bin/bash
# Re-run the injection-optimization method evals from a /datasets cwd, because
# /project hit its group disk quota (opencompass writes tmp/<id>_params.py
# relative to cwd). Reuses the live kn066 32B stack + proxy. Per method: set the
# proxy mask + GTA_EXTRA_TOOLS from selectors.json, then run one full eval.
set -uo pipefail
source /project/6101776/xzhan576/gta2-envlab/scripts/common_env.sh
conda activate $GTA_BIG/envs/opencompass
RUN=/datasets/omni_pretraining/gta2/ocrun
PROXY=http://127.0.0.1:${GTA_PROXY_PORT:-16281}
SEL=$GTA_BIG/results/inject_opt/selectors.json
FULL=$GTA_BIG/data/gta_dataset/toolmeta.json
POISON=$(python3 -c "import json;print(','.join(json.load(open('$GTA_LAB/envgen/variants/poison/names.json'))))")
export GTA_MODEL_NAME=qwen2.5-32b-instruct
cd $RUN

for m in "$@"; do
  RID=inopt_$m
  if [ -e $GTA_BIG/results/$RID/DONE ]; then echo "skip $m (done)"; continue; fi
  # keep-set + kept poison for this method
  IFS=$'\t' read KEEP EXTRA < <(python3 - "$SEL" "$m" "$POISON" <<'EOF'
import json,sys
sel=json.load(open(sys.argv[1])); m=sys.argv[2]; poison=set(sys.argv[3].split(','))
keep=sel[m]['keep']; kept_poison=[t for t in keep if t in poison]
print(json.dumps(keep,separators=(',',':')) + '\t' + (','.join(kept_poison) if kept_poison else '_'))
EOF
)
  [ "$EXTRA" = "_" ] && EXTRA=""
  echo "[rerun] $m : keep $(python3 -c "import json,sys;print(len(json.loads(sys.argv[1])))" "$KEEP") tools, extra=$EXTRA"
  # configure proxy
  python3 - "$PROXY" "$KEEP" "$GTA_BIG/results/$RID/proxy_calls.jsonl" "$RID" <<'EOF'
import json,sys,urllib.request,os
proxy,keep,log,rid=sys.argv[1:5]
os.makedirs(os.path.dirname(log),exist_ok=True)
body=json.dumps({"mode":"passthrough","mask":json.loads(keep),"per_tool_modes":{},
  "unavailable_tools":["GoogleSearch","MathOCR"],"log_path":log,"run_meta":{"run_id":rid}}).encode()
urllib.request.urlopen(urllib.request.Request(proxy+"/proxy_config",data=body,method="POST",
  headers={"Content-Type":"application/json"}),timeout=30).read()
EOF
  WORKDIR=$GTA_BIG/results/$RID; mkdir -p $WORKDIR
  GTA_EVAL_MODES=end GTA_TOOLMETA=$FULL GTA_EXTRA_TOOLS=$EXTRA GTA_TOOLSERVER=$PROXY \
    python $GTA_REPO/opencompass/run.py configs/gta_atomic_env.py --max-num-workers 1 --debug -w $WORKDIR \
    2>&1 | grep -vE "FutureWarning|warnings.warn|ResourceWarning|TRANSFORMERS_CACHE" | tail -3
  if ls $WORKDIR/*/results/*/gta_bench_end.json >/dev/null 2>&1; then
    gzip -f $WORKDIR/proxy_calls.jsonl 2>/dev/null || true
    touch $WORKDIR/DONE
    echo "[rerun] $m DONE acc=$(python3 -c "import json,glob;print(round(json.load(open(sorted(glob.glob('$WORKDIR/*/results/*/gta_bench_end.json'))[-1]))['answer_acc'],2))")"
  else
    echo "[rerun] $m FAILED (no results)"
  fi
done
echo "[rerun] all requested methods complete"
