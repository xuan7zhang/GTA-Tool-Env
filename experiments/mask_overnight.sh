#!/bin/bash
# Overnight no-GT per-task mask battery. All eval on stack#1 (LLM varies, proxy 16281,
# tool server 16181). Agent models served sequentially on GPU3 (7B reuses 12580).
#   Phase 0: 14B strong router -> router_masks_q14b -> mask_hybridStrong.json
#   Phase 1: for agent in {7B,3B,14B}: full / pertask(oracle) / predicted(7B-hybrid) /
#            predstr(14B-hybrid), 3 seeds, full 229
#   Phase 2: recovery = (predicted-full)/(oracle-full) per model
set -uo pipefail
source /project/6101776/xzhan576/gta2-envlab/scripts/common_env.sh
LAB=/project/6101776/xzhan576/gta2-envlab
SE=/datasets/omni_pretraining/gta2/scripts_extra
IO=/datasets/omni_pretraining/gta2/results/inject_opt
LOGDIR=/datasets/omni_pretraining/gta2/results/rr_stack
PORT=12591
export MASK_A=$IO/mask_hybridA.json
export MASK_STRONG=$IO/mask_hybridStrong.json

gpu3_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 2>/dev/null | tr -d ' '; }
serve(){ # <path> <name> <tp>  -> serves on GPU3:$PORT, returns pid in SERVE_PID; waits ready
  local MPATH=$1 MNAME=$2 TP=${3:-1}
  CUDA_VISIBLE_DEVICES=3 GTA_LLM_GPUS=3 GTA_LLM_PORT=$PORT GTA_MODEL_PATH="$MPATH" \
    GTA_MODEL_NAME="$MNAME" GTA_KV_FRAC=0.4 bash "$LAB/scripts/start_llm.sh" \
    > "$LOGDIR/llm_$MNAME.log" 2>&1 &
  SERVE_PID=$!
  for i in $(seq 1 90); do
    curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 200 && { echo "[serve] $MNAME ready"; return 0; }
    kill -0 $SERVE_PID 2>/dev/null || { echo "[serve] $MNAME DIED"; tail -4 "$LOGDIR/llm_$MNAME.log"; return 1; }
    sleep 10
  done; echo "[serve] $MNAME TIMEOUT"; return 1
}
unserve(){ kill $SERVE_PID 2>/dev/null || true
  for i in $(seq 1 30); do mb=$(gpu3_used); [ -z "$mb" ] && mb=0; [ "$mb" -lt 2000 ] && break; sleep 5; done
  echo "[serve] GPU3 freed"; }

# ---------- Phase 0: 14B strong router builds hybridStrong ----------
echo "==================== Phase 0: 14B router ===================="
if [ ! -e "$MASK_STRONG" ]; then
  if serve "$GTA_BIG/models/Qwen2.5-14B-Instruct" qwen2.5-14b-instruct 1; then
    python "$SE/router_selector.py" qwen2.5-14b-instruct $PORT q14b 2>&1 | grep -vE "Future|warn|\.\.\." | tail -2
    python "$SE/build_hybrid.py" "$IO/router_masks_q14b.json" "$MASK_STRONG"
    unserve
  else echo "[phase0] 14B router failed; hybridStrong = hybridA fallback"; cp "$MASK_A" "$MASK_STRONG"; unserve; fi
else echo "hybridStrong exists, skip"; fi

# ---------- Phase 1: recovery curve ----------
echo "==================== Phase 1: 7B (reuse 12580) ===================="
curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:12580/v1/models | grep -q 200 \
  && bash "$SE/predmask_curve.sh" qwen2.5-7b-instruct 12580 pm7b 3 || echo "[phase1] 7B(12580) not up, skip"

echo "==================== Phase 1: 3B ===================="
if serve "$GTA_BIG/models/Qwen2.5-3B-Instruct" qwen2.5-3b-instruct 1; then
  bash "$SE/predmask_curve.sh" qwen2.5-3b-instruct $PORT pm3b 3; unserve
fi

echo "==================== Phase 1: 14B ===================="
if serve "$GTA_BIG/models/Qwen2.5-14B-Instruct" qwen2.5-14b-instruct 1; then
  bash "$SE/predmask_curve.sh" qwen2.5-14b-instruct $PORT pm14b 3; unserve
fi

# ---------- Phase 2: recovery table ----------
echo "==================== Phase 2: recovery ===================="
python3 - <<'PY'
import json,glob,statistics as st
BASE="/datasets/omni_pretraining/gta2/results"
def acc(rid):
    j=sorted(glob.glob(f"{BASE}/{rid}/*/results/*/gta_bench_end.json"));
    return json.load(open(j[-1]))['answer_acc'] if j else None
def m(pfx,cond):
    v=[acc(f"{pfx}_{cond}_s{s}") for s in (1,2,3)]; v=[x for x in v if x is not None]
    return st.mean(v) if v else None
print(f"{'model':<8} {'full':>6} {'oracle':>7} {'pred7B':>7} {'pred14B':>8} {'recov%(pred7B)':>13}")
for pfx,name in [('pm7b','7B'),('pm3b','3B'),('pm14b','14B')]:
    f,o,p,ps=m(pfx,'full'),m(pfx,'pertask'),m(pfx,'predicted'),m(pfx,'predstr')
    if None in (f,o): print(f"{name:<8} (incomplete)"); continue
    rec=(p-f)/(o-f)*100 if (o-f) and p is not None else float('nan')
    print(f"{name:<8} {f:6.1f} {o:7.1f} {str(round(p,1) if p else '-'):>7} {str(round(ps,1) if ps else '-'):>8} {rec:12.0f}%")
PY
echo "[overnight] ALL DONE"
