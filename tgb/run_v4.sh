#!/bin/bash
# TGB-v4 (text-only, no OCR anywhere) scoring + greedy search, one model per GPU.
#
#   ./tgb/run_v3.sh [job-name]     default: l2
#
# Guards, both learned the hard way: an empty jobid makes srun fail instantly
# and silently, which cost a whole night; and a busy GPU makes vLLM abort at
# startup. Neither failure is visible unless you go read a 150-byte log, so
# both are checked here before anything is launched.
set -u
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/killarney/slurm.conf
NAME=${1:-l2}
LAB=/project/6101776/xzhan576/gta2-envlab
M=/datasets/omni_pretraining/gta2/models
T3=/datasets/omni_pretraining/gta2/results/taco/tgb4
PY=/project/aip-xli135/xzhan576/OPSD/.venv/bin/python
LOG=$LAB/logs_v4
mkdir -p "$LOG"

JOB=$(squeue -u "$USER" -h -o "%i %j" | awk -v n="$NAME" '$2==n{print $1; exit}')
if [ -z "$JOB" ]; then
  echo "FATAL: no running job named '$NAME'. squeue says:"
  squeue -u "$USER" -o "%.10i %.8j %.8T %R"
  exit 1
fi
NODE=$(squeue -u "$USER" -h -o "%i %N" | awk -v j="$JOB" '$1==j{print $2}')
echo "using job $JOB on $NODE"

FREE=$(srun --overlap --jobid="$JOB" --ntasks=1 \
       nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
       2>/dev/null | awk -F', ' '$2 < 2000 {print $1}')
echo "free GPUs: $(echo $FREE | tr '\n' ' ')"
[ -z "$FREE" ] && { echo "FATAL: no free GPU on $NODE"; exit 1; }

set -- $FREE
LANES=("7b Qwen2.5-7B-Instruct" "14b Qwen2.5-14B-Instruct"
       "llama8b Llama-3.1-8B-Instruct" "mistral7b Mistral-7B-Instruct-v0.3")
i=0
for lane in "${LANES[@]}"; do
  set -- $lane
  tag=$1; model=$2
  gpu=$(echo $FREE | cut -d' ' -f$((i + 1)))
  [ -z "$gpu" ] && { echo "  no GPU left for $tag, skipping"; continue; }
  nohup setsid bash -c "
    for attempt in 1 2 3 4 5 6 7 8; do
      srun --overlap --jobid=$JOB --nodes=1 --ntasks=1 bash -c '
        export CUDA_VISIBLE_DEVICES=$gpu GD_TAG=$tag TGB_V4=1
        export GD_MODEL=$M/$model HF_HOME=/datasets/omni_pretraining/gta2/hf_home
        export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
        cd $LAB
        $PY -m tgb.score_dl --tasks $T3/tgb_tasks.json --out $T3 --chunk 200
        $PY -m tgb.greedy_env --tasks $T3/tgb_tasks.json --out $T3 --train 400 --v4
      '
      rc=\$?
      echo \"[runner] $tag attempt \$attempt exited rc=\$rc\"
      # both stages are resumable, so a retry picks up where the kill landed;
      # rc=0 means the lane genuinely finished
      [ \$rc -eq 0 ] && break
      sleep 30
    done
    echo '[runner] $tag lane finished'
  " > "$LOG/lane_${tag}.log" 2>&1 &
  echo "  lane $tag -> $NODE:gpu$gpu"
  i=$((i + 1))
  sleep 100
done
echo "launched; logs in $LOG"
