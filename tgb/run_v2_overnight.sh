#!/bin/bash
# TGB-v2 overnight: does per-task likelihood masking beat greedy environment
# search once the tool menu actually conflicts?
#
#   ./tgb/run_v2_overnight.sh [node]
#
# Stages per model, sequential on one GPU so nothing collides:
#   1 score_dl    dL + greedy answer for every candidate coalition
#   2 greedy_env  backward elimination over the 21-tool menu (train 400)
#   3 loo_mask    coalition-aware per-task mask, tau sweep
# Models run in parallel on separate GPUs.
set -u
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/killarney/slurm.conf
NODE=${1:-kn068}
LAB=/project/6101776/xzhan576/gta2-envlab
M=/datasets/omni_pretraining/gta2/models
TGB2=/datasets/omni_pretraining/gta2/results/taco/tgb2
PY=/project/aip-xli135/xzhan576/OPSD/.venv/bin/python
LOG=$TGB2/logs
mkdir -p "$LOG"
JOB=$(squeue -u "$USER" -h -o "%i %N" | awk -v n="$NODE" '$2==n{print $1; exit}')
[ -z "$JOB" ] && { echo "no job on $NODE"; exit 1; }

lane() {  # gpu tag model
  local gpu=$1 tag=$2 model=$3
  nohup setsid srun --overlap --jobid="$JOB" --nodes=1 --ntasks=1 bash -c "
    export CUDA_VISIBLE_DEVICES=$gpu
    export GD_TAG=$tag GD_MODEL=$model GD_TP=1 GD_UTIL=0.85
    export HF_HOME=/datasets/omni_pretraining/gta2/hf_home
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    export TGB_V2=1
    cd $LAB
    echo '=== stage 1 score_dl'
    $PY -m tgb.score_dl --tasks $TGB2/tgb_tasks.json --out $TGB2
    echo '=== stage 2 greedy_env'
    $PY -m tgb.greedy_env --tasks $TGB2/tgb_tasks.json --out $TGB2 --train 400
    echo '=== stage 3 loo_mask'
    $PY -m tgb.loo_mask --dir $TGB2 --tools \"$TGB2_TOOLS\" --tau=-2.0,-1.0,-0.5,0.0,0.25
    echo '=== lane done'
  " > "$LOG/lane_${tag}.log" 2>&1 &
  echo "  lane $tag -> $NODE:gpu$gpu"
}

export TGB2_TOOLS="OCR,Calculator,CountGivenObject,GoogleSearch,UnitConvert,TempConvert,Solver,DurationCalc,CurrencyConvert"
lane 0 7b        "$M/Qwen2.5-7B-Instruct"
sleep 90
lane 1 14b       "$M/Qwen2.5-14B-Instruct"
sleep 90
lane 2 llama8b   "$M/Llama-3.1-8B-Instruct"
sleep 90
lane 3 mistral7b "$M/Mistral-7B-Instruct-v0.3"
echo "launched on $NODE; logs in $LOG"
