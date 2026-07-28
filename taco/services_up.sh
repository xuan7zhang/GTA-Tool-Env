#!/bin/bash
# TACO two-lane service bring-up. Each lane is an independent (LLM, tool
# server, proxy) triple so two conditions can run concurrently without
# sharing any state -- important because the tool server is stateful enough
# (per-call GPU work) that sharing it across lanes would couple their timing.
#
#   lane1: LLM gpu0 :12580   tools gpu2 :16181   proxy :16281
#   lane2: LLM gpu1 :22580   tools gpu3 :26181   proxy :26281
#
# Usage: taco/services_up.sh <JOBID> <lane> <model_name> <model_path>
#   e.g. taco/services_up.sh 4396946 1 qwen2.5-7b-instruct  $GTA_BIG/models/Qwen2.5-7B-Instruct
#        taco/services_up.sh 4396946 2 qwen2.5-14b-instruct $GTA_BIG/models/Qwen2.5-14B-Instruct
set -euo pipefail
export SLURM_CONF=${SLURM_CONF:-/cm/shared/apps/slurm/var/etc/killarney/slurm.conf}
LAB=/project/6101776/xzhan576/gta2-envlab
BIG=/datasets/omni_pretraining/gta2
LOGD=$BIG/results/taco/logs
mkdir -p "$LOGD"

JOBID=$1; LANE=$2; MODEL_NAME=$3; MODEL_PATH=$4

if [ "$LANE" = "1" ]; then
  LLM_GPU=0; TOOL_GPU=2; LLM_PORT=12580; TOOL_PORT=16181; PROXY_PORT=16281
else
  LLM_GPU=1; TOOL_GPU=3; LLM_PORT=22580; TOOL_PORT=26181; PROXY_PORT=26281
fi

ENVX="export SLURM_CONF=$SLURM_CONF GTA_LLM_GPUS=$LLM_GPU GTA_TOOL_GPU=$TOOL_GPU \
GTA_LLM_PORT=$LLM_PORT GTA_TOOL_PORT=$TOOL_PORT GTA_PROXY_PORT=$PROXY_PORT \
GTA_MODEL_NAME=$MODEL_NAME GTA_MODEL_PATH=$MODEL_PATH \
GTA_PROXY_LOG=$BIG/results/taco/logs/proxy_lane$LANE.jsonl"

up() {  # up <session> <script>
  tmux kill-session -t "$1" 2>/dev/null || true
  tmux new-session -d -s "$1" \
    "$ENVX; srun --overlap --jobid=$JOBID bash $LAB/scripts/$2 2>&1 | tee -a $LOGD/$1.log"
  echo "started tmux:$1 (lane $LANE) -> $2   log $LOGD/$1.log"
}

up "taco_llm$LANE"   start_llm.sh
up "taco_tool$LANE"  start_toolserver.sh
sleep 5
up "taco_proxy$LANE" start_proxy.sh
tmux ls | grep taco_ || true
