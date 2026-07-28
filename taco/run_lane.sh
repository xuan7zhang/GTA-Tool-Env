#!/bin/bash
# Run a TACO manifest on one lane, inside a login-node tmux session holding an
# srun --overlap client on the GPU node. Survives assistant/session restarts.
#
# Usage: taco/run_lane.sh <JOBID> <lane> <model_name> <manifest> [session_suffix]
set -euo pipefail
export SLURM_CONF=${SLURM_CONF:-/cm/shared/apps/slurm/var/etc/killarney/slurm.conf}
LAB=/project/6101776/xzhan576/gta2-envlab
# TACO_ROOT isolates this session's outputs from any other session running
# the same code concurrently (see taco/paths.py).
export TACO_ROOT=${TACO_ROOT:-/datasets/omni_pretraining/gta2/results/taco_b}
LOGD=$TACO_ROOT/logs
mkdir -p "$LOGD"

JOBID=$1; LANE=$2; MODEL=$3; MANIFEST=$4; SUF=${5:-run}
SESS="taco_lane${LANE}_${SUF}"
LOG="$LOGD/${SESS}.log"

tmux kill-session -t "$SESS" 2>/dev/null || true
tmux new-session -d -s "$SESS" "export SLURM_CONF=$SLURM_CONF; \
srun --overlap --jobid=$JOBID bash -lc '
export TACO_ROOT=$TACO_ROOT
source $LAB/scripts/common_env.sh
conda activate \$GTA_BIG/envs/opencompass
cd $LAB
python taco/runner.py --manifest $MANIFEST --lane $LANE --model $MODEL
' 2>&1 | tee -a $LOG"
echo "started $SESS: lane $LANE $MODEL $(basename "$MANIFEST")  log $LOG"
