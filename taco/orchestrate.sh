#!/bin/bash
# Unattended TACO lane driver. Runs a lane's whole remaining queue in order,
# inside ONE login-node tmux session holding ONE srun client, so the queue
# survives the assistant session ending. Every stage is idempotent: the runner
# skips any condition that already has a DONE marker, so this script can be
# killed and restarted at any point without losing or repeating work.
#
# Usage: taco/orchestrate.sh <JOBID> <lane> <model> <tag> <stage2_manifest> [more manifests...]
set -uo pipefail
export SLURM_CONF=${SLURM_CONF:-/cm/shared/apps/slurm/var/etc/killarney/slurm.conf}
LAB=/project/6101776/xzhan576/gta2-envlab
export TACO_ROOT=${TACO_ROOT:-/datasets/omni_pretraining/gta2/results/taco_b}
LOGD=$TACO_ROOT/logs; mkdir -p "$LOGD"

JOBID=$1; LANE=$2; MODEL=$3; TAG=$4; shift 4
MANIFESTS="$*"
SESS="taco_orch${LANE}"
LOG="$LOGD/${SESS}.log"

INNER="
export TACO_ROOT=$TACO_ROOT
source $LAB/scripts/common_env.sh
conda activate \$GTA_BIG/envs/opencompass
cd $LAB
for M in $MANIFESTS; do
  echo \"[orch] ===== lane $LANE $MODEL manifest \$M  \$(date -Is)\"
  python taco/runner.py --manifest \$M --lane $LANE --model $MODEL || echo '[orch] manifest returned nonzero'
done
echo \"[orch] lane $LANE queue complete \$(date -Is)\"
"

tmux kill-session -t "$SESS" 2>/dev/null
tmux new-session -d -s "$SESS" \
  "export SLURM_CONF=$SLURM_CONF; srun --overlap --jobid=$JOBID bash -lc '$INNER' 2>&1 | tee -a $LOG"
echo "started $SESS -> lane $LANE $MODEL: $MANIFESTS"
echo "log: $LOG"
