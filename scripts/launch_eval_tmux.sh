#!/bin/bash
# Launch an eval run in a login-node tmux session (survives assistant restarts).
# Usage: launch_eval_tmux.sh <run_id> [eval_modes(default step,end)] [extra env "K=V K=V"]
set -euo pipefail
export SLURM_CONF=${SLURM_CONF:-/cm/shared/apps/slurm/var/etc/killarney/slurm.conf}
S=/project/6101776/xzhan576/gta2-envlab/scripts
LOGD=/datasets/omni_pretraining/gta2/service_logs
mkdir -p "$LOGD"
RUN=${1:?usage: launch_eval_tmux.sh <run_id> [modes] [env]}
MODES=${2:-step,end}
EXTRA_ENV=${3:-}
JOBID=${GTA_JOBID:-$(squeue -u "$USER" -h -o "%i %j" | awk -v n="${GTA_JOB_NAME:-l1}" '$2==n{print $1; exit}')}
SESS=gta_eval_$RUN
tmux kill-session -t "$SESS" 2>/dev/null || true
tmux new-session -d -s "$SESS" \
  "export SLURM_CONF=$SLURM_CONF; srun --overlap --jobid=$JOBID bash -c 'export GTA_EVAL_MODES=$MODES $EXTRA_ENV; bash $S/run_eval.sh $RUN; echo EVAL_EXIT=\$?' 2>&1 | tee -a $LOGD/$SESS.log"
echo "started tmux:$SESS (log: $LOGD/$SESS.log)"
