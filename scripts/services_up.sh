#!/bin/bash
# Start (or restart) the three services from the LOGIN node: each runs as an
# `srun --overlap` client held inside a login-node tmux session, so it survives
# assistant/session restarts and is user-manageable (`tmux attach -t gta_llm`).
# Usage: services_up.sh [llm|tool|proxy|all(default)]
set -euo pipefail
export SLURM_CONF=${SLURM_CONF:-/cm/shared/apps/slurm/var/etc/killarney/slurm.conf}
S=/project/6101776/xzhan576/gta2-envlab/scripts
LOGD=/datasets/omni_pretraining/gta2/service_logs
mkdir -p "$LOGD"
JOBID=${GTA_JOBID:-$(squeue -u "$USER" -h -o "%i %j" | awk -v n="${GTA_JOB_NAME:-l1}" '$2==n{print $1; exit}')}
[ -z "$JOBID" ] && { echo "no SLURM job found; set GTA_JOBID" >&2; exit 1; }
WHAT=${1:-all}

up() {  # up <session> <script>
  tmux kill-session -t "$1" 2>/dev/null || true
  tmux new-session -d -s "$1" \
    "export SLURM_CONF=$SLURM_CONF; srun --overlap --jobid=$JOBID bash $S/$2 2>&1 | tee -a $LOGD/$1.log"
  echo "started tmux:$1 -> srun(job $JOBID) $2 (log: $LOGD/$1.log)"
}

case "$WHAT" in
  llm|all)   up gta_llm  start_llm.sh ;;&
  tool|all)  up gta_tool start_toolserver.sh ;;&
  proxy|all) up gta_proxy start_proxy.sh ;;&
esac
tmux ls | grep gta_ || true
