#!/bin/bash
# Run a command on the GPU node via the user's running SLURM job.
#   on_node.sh <command...>
# Env: GTA_JOBID or GTA_JOB_NAME (default l1 = kn066, idle 4x L40S;
#      kn064/l3 currently runs the SDPO RL training — do not use).
set -euo pipefail
export SLURM_CONF=${SLURM_CONF:-/cm/shared/apps/slurm/var/etc/killarney/slurm.conf}
JOBID=${GTA_JOBID:-$(squeue -u "$USER" -h -o "%i %j" | awk -v n="${GTA_JOB_NAME:-l1}" '$2==n{print $1; exit}')}
[ -z "$JOBID" ] && { echo "no SLURM job found; set GTA_JOBID" >&2; exit 1; }
exec srun --overlap --jobid="$JOBID" --ntasks=1 "$@"
