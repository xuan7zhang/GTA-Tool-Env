#!/bin/bash
# Bring up the full stack with Qwen2.5-32B-Instruct on an idle 4-GPU node.
# 32B fp16 on tp=2 (GPU0,1); tool server GPU2; proxy CPU. Login-node tmux holds
# the srun clients so they survive assistant restarts.
# Env: GTA_JOB_NAME (default l2 = kn066), GTA_NODE_IP auto-detected for eval.
set -euo pipefail
export SLURM_CONF=${SLURM_CONF:-/cm/shared/apps/slurm/var/etc/killarney/slurm.conf}
S=/project/6101776/xzhan576/gta2-envlab/scripts
LOGD=/datasets/omni_pretraining/gta2/service_logs
mkdir -p "$LOGD"
JOBID=${GTA_JOBID:-$(squeue -u "$USER" -h -o "%i %j" | awk -v n="${GTA_JOB_NAME:-l2}" '$2==n{print $1; exit}')}
[ -z "$JOBID" ] && { echo "no SLURM job for ${GTA_JOB_NAME:-l2}" >&2; exit 1; }

COMMON="export SLURM_CONF=$SLURM_CONF; \
export GTA_MODEL_PATH=/datasets/omni_pretraining/gta2/models/Qwen2.5-32B-Instruct; \
export GTA_MODEL_NAME=qwen2.5-32b-instruct; \
export GTA_LLM_GPUS=0,1 GTA_TOOL_GPU=2 GTA_KV_FRAC=0.4 GTA_POISON=${GTA_POISON:-1}"

LANE=${GTA_LANE:-a}   # session suffix so multiple lanes on distinct nodes coexist
for s in llm tool proxy; do tmux kill-session -t gta32${LANE}_$s 2>/dev/null || true; done

tmux new-session -d -s gta32${LANE}_llm   "$COMMON; srun --overlap --jobid=$JOBID bash $S/start_llm.sh 2>&1 | tee -a $LOGD/gta32${LANE}_llm.log"
tmux new-session -d -s gta32${LANE}_tool  "$COMMON; srun --overlap --jobid=$JOBID bash $S/start_toolserver.sh 2>&1 | tee -a $LOGD/gta32${LANE}_tool.log"
tmux new-session -d -s gta32${LANE}_proxy "$COMMON; srun --overlap --jobid=$JOBID bash $S/start_proxy.sh 2>&1 | tee -a $LOGD/gta32${LANE}_proxy.log"
echo "launched 32B lane '$LANE' on job $JOBID (${GTA_JOB_NAME:-l2}); logs $LOGD/gta32${LANE}_*.log"
tmux ls | grep gta32${LANE}
