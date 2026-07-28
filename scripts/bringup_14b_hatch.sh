#!/bin/bash
# Bring up the dynamic-mask (escape-hatch) stack with Qwen2.5-14B-Instruct.
# 14B fp16 tp=2 (GPU0,1); tool server GPU2 with GTA_HATCH=1 on DIRECT port 16182
# (the proxy filters custom meta-tools, so hatch_exp.sh talks to 16182 directly —
# no proxy needed). No poison (clean pool). Login-node tmux holds the srun clients.
# Env: GTA_JOB_NAME (default the running interactive job you pass), GTA_LANE.
set -euo pipefail
export SLURM_CONF=${SLURM_CONF:-/cm/shared/apps/slurm/var/etc/killarney/slurm.conf}
S=/project/6101776/xzhan576/gta2-envlab/scripts
LOGD=/datasets/omni_pretraining/gta2/service_logs
mkdir -p "$LOGD"
JOBID=${GTA_JOBID:?set GTA_JOBID to the target interactive job id}

COMMON="export SLURM_CONF=$SLURM_CONF; \
export GTA_MODEL_PATH=/datasets/omni_pretraining/gta2/models/Qwen2.5-14B-Instruct; \
export GTA_MODEL_NAME=qwen2.5-14b-instruct; \
export GTA_LLM_GPUS=0,1 GTA_LLM_PORT=12580 GTA_KV_FRAC=0.4 \
       GTA_TOOL_GPU=2 GTA_TOOL_PORT=16182 GTA_HATCH=1 GTA_POISON=0"

LANE=${GTA_LANE:-h14}
for s in llm tool; do tmux kill-session -t gta14${LANE}_$s 2>/dev/null || true; done

tmux new-session -d -s gta14${LANE}_llm  "$COMMON; srun --overlap --jobid=$JOBID bash $S/start_llm.sh 2>&1 | tee -a $LOGD/gta14${LANE}_llm.log"
tmux new-session -d -s gta14${LANE}_tool "$COMMON; srun --overlap --jobid=$JOBID bash $S/start_toolserver.sh 2>&1 | tee -a $LOGD/gta14${LANE}_tool.log"
echo "launched 14B hatch lane '$LANE' on job $JOBID; logs $LOGD/gta14${LANE}_*.log"
tmux ls | grep gta14${LANE}
