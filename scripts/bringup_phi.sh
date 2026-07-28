#!/bin/bash
# Φ-axis stack: Qwen2.5-7B on GPU0, tool server on GPU1, proxy on CPU (Φ description
# rewrite + output-length manipulation happen at the proxy). Uses the 2 free GPUs on
# kn069; leaves GPU2/3 (GSS/OPSD) untouched. Login-node tmux holds the srun clients.
set -euo pipefail
export SLURM_CONF=${SLURM_CONF:-/cm/shared/apps/slurm/var/etc/killarney/slurm.conf}
S=/project/6101776/xzhan576/gta2-envlab/scripts
LOGD=/datasets/omni_pretraining/gta2/service_logs; mkdir -p "$LOGD"
JOBID=${GTA_JOBID:?set GTA_JOBID}
MODEL=${PHI_MODEL:-Qwen2.5-7B-Instruct}; NAME=${PHI_NAME:-qwen2.5-7b-instruct}
LANE=${PHI_LANE:-phi7}

COMMON="export SLURM_CONF=$SLURM_CONF; \
export GTA_MODEL_PATH=/datasets/omni_pretraining/gta2/models/$MODEL GTA_MODEL_NAME=$NAME; \
export GTA_LLM_GPUS=0 GTA_LLM_PORT=12580 GTA_KV_FRAC=0.5 \
       GTA_TOOL_GPU=1 GTA_TOOL_PORT=16181 GTA_PROXY_PORT=16281 GTA_POISON=0"

for s in llm tool proxy; do tmux kill-session -t ${LANE}_$s 2>/dev/null || true; done
tmux new-session -d -s ${LANE}_llm   "$COMMON; srun --overlap --jobid=$JOBID bash $S/start_llm.sh 2>&1 | tee -a $LOGD/${LANE}_llm.log"
tmux new-session -d -s ${LANE}_tool  "$COMMON; srun --overlap --jobid=$JOBID bash $S/start_toolserver.sh 2>&1 | tee -a $LOGD/${LANE}_tool.log"
tmux new-session -d -s ${LANE}_proxy "$COMMON; srun --overlap --jobid=$JOBID bash $S/start_proxy.sh 2>&1 | tee -a $LOGD/${LANE}_proxy.log"
echo "launched Φ stack lane '$LANE' ($NAME) on job $JOBID; logs $LOGD/${LANE}_*.log"
tmux ls | grep ${LANE}
