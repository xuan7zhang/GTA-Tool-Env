#!/bin/bash
# Score TGB with one model on one GPU of an already-running interactive job.
#   ./tgb/run_score.sh <node> <gpu> <tag> <model-dir> [tp] [extra args...]
# CUDA_VISIBLE_DEVICES is set *inside* the srun payload on purpose: srun
# rewrites it on the way in, so setting it outside has no effect.
set -u
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/killarney/slurm.conf
NODE=$1; GPU=$2; TAG=$3; MODEL=$4; TP=${5:-1}; shift 5 2>/dev/null || shift 4
PY=/project/aip-xli135/xzhan576/OPSD/.venv/bin/python
LAB=/project/6101776/xzhan576/gta2-envlab
TGB=/datasets/omni_pretraining/gta2/results/taco/tgb
JOB=$(squeue -u "$USER" -h -o "%i %N" | awk -v n="$NODE" '$2==n{print $1; exit}')
[ -z "$JOB" ] && { echo "no job on $NODE"; exit 1; }
mkdir -p "$TGB/logs"
# GPU='auto' picks a device with enough free memory *inside* the step. srun
# re-enumerates GPUs when other steps already hold some, so a fixed index
# handed in from outside can land on a busy card ("Free memory on device
# 4.65/44.39 GiB is less than desired GPU memory utilization").
srun --overlap --jobid="$JOB" --nodes=1 --ntasks=1 bash -c "
  if [ '$GPU' = auto ]; then
    export CUDA_VISIBLE_DEVICES=\$(nvidia-smi --query-gpu=index,memory.used \
      --format=csv,noheader,nounits | awk -F', ' '\$2 < 2000 {print \$1; exit}')
    echo \"[auto] picked GPU \$CUDA_VISIBLE_DEVICES\"
  else
    export CUDA_VISIBLE_DEVICES=$GPU
  fi
  export GD_TAG=$TAG GD_MODEL=$MODEL GD_TP=$TP GD_UTIL=\${GD_UTIL:-0.85}
  export HF_HOME=/datasets/omni_pretraining/gta2/hf_home
  # Mistral's sentencepiece proto needs the pure-Python protobuf backend in
  # this venv; harmless for the other models.
  export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
  cd $LAB && $PY -m tgb.${SCRIPT:-score_dl} --tasks $TGB/tgb_tasks.json --out $TGB $*
" 2>&1 | tee "$TGB/logs/${SCRIPT:-score_dl}_${TAG}.log"
