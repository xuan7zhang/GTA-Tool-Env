#!/bin/bash
# One TGB-v4 lane: score + greedy for a single model, retried until it exits 0.
#   ./tgb/lane.sh <jobname> <tag> <model-dir-name> <tasks-dir>
# The GPU is chosen *inside* the srun step: srun re-enumerates devices, so an
# index passed in from outside lands on whatever is already busy -- a mistake
# that has cost two runs here. Both stages are resumable, so a retry continues
# rather than restarting.
set -u
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/killarney/slurm.conf
NAME=$1; TAG=$2; MODEL=$3; T=${4:-/datasets/omni_pretraining/gta2/results/taco/tgb4}
LAB=/project/6101776/xzhan576/gta2-envlab
PY=/project/aip-xli135/xzhan576/OPSD/.venv/bin/python
M=/datasets/omni_pretraining/gta2/models
JOB=$(squeue -u "$USER" -h -o "%i %j" | awk -v n="$NAME" '$2==n{print $1; exit}')
[ -z "$JOB" ] && { echo "FATAL: no running job named $NAME"; exit 1; }
for a in 1 2 3 4 5 6 7 8; do
  srun --overlap --jobid="$JOB" --nodes=1 --ntasks=1 bash -c "
    export CUDA_VISIBLE_DEVICES=\$(nvidia-smi --query-gpu=index,memory.used \
      --format=csv,noheader,nounits | awk -F', ' '\$2 < 3000 {print \$1; exit}')
    [ -z \"\$CUDA_VISIBLE_DEVICES\" ] && { echo 'no free GPU'; exit 9; }
    echo \"[$TAG] gpu \$CUDA_VISIBLE_DEVICES\"
    export GD_TAG=$TAG TGB_V4=1 GD_MODEL=$M/$MODEL
    export HF_HOME=/datasets/omni_pretraining/gta2/hf_home
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    cd $LAB
    $PY -m tgb.score_dl --tasks $T/tgb_tasks.json --out $T --chunk 200 && \
    $PY -m tgb.greedy_env --tasks $T/tgb_tasks.json --out $T --train 400 --v4"
  rc=$?
  echo "[lane] $TAG attempt $a rc=$rc"
  [ $rc -eq 0 ] && break
  sleep 45
done
echo "[lane] $TAG done"
