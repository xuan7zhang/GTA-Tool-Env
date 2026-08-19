#!/bin/bash
set -u
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/killarney/slurm.conf
NAME=$1; TAG=$2; MODEL=$3
LAB=/project/6101776/xzhan576/gta2-envlab
PY=/project/aip-xli135/xzhan576/OPSD/.venv/bin/python
M=/datasets/omni_pretraining/gta2/models
T=/datasets/omni_pretraining/gta2/results/taco/tgb4
JOB=$(squeue -u "$USER" -h -o "%i %j" | awk -v n="$NAME" '$2==n{print $1; exit}')
[ -z "$JOB" ] && { echo "FATAL: no job $NAME"; exit 1; }
for a in 1 2 3 4 5; do
  srun --overlap --jobid="$JOB" --nodes=1 --ntasks=1 bash -c "
    export CUDA_VISIBLE_DEVICES=\$(nvidia-smi --query-gpu=index,memory.used \
      --format=csv,noheader,nounits | awk -F', ' '\$2 < 3000 {print \$1; exit}')
    [ -z \"\$CUDA_VISIBLE_DEVICES\" ] && { echo 'no free GPU'; exit 9; }
    echo \"[$TAG] gpu \$CUDA_VISIBLE_DEVICES\"
    export GD_TAG=$TAG TGB_V4=1 GD_MODEL=$M/$MODEL
    export HF_HOME=/datasets/omni_pretraining/gta2/hf_home
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    cd $LAB && $PY -m tgb.goldfree --dir $T --v4 --k 4"
  rc=$?; echo "[gf] $TAG attempt $a rc=$rc"; [ $rc -eq 0 ] && break; sleep 45
done
