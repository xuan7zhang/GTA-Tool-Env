#!/bin/bash
# Algorithm 1 on TGB-v4 for one model. GD_GPU=0 ./tgb/algo1_lane.sh 7b Qwen2.5-7B-Instruct split union
set -u
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/killarney/slurm.conf
TAG=$1; MODEL=$2; LEVEL=$3; BASE=$4
LAB=/project/6101776/xzhan576/gta2-envlab
M=/datasets/omni_pretraining/gta2/models
T4=/datasets/omni_pretraining/gta2/results/taco/tgb4
PY=/project/aip-xli135/xzhan576/OPSD/.venv/bin/python
GPU=${GD_GPU:-}
for attempt in $(seq 1 48); do
  J=$(squeue -u "$USER" -h -t RUNNING -o "%i %j" | awk '$2=="l2"{print $1; exit}')
  [ -z "$J" ] && { echo "no l2; retry $attempt"; sleep 300; continue; }
  srun --overlap --jobid="$J" --nodes=1 --ntasks=1 bash -c "
    g=$GPU; [ -z \"\$g\" ] && g=\$($LAB/tgb/pick_gpu.sh 22000 1)
    [ -z \"\$g\" ] && { echo 'no free GPU'; exit 9; }
    # Weights are a hard floor the utilisation fraction cannot go under: 14B
    # needs ~28 GB, so a card measured with a 16 GB leftover on it yields
    # util 0.22 and a guaranteed OOM. Refuse and retry instead of launching
    # something that cannot start.
    read free tot <<< \$(nvidia-smi --query-gpu=index,memory.total,memory.used \
        --format=csv,noheader,nounits |
      awk -F', ' -v sel=\"\$g\" 'BEGIN{n=split(sel,a,\",\"); for(i=1;i<=n;i++) want[a[i]+0]=1}
        want[\$1+0]{f=\$2-\$3; if(m==\"\"||f<m){m=f;t=\$2}} END{print m, t}')
    if [ \"\${free:-0}\" -lt \"\${NEED_MB:-24000}\" ]; then
      echo \"GPU \$g has only \${free}MB free, need \${NEED_MB:-24000}; retry\"; exit 9
    fi
    u=\$(awk -v m=\"\$free\" -v t=\"\$tot\" 'BEGIN{v=(m-2048)/t
        if(v>0.85)v=0.85; if(v<0.10)v=0.10; printf \"%.2f\", v}')
    echo \"[\$(date +%H:%M:%S)] $TAG algo1 $LEVEL/$BASE on GPU \$g util \$u\"
    export CUDA_VISIBLE_DEVICES=\$g GD_TAG=$TAG GD_UTIL=\$u TGB_V4=1
    export GD_MODEL=$M/$MODEL HF_HOME=/datasets/omni_pretraining/gta2/hf_home
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    cd $LAB && $PY -m tgb.algo1 --dir $T4 --train 400 --level $LEVEL --base $BASE
  "
  rc=$?; echo "[$(date +%H:%M:%S)] $TAG $LEVEL/$BASE attempt $attempt rc=$rc"
  [ "$rc" -eq 0 ] && break; sleep 180
done
# NOTE on watching these lanes: count failures as a *delta*, not a total.
# A lane that failed 10 times fighting for a GPU and then recovered still has
# 10 "rc=1" lines in its log forever, so a `>= 3` rule fires on a healthy run.
# Watch (failures increasing) AND (no output file), or just watch the output.
