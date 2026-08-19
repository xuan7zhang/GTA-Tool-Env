#!/bin/bash
# Greedy backward elimination on TGB-v4 for one model.
#
#   GD_GPU=0 ./tgb/greedy_lane_v4.sh 7b Qwen2.5-7B-Instruct
#
# This is the competitor to per-task likelihood selection, and it is the
# expensive one: backward elimination over a 15-tool menu visits up to
# 15+14+...+1 = 120 candidate masks, each executed over the 400-task train
# split, so ~48k generations before a single test task is answered.
#
# greedy_env checkpoints after every round -- three earlier runs were killed
# mid-search and lost all of it -- so a restarted lane resumes from the last
# completed round rather than the full menu.
set -u
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/killarney/slurm.conf
TAG=$1
MODEL=$2
LAB=/project/6101776/xzhan576/gta2-envlab
M=/datasets/omni_pretraining/gta2/models
T4=/datasets/omni_pretraining/gta2/results/taco/tgb4
PY=/project/aip-xli135/xzhan576/OPSD/.venv/bin/python
ALLOWED=${GD_JOBS:-"l2 interactive"}
GPU=${GD_GPU:-}
TRAIN=${GD_TRAIN:-400}

log() { echo "[$(date +%H:%M:%S)] $*"; }

for attempt in $(seq 1 96); do
  JOB=""
  for name in $ALLOWED; do
    J=$(squeue -u "$USER" -h -t RUNNING -o "%i %j" | awk -v n="$name" '$2==n{print $1; exit}')
    [ -n "$J" ] && { JOB=$J; break; }
  done
  if [ -z "$JOB" ]; then
    log "$TAG: no borrowable job; retry $attempt"; sleep 300; continue
  fi

  srun --overlap --jobid="$JOB" --nodes=1 --ntasks=1 bash -c "
    g=$GPU
    [ -z \"\$g\" ] && g=\$($LAB/tgb/pick_gpu.sh 22000 1)
    [ -z \"\$g\" ] && { echo 'no free GPU'; exit 9; }
    u=\$(nvidia-smi --query-gpu=index,memory.total,memory.used \
          --format=csv,noheader,nounits |
        awk -F', ' -v sel=\"\$g\" 'BEGIN{n=split(sel,a,\",\")
            for(i=1;i<=n;i++) want[a[i]+0]=1}
          want[\$1+0]{f=\$2-\$3; if(m==\"\"||f<m){m=f;t=\$2}}
          END{if(t==\"\"){print 0.55; exit} v=(m-2048)/t
              if(v>0.85)v=0.85; if(v<0.10)v=0.10; printf \"%.2f\", v}')
    echo \"[\$(date +%H:%M:%S)] $TAG greedy on GPU \$g, util \$u\"
    export CUDA_VISIBLE_DEVICES=\$g GD_TAG=$TAG GD_UTIL=\$u
    export GD_MODEL=$M/$MODEL HF_HOME=/datasets/omni_pretraining/gta2/hf_home
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    cd $LAB && $PY -m tgb.greedy_env --tasks $T4/tgb_tasks.json --out $T4 \
        --v4 --train $TRAIN --direction backward
  "
  rc=$?
  log "$TAG: attempt $attempt rc=$rc"
  [ "$rc" -eq 0 ] && { log "$TAG: done"; break; }
  sleep 180
done
