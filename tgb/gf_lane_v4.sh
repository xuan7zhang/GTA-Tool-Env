#!/bin/bash
# Gold-free scoring for one model, borrowing a GPU from a running job.
#
#   GD_GPU=0 ./tgb/gf_lane_v4.sh 7b Qwen2.5-7B-Instruct
#
# The old gf_lane.sh picked a GPU with "used < 3000", which rejects every card
# on a shared node, and passed no utilisation at all -- vLLM's 0.85 default is
# a fraction of the WHOLE card and dies wherever a neighbour is resident. Both
# are fixed the same way overnight_v4.sh fixes them: pin the card with GD_GPU,
# derive the fraction from what is actually free on it.
#
# Unlike score_dl, goldfree is NOT resumable -- it writes one JSON at the end.
# A lane that dies mid-run has to redo the model, so retries matter here.
set -u
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/killarney/slurm.conf
TAG=$1
MODEL=$2
LAB=/project/6101776/xzhan576/gta2-envlab
M=/datasets/omni_pretraining/gta2/models
T4=/datasets/omni_pretraining/gta2/results/taco/tgb4
PY=/project/aip-xli135/xzhan576/OPSD/.venv/bin/python
ALLOWED=${GF_JOBS:-"l2 interactive"}
GPU=${GD_GPU:-}

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
    g=\${GD_GPU_INNER:-$GPU}
    [ -z \"\$g\" ] && g=\$($LAB/tgb/pick_gpu.sh 22000 1)
    [ -z \"\$g\" ] && { echo 'no free GPU'; exit 9; }
    u=\$(nvidia-smi --query-gpu=index,memory.total,memory.used \
          --format=csv,noheader,nounits |
        awk -F', ' -v sel=\"\$g\" 'BEGIN{n=split(sel,a,\",\")
            for(i=1;i<=n;i++) want[a[i]+0]=1}
          want[\$1+0]{f=\$2-\$3; if(m==\"\"||f<m){m=f;t=\$2}}
          END{if(t==\"\"){print 0.55; exit} v=(m-2048)/t
              if(v>0.85)v=0.85; if(v<0.10)v=0.10; printf \"%.2f\", v}')
    echo \"[\$(date +%H:%M:%S)] $TAG gold-free on GPU \$g, util \$u\"
    export CUDA_VISIBLE_DEVICES=\$g GD_TAG=$TAG TGB_V4=1 GD_UTIL=\$u
    export GD_MODEL=$M/$MODEL HF_HOME=/datasets/omni_pretraining/gta2/hf_home
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    cd $LAB && $PY -m tgb.goldfree --dir $T4 --v4 --k 4
  "
  rc=$?
  log "$TAG: attempt $attempt rc=$rc"
  [ "$rc" -eq 0 ] && { log "$TAG: done"; break; }
  sleep 180
done
