#!/bin/bash
# Score TGB-v4 with one model, waiting for a GPU to appear.
#
#   ./tgb/overnight_v4.sh <tag> <model-dir-name>
#
# Each lane owns its own wait loop. A parent that polls and then spawns does
# not survive this environment -- three earlier attempts died while sleeping
# between launches -- so there is no parent: every lane is launched separately
# and blocks on its own.
#
# Three failure modes this has already hit, all now handled:
#   * jobid empty  -> srun exits instantly with "Invalid job id"; a whole night
#                     was lost to four lanes failing in under a second
#   * GPU chosen outside the step -> srun re-enumerates devices, so a fixed
#                     index lands on a busy card; the pick happens inside
#   * picker returns nothing -> CUDA_VISIBLE_DEVICES= means "no GPUs" to vLLM,
#                     which reports "No CUDA GPUs are available"; checked
#
# score_dl is chunked and resumable, so every retry continues from the last
# flush rather than restarting.
set -u
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/killarney/slurm.conf
TAG=$1
MODEL=$2
LAB=/project/6101776/xzhan576/gta2-envlab
M=/datasets/omni_pretraining/gta2/models
T4=/datasets/omni_pretraining/gta2/results/taco/tgb4
PY=/project/aip-xli135/xzhan576/OPSD/.venv/bin/python
NEED=${NEED_MB:-22000}
# vLLM's gpu_memory_utilization is a fraction of the WHOLE card, and the
# weights come out of it: 14B needs ~28 GB, which is 35% of an 80 GB card, so a
# flat 0.25 leaves nothing for the KV cache and the engine dies with "No
# available memory for the cache blocks". 87 retries burned on exactly that.
UTIL=${GD_UTIL_OVERRIDE:-0.25}
# tensor-parallel width. 14B's weights alone match the free space on a borrowed
# 80 GB card, so no single-card utilisation leaves room for a KV cache; TP=2
# halves the weights per card and makes it fit.
TP=${GD_TP:-1}
# jobs this lane is allowed to borrow GPUs from, in order of preference
ALLOWED="l2 interactive"

log() { echo "[$(date +%H:%M:%S)] $*"; }

for attempt in $(seq 1 240); do
  JOB=""; NODE=""
  for name in $ALLOWED; do
    J=$(squeue -u "$USER" -h -t RUNNING -o "%i %j" | awk -v n="$name" '$2==n{print $1; exit}')
    [ -n "$J" ] || continue
    free=$(srun --overlap --jobid="$J" --ntasks=1 "$LAB/tgb/pick_gpu.sh" "$NEED" "$TP" 2>/dev/null)
    if [ -n "$free" ]; then
      JOB=$J; NODE=$name; break
    fi
  done

  if [ -z "$JOB" ]; then
    log "$TAG: no job with a $((NEED / 1000))GB-free GPU; retry $attempt"
    sleep 300
    continue
  fi

  log "$TAG: using job $JOB ($NODE)"
  srun --overlap --jobid="$JOB" --nodes=1 --ntasks=1 bash -c "
    # GD_GPU pins this lane to a card. The picker always returns the lowest
    # free index, so four lanes started together all choose GPU 0 and then
    # starve each other: the third one read \"Free memory on device
    # (14.91/44.39 GiB)\" on a card it had measured as empty seconds earlier.
    # With one lane per card there is no race and no need to over-reserve.
    g=\${GD_GPU:-\$($LAB/tgb/pick_gpu.sh $NEED $TP)}
    [ -z \"\$g\" ] && { echo 'gpu taken between check and launch'; exit 9; }
    # Derive gpu_memory_utilization from the card actually picked instead of
    # passing a fixed fraction. The fraction is of the WHOLE card and the
    # weights come out of it, so the safe value depends on card size -- and
    # this cluster is heterogeneous: kn081 has 80 GB cards, kn071 has 46 GB.
    # A flat 0.30 sized for 80 GB gives a 7B model 13.8 GB on a 46 GB card,
    # less than its 15 GB of weights. That mistake cost a night at 0.25 on
    # 14B, then repeated itself on the 7B lanes.
    u=\$(nvidia-smi --query-gpu=index,memory.total,memory.used \
          --format=csv,noheader,nounits |
        awk -F', ' -v sel=\"\$g\" -v cap=$UTIL 'BEGIN{n=split(sel,a,\",\")
            for(i=1;i<=n;i++) want[a[i]+0]=1}
          want[\$1+0]{f=\$2-\$3; if(m==\"\"||f<m){m=f;t=\$2}}
          END{if(t==\"\"){print cap; exit} v=(m-2048)/t
              if(v>0.85)v=0.85; if(v<0.10)v=0.10; printf \"%.2f\", v}')
    echo \"[\$(date +%H:%M:%S)] $TAG on GPU \$g, util \$u (tp=$TP)\"
    export CUDA_VISIBLE_DEVICES=\$g GD_TAG=$TAG TGB_V4=1 GD_UTIL=\$u GD_TP=$TP
    export GD_MODEL=$M/$MODEL HF_HOME=/datasets/omni_pretraining/gta2/hf_home
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    cd $LAB && $PY -m tgb.score_dl --tasks $T4/tgb_tasks.json --out $T4 --chunk 200
  "
  rc=$?
  n=$(wc -l < "$T4/tgb_scored_$TAG.jsonl" 2>/dev/null || echo 0)
  log "$TAG: attempt $attempt rc=$rc, $n/4000 scored"
  [ "$rc" -eq 0 ] && { log "$TAG: done"; break; }
  sleep 300
done
