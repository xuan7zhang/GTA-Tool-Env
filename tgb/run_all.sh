#!/bin/bash
# Score TGB with the whole model fleet, one model per idle L40S.
# Layout mirrors the paper's model set: scale ladder (3B/7B/14B/32B) plus two
# other families (Llama, Mistral) and Qwen3 for the cross-generation check.
set -u
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/killarney/slurm.conf
LAB=/project/6101776/xzhan576/gta2-envlab
M=/datasets/omni_pretraining/gta2/models
LOG=/datasets/omni_pretraining/gta2/results/taco/tgb/logs
mkdir -p "$LOG"

launch() {  # node gpu tag model tp [script]
  local node=$1 gpu=$2 tag=$3 model=$4 tp=$5 script=${6:-score_dl}
  SCRIPT=$script nohup setsid "$LAB/tgb/run_score.sh" "$node" "$gpu" "$tag" \
    "$model" "$tp" > "$LOG/launch_${script}_${tag}.out" 2>&1 &
  echo "  launched $script $tag on $node:gpu$gpu"
}

launch kn053 0 7b        "$M/Qwen2.5-7B-Instruct"      1
launch kn053 1 14b       "$M/Qwen2.5-14B-Instruct"     1
launch kn053 2 llama8b   "$M/Llama-3.1-8B-Instruct"    1
launch kn053 3 mistral7b "$M/Mistral-7B-Instruct-v0.3" 1
launch kn064 0,1 32b     "$M/Qwen2.5-32B-Instruct"     2
launch kn064 2 3b        "$M/Qwen2.5-3B-Instruct"      1
launch kn064 3 qwen3-8b  "$M/Qwen3-8B"                 1
echo "all launched; logs in $LOG"
