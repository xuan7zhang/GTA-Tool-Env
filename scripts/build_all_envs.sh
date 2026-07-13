#!/bin/bash
# Serially (re)build the three conda envs; intended to run on a compute node
# (login-node builds hit Bus errors under concurrent pip load).
S="$(dirname "$0")"; B=/datasets/omni_pretraining/gta2
{
  echo "[build_all] host=$(hostname) start=$(date)"
  bash $S/setup_env_lmdeploy.sh    > $B/setup_lmdeploy.log    2>&1 && echo "lmdeploy OK"    || echo "lmdeploy FAIL"
  bash $S/setup_env_opencompass.sh > $B/setup_opencompass.log 2>&1 && echo "opencompass OK" || echo "opencompass FAIL"
  bash $S/setup_env_agentlego.sh   > $B/setup_agentlego.log   2>&1 && echo "agentlego OK"   || echo "agentlego FAIL"
  echo "[build_all] end=$(date)"
} > $B/build_all.log 2>&1
