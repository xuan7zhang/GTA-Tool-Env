#!/bin/bash
# Build the lmdeploy conda env (LLM serving).
set -euxo pipefail
source "$(dirname "$0")/common_env.sh"
export CONDA_PKGS_DIRS=$GTA_BIG/conda_pkgs/lmdeploy
rm -rf $GTA_BIG/envs/lmdeploy && conda create -y -p $GTA_BIG/envs/lmdeploy python=3.10
conda activate $GTA_BIG/envs/lmdeploy
pip install lmdeploy 'huggingface_hub[cli]' hf_transfer
pip freeze > $GTA_BIG/envs/lmdeploy.freeze.txt
echo "LMDEPLOY ENV DONE"
