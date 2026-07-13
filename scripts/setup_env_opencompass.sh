#!/bin/bash
# Build the opencompass conda env (agent loop + eval).
# Deviation from README: torch installed via pip (cu121 wheels) instead of the
# conda pytorch channel — same versions, much faster and reproducible.
set -euxo pipefail
source "$(dirname "$0")/common_env.sh"
export CONDA_PKGS_DIRS=$GTA_BIG/conda_pkgs/opencompass
rm -rf $GTA_BIG/envs/opencompass && conda create -y -p $GTA_BIG/envs/opencompass python=3.10
conda activate $GTA_BIG/envs/opencompass

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

cd $GTA_REPO/agentlego && pip install -e .
cd $GTA_REPO/opencompass && pip install -e .
# lagent 0.2.3, NOT 0.1.2: the vendored agentlego LagentTool wrapper implements
# run() and relies on 0.2.x BaseAction.__call__ dispatch; 0.1.2's __call__ is
# abstract -> NotImplementedError on every tool call. (requirements/agent.txt's
# 0.1.2 pin predates the GTA-2 vendored code.)
pip install huggingface_hub==0.25.2 transformers==4.40.1 lagent==0.2.3 importlib_metadata

pip freeze > $GTA_BIG/envs/opencompass.freeze.txt
echo "OPENCOMPASS ENV DONE"
