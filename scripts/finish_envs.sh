#!/bin/bash
# Finish the agentlego + opencompass envs after the pip-26 editable-install
# failure: agentlego pins setuptools<64 (no PEP 660 build_editable hook) and
# pip>=25.3 removed legacy `setup.py develop`. Fix: downgrade pip to 24.x in
# both envs before the -e installs (recorded deviation).
set -euxo pipefail
source "$(dirname "$0")/common_env.sh"

# ---- agentlego env: remaining steps ----
conda activate $GTA_BIG/envs/agentlego
pip install 'pip<25'
cd $GTA_REPO/agentlego
pip install -e .
mim install mmengine
mim install mmcv==2.1.0
MU=$(python -c "import transformers, pathlib; print(pathlib.Path(transformers.__file__).parent / 'modeling_utils.py')")
grep -n "_supports_sdpa = False" "$MU" || true
sed -i '0,/_supports_sdpa = False/s//_supports_sdpa = True/' "$MU"
grep -n "_supports_sdpa = True" "$MU" | head -3
pip freeze > $GTA_BIG/envs/agentlego.freeze.txt
conda deactivate

# ---- opencompass env: remaining steps ----
conda activate $GTA_BIG/envs/opencompass
pip install 'pip<25'
cd $GTA_REPO/agentlego && pip install -e .
cd $GTA_REPO/opencompass && pip install -e .
pip install huggingface_hub==0.25.2 transformers==4.40.1 lagent==0.1.2
pip freeze > $GTA_BIG/envs/opencompass.freeze.txt

echo "FINISH ENVS DONE"
