# Shared paths for gta2-envlab. Source this from every script.
export GTA_BIG=/datasets/omni_pretraining/gta2          # big volume: envs, models, data, results
export GTA_LAB=/project/6101776/xzhan576/gta2-envlab    # code
export GTA_REPO=$GTA_LAB/GTA
export PIP_CACHE_DIR=$GTA_BIG/pip_cache
# Override ALL HF cache vars: the user's ~/.bashrc points them at
# ~/scratch/huggingface (scratch is 100% full -> Errno 122 quota), and
# HF_HUB_CACHE/TRANSFORMERS_CACHE take precedence over HF_HOME.
export HF_HOME=$GTA_BIG/hf_home
export HF_HUB_CACHE=$GTA_BIG/hf_home/hub
export HF_DATASETS_CACHE=$GTA_BIG/hf_home/datasets
export HF_ASSETS_CACHE=$GTA_BIG/hf_home/assets
export TRANSFORMERS_CACHE=$GTA_BIG/hf_home/hub
# Neutralize Compute Canada python defaults: their pip.conf applies a
# constraints file + local wheelhouse (breaks PyPI resolution, e.g. lmdeploy
# "versions: none"), and PYTHONPATH leaks CC site-packages into conda envs.
export PIP_CONFIG_FILE=/dev/null
unset PYTHONPATH
# Keep all model/weight caches off the small home quota:
export TORCH_HOME=$GTA_BIG/torch_home          # torch.hub checkpoints (GLIP etc.)
export XDG_CACHE_HOME=$GTA_BIG/xdg_cache       # mim/mmengine/easyocr caches
source $HOME/miniconda3/etc/profile.d/conda.sh

# GPU assignment (override via env)
export GTA_LLM_GPUS=${GTA_LLM_GPUS:-0}        # LLM serving (7B fits on one L40S)
export GTA_TOOL_GPU=${GTA_TOOL_GPU:-2}        # AgentLego tool server
export GTA_LANE2_LLM_GPU=${GTA_LANE2_LLM_GPU:-1}   # second sweep lane LLM
export GTA_LANE2_TOOL_GPU=${GTA_LANE2_TOOL_GPU:-3} # second sweep lane tool server

export GTA_LLM_PORT=${GTA_LLM_PORT:-12580}
export GTA_TOOL_PORT=${GTA_TOOL_PORT:-16181}
export GTA_PROXY_PORT=${GTA_PROXY_PORT:-16281}
