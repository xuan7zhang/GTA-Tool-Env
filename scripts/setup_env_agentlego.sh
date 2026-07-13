#!/bin/bash
# Build the agentlego conda env (tool server), per GTA README with recorded deviations:
#  - requirements_all.txt filtered: mkl-fft/mkl-random/mkl-service are conda-channel
#    artifacts with no reliable PyPI wheels; not needed by agentlego.
#  - requirements_gta_v2.txt is a raw `pip freeze` from the authors' ROS machine
#    (ament-*, *-msgs, cartographer-* etc. are not on PyPI). We install the curated
#    delta of real PyPI deps needed by the v2 tools instead.
#  - transformers _supports_sdpa patch applied via sed (README step, line ~1279).
set -euxo pipefail
source "$(dirname "$0")/common_env.sh"
export CONDA_PKGS_DIRS=$GTA_BIG/conda_pkgs/agentlego
rm -rf $GTA_BIG/envs/agentlego && conda create -y -p $GTA_BIG/envs/agentlego python=3.11.9
conda activate $GTA_BIG/envs/agentlego

cd $GTA_REPO/agentlego
grep -vE '^(mkl-fft|mkl-random|mkl-service)' requirements_all.txt > /tmp/req_all_filtered_$$.txt
pip install -r /tmp/req_all_filtered_$$.txt
rm /tmp/req_all_filtered_$$.txt

# curated v2-tool deps (file gen/readers, audio, video) — real PyPI subset of requirements_gta_v2.txt
pip install openpyxl python-docx python-pptx PyPDF2 reportlab noisereduce soundfile \
    librosa moviepy==1.0.3 imageio-ffmpeg openai-whisper beautifulsoup4 curl_cffi \
    func-timeout z3-solver

pip install -e .
mim install mmengine
mim install mmcv==2.1.0

# README patch: _supports_sdpa False -> True in transformers modeling_utils.py
MU=$(python -c "import transformers, pathlib; print(pathlib.Path(transformers.__file__).parent / 'modeling_utils.py')")
grep -n "_supports_sdpa = False" "$MU"
sed -i '0,/_supports_sdpa = False/s//_supports_sdpa = True/' "$MU"
grep -n "_supports_sdpa = True" "$MU" | head -3

pip freeze > $GTA_BIG/envs/agentlego.freeze.txt
echo "AGENTLEGO ENV DONE"
