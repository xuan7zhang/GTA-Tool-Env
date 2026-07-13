# versions.lock — pins and deviations from upstream README

Full `pip freeze` per env: `/datasets/omni_pretraining/gta2/envs/{lmdeploy,agentlego,opencompass}.freeze.txt`
(written by the setup scripts on success). GTA repo commit: see `GTA/.git` (shallow clone of open-compass/GTA master, 2026-07-04).

## Environments (conda prefixes under /datasets/omni_pretraining/gta2/envs)

| env | python | key pins |
|---|---|---|
| lmdeploy | 3.10 | lmdeploy (latest PyPI at build), huggingface_hub[cli], hf_transfer |
| agentlego | 3.11.9 | torch 2.1.2+cu121, transformers 4.40.1, mmcv 2.1.0 (mim), mmdet 3.3.0, mmpretrain 1.2.0, easyocr 1.7.1, diffusers 0.27.2 |
| opencompass | 3.10 | torch 2.1.2+cu121 (pip), transformers 4.40.1, huggingface_hub 0.25.2, **lagent 0.2.3** (0.1.2 from agent.txt is incompatible with the vendored LagentTool wrapper — abstract `__call__` → NotImplementedError), sentence-transformers 2.2.2 |

## Deviations from upstream README (all deliberate, all recorded)

1. **Dataset**: README's `gta_dataset_v2.zip` release link 404s (no such asset
   exists in any release; HF has only the original `Jize1/GTA`). GTA-Atomic =
   original `gta_dataset` (v0.1.0, 229 tasks, toolmeta+images+reference chains).
   Consequently we use `benchmark_toollist.txt` (14 tools) and the
   `GTABenchEvaluator` step-wise/end-to-end metrics — this is the benchmark the
   public leaderboard rows are computed on. The shipped `eval_gta_bench_v2.py`
   is a GTA-Workflow (end.json + GPT-judge) config, out of scope.
2. **requirements_all.txt filtered**: dropped `mkl-fft`, `mkl-random`,
   `mkl-service` (conda-channel artifacts, unreliable on PyPI, unused).
3. **requirements_gta_v2.txt not installed verbatim**: it is a raw `pip freeze`
   from a ROS2 machine (ament-*, *-msgs, cartographer-* are not PyPI packages).
   Curated delta installed instead (openpyxl, python-docx, python-pptx, PyPDF2,
   reportlab, noisereduce, soundfile, librosa, moviepy 1.0.3, openai-whisper,
   beautifulsoup4, curl_cffi, z3-solver). Only relevant to v2 tools, which the
   Atomic baseline does not exercise.
4. **opencompass torch**: pip cu121 wheels (torch 2.1.2) instead of conda
   `pytorch`/`pytorch-cuda` channel — same version, reproducible.
5. **transformers `_supports_sdpa` patch** (README step): applied by
   `setup_env_agentlego.sh` via sed on `modeling_utils.py` in the agentlego env.
6. **Compute Canada pip config neutralized** in all scripts
   (`PIP_CONFIG_FILE=/dev/null`, `unset PYTHONPATH`) — their constraints file
   breaks PyPI resolution (`lmdeploy: versions none`).
7. **Benchmark-code patches (ours, minimal, for logging/robustness)** in the
   vendored opencompass:
   - `models/lagent.py`: keep RemoteTool handles + `set_task_id()` (X-GTA-Task-Id
     header per sample for proxy logs); skip per-task resource tools missing
     from the deployed set (mask variants) instead of KeyError; append
     `GTA_EXTRA_TOOLS` to the per-task toolset (compose/distractor variants).
   - `openicl/icl_inferencer/icl_agent_inferencer.py`: call `set_task_id(index)`
     at the top of `infer_every` / `infer_every_with_gt`.
   - `configs/gta_atomic_env.py`: our env-var-driven eval config (new file).
8. **Model**: Qwen2.5-7B-Instruct (leaderboard row exists → sanity check
   anchor), served by lmdeploy on 1× L40S, `--cache-max-entry-count 0.5`.
9. **pip downgraded to <25 in agentlego/opencompass envs** before the
   `pip install -e .` steps: agentlego's pyproject pins `setuptools>=62.6,<64`
   (no PEP 660 `build_editable` hook) and pip>=25.3 removed legacy
   `setup.py develop` (`scripts/finish_envs.sh`).
10. **Builds must run on a compute node**: concurrent pip installs on klogin
    nodes die with SIGBUS (Bus error). Also, a detached tmux started inside a
    short `srun --overlap` step is killed when the step exits (slurmstepd
    cgroup cleanup) — keep the srun alive for the duration instead.
11. **Host**: brief said kn064, but kn064's 4 GPUs run the user's SDPO RL
    training; deployed on the idle kn066 (job `l1`, same 4× L40S). Override
    with `GTA_JOB_NAME`/`GTA_JOBID` in `scripts/on_node.sh`.
12. **Config dump quirk**: OpenCompass re-dumps the eval config and re-parses
    it; bare modules in the config namespace (e.g. `import os` left at top
    level) serialize as invalid syntax. `configs/gta_atomic_env.py` ends with
    `del os` for this reason.
13. **Services & long evals run as login-node tmux sessions holding the srun
    client** (`scripts/services_up.sh`): sessions `gta_llm/gta_tool/gta_proxy`,
    logs in `$GTA_BIG/service_logs/`. Anything parented to the assistant
    session dies on its restart; login-node tmux survives.
14. **HF cache vars fully overridden** in `common_env.sh`: the user's ~/.bashrc
    exports HF_HUB_CACHE/TRANSFORMERS_CACHE etc. pointing at ~/scratch/huggingface
    (scratch quota full → `Errno 122` at tool-server startup), and those take
    precedence over HF_HOME. All five vars now point into $GTA_BIG/hf_home.
