# gta2-envlab service + eval entry points.
# Services run ON the GPU node (kn064): use `make -C ... <target> ` from a shell
# on the node (tmux), or prefix with the srun helper:
#   scripts/on_node.sh make llm
SHELL := /bin/bash
S := scripts

llm:            ## serve Qwen2.5-7B-Instruct with LMDeploy (GPU $$GTA_LLM_GPUS)
	bash $(S)/start_llm.sh

toolserver:     ## AgentLego tool server, 14 GTA-Atomic tools (GPU $$GTA_TOOL_GPU)
	bash $(S)/start_toolserver.sh

proxy:          ## probe/noise proxy in front of the tool server (CPU)
	bash $(S)/start_proxy.sh

services:       ## all three, each in its own tmux session (run on the node)
	tmux new-session -d -s gta_llm  'bash $(S)/start_llm.sh 2>&1 | tee /tmp/gta_llm.log'
	tmux new-session -d -s gta_tool 'bash $(S)/start_toolserver.sh 2>&1 | tee /tmp/gta_tool.log'
	tmux new-session -d -s gta_proxy 'bash $(S)/start_proxy.sh 2>&1 | tee /tmp/gta_proxy.log'
	@echo "tmux sessions: gta_llm gta_tool gta_proxy"

baseline:       ## Task 0 baseline eval (both step+end modes)
	GTA_EVAL_MODES=step,end bash $(S)/run_eval.sh baseline

causal-b:       ## Task 1 run B (tool_free) — proxy mode switched via /proxy_config
	curl -s -X POST localhost:$${GTA_PROXY_PORT:-16281}/proxy_config -H 'Content-Type: application/json' \
	  -d '{"mode":"tool_free","log_path":"/datasets/omni_pretraining/gta2/results/causal_B_toolfree/proxy_calls.jsonl","run_meta":{"run_id":"causal_B_toolfree"}}'
	GTA_EVAL_MODES=end bash $(S)/run_eval.sh causal_B_toolfree

causal-c:       ## Task 1 run C (corrupt_output)
	curl -s -X POST localhost:$${GTA_PROXY_PORT:-16281}/proxy_config -H 'Content-Type: application/json' \
	  -d '{"mode":"corrupt_output","seed":0,"log_path":"/datasets/omni_pretraining/gta2/results/causal_C_corrupt/proxy_calls.jsonl","run_meta":{"run_id":"causal_C_corrupt"}}'
	GTA_EVAL_MODES=end bash $(S)/run_eval.sh causal_C_corrupt

pilot-sweep:    ## Task 2 pilot: leave-one-out masks x {passthrough,tool_free}
	bash $(S)/sweep.sh configs/manifest_pilot_loo.jsonl

aggregate:      ## one CSV row per run
	python3 analysis/aggregate_results.py --results /datasets/omni_pretraining/gta2/results --out analysis/sweep.csv

.PHONY: llm toolserver proxy services baseline causal-b causal-c pilot-sweep aggregate
