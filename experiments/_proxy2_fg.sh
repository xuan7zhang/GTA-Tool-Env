#!/bin/bash
# foreground proxy #2 (held by srun in tmux). 16282 -> tool server 16182.
LOG=/datasets/omni_pretraining/gta2/results/rr_stack
exec env PYTHONUNBUFFERED=1 /datasets/omni_pretraining/gta2/envs/agentlego/bin/python \
  /project/6101776/xzhan576/gta2-envlab/proxy/proxy.py --port 16282 \
  --upstream http://127.0.0.1:16182 --log "$LOG/proxy_calls.jsonl"
