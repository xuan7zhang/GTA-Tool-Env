#!/bin/bash
# Preflight, then nohup-detach the orchestrator so it survives SSH disconnect.
#   ./launch.sh            fresh run
#   ./launch.sh --resume   continue from state.json
set -uo pipefail
cd "$(dirname "$0")"
RESUME="${1:-}"

echo "=== preflight ==="
command -v vllm >/dev/null || { echo "FAIL: vllm not on PATH"; exit 1; }
python -c "import yaml" 2>/dev/null || { echo "FAIL: pyyaml missing"; exit 1; }
nvidia-smi -L >/dev/null 2>&1 || { echo "FAIL: no GPU visible"; exit 1; }
# tool server + proxy must already be up (probes and evals go through the proxy)
PROXY=$(python -c "import yaml;print(yaml.safe_load(open('run_config.yaml'))['global']['proxy_url'])")
curl -s -o /dev/null -w '' --max-time 5 "$PROXY/openapi.json" || { echo "FAIL: proxy $PROXY not reachable"; exit 1; }
echo "  vllm/PATH ok, GPU visible, proxy reachable"

echo "=== dry-run gate ==="
python orchestrate.py --config run_config.yaml --dry-run || { echo "FAIL: dry-run failed"; exit 1; }
python test_gate.py || { echo "FAIL: unit tests"; exit 1; }

echo "=== launch (detached) ==="
mkdir -p logs
nohup python orchestrate.py --config run_config.yaml $RESUME > logs/orchestrator.out 2>&1 &
echo $! > logs/orchestrator.pid
echo "  started pid $(cat logs/orchestrator.pid); logs/orchestrator.out ; check with ./check.sh"
