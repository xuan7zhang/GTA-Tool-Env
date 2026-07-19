#!/bin/bash
# Morning status: heartbeat + per-job done/failed/skipped + flags + last log lines + GPU.
cd "$(dirname "$0")"
echo "===== HEARTBEAT ====="
cat logs/heartbeat.txt 2>/dev/null || echo "(none yet)"
echo; echo "===== ORCHESTRATOR ====="
if [ -f logs/orchestrator.pid ]; then
  pid=$(cat logs/orchestrator.pid)
  kill -0 "$pid" 2>/dev/null && echo "alive (pid $pid)" || echo "not running (pid $pid)"
fi
echo; echo "===== JOBS (state.json) ====="
python - <<'PY'
import json, pathlib
p = pathlib.Path("state.json")
if not p.exists():
    print("(no state.json yet)"); raise SystemExit
s = json.loads(p.read_text())
for b in ("completed", "failed", "skipped"):
    d = s.get(b, {})
    print(f"{b} ({len(d)}):")
    for jid, info in d.items():
        m = info.get("metrics", {})
        acc = m.get("mean_acc")
        print(f"  {jid}" + (f"  mean_acc={acc}" if acc is not None else "") +
              (f"  {info.get('error','')}" if b == "failed" else ""))
print("flags:", s.get("flags", {}))
PY
echo; echo "===== LAST LOG ====="
tail -n 15 logs/orchestrator.out 2>/dev/null
echo; echo "===== GPU ====="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"
