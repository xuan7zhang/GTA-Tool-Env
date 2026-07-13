#!/bin/bash
# foreground tool server #2 (held by srun in tmux). GPU1:16182, RegionRead macro.
exec env PYTHONUNBUFFERED=1 \
  GTA_TOOL_GPU=1 GTA_TOOL_PORT=16182 \
  GTA_MACRO=1 GTA_MACRO_NAMES="PerceiveAll RegionRead" \
  GTA_PROXY_URL_INTERNAL=http://127.0.0.1:16282 \
  bash /project/6101776/xzhan576/gta2-envlab/scripts/start_toolserver.sh
