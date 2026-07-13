# GTA patches
GTA/ itself is a vendored clone of https://github.com/open-compass/GTA (master,
~2026-07-04), gitignored here (465MB). To reproduce: clone GTA, then overlay the
files below (they preserve their in-repo paths under patches/GTA/).

Modified files (all changes are experiment-gated; no-op for the unmodified baseline):
- opencompass/.../models/lagent.py            — GTA_EXTRA_TOOLS injection, skip-missing-resource-tools (mask), set_task_id header
- opencompass/.../datasets/gta_bench.py       — GTA_TASK_SUBSET slicing
- opencompass/.../icl_inferencer/icl_agent_inferencer.py — set_task_id(index) hooks
- opencompass/configs/gta_atomic_env.py       — our env-var-driven eval config (NEW)
- agentlego/.../tools/wrappers/lagent.py       — (if modified) lagent 0.2.3 compat

Deps: lagent==0.2.3 (NOT the repo-pinned 0.1.2, whose BaseAction.__call__ is
abstract and breaks the vendored LagentTool wrapper). See versions.lock.md.
