"""Tool-server reverse proxy — the single injection point for causal probes and noise.

Sits between the Lagent/OpenCompass agent and the AgentLego tool server.
The agent's `tool_server` config points here; we forward to the real server.

Modes (global default + per-tool overrides), set via config file and/or the
/config endpoint at runtime:

  passthrough     forward verbatim (default)
  tool_free       every tool call returns the fixed unavailable-string
                  (tool-free counterfactual, arXiv 2606.02357 ablation)
  corrupt_output  forward, then corrupt the real return: images -> black image
                  of same size (VisualNeedle crop-black); text/JSON -> seeded
                  token shuffle / type-consistent random values (format-valid)
  format_only     alias of corrupt_output (2606.02357 decomposition)
  result_only     stub, currently behaves as passthrough (reserved)
  unreliable      per-tool Bernoulli(p_fail) failure with error message
  unavailable     like tool_free but for tools with missing API keys
                  (--no-external-api routing); logged distinctly

Masking: if config "mask" is a list of tool names, /openapi.json is filtered to
that subset, so RemoteTool.from_server() only builds the masked tool set —
mask variants then need no tool-server restart. Calls to masked tools 404.

Every proxied call is appended to JSONL (one line per call):
  ts, task_id (X-GTA-Task-Id header if present), turn (per-task counter),
  tool, mode, args (scalar params; file field names only), status,
  raw_return_hash, corrupted_return_hash, latency_s, seed_draw (unreliable)

Run (inside the agentlego conda env — has fastapi/uvicorn/httpx/PIL/numpy):
  python proxy.py --port 16281 --upstream http://127.0.0.1:16181 \
      --config proxy_config.json --log /path/to/proxy_calls.jsonl
"""
import argparse
import base64
import hashlib
import io
import json
import os
import random
import re
import threading
import time
from collections import defaultdict
from typing import Optional

import sys

import httpx
import uvicorn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from taco import transform as _taco  # noqa: E402  (TACO output intervention layer)
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

UNAVAILABLE_MSG = "Tool temporarily unavailable. Answer using your own reasoning."
FAIL_MSG = "Error: tool execution failed. Please try again or use another approach."

# openapi_url=None: FastAPI's built-in /openapi.json route would otherwise
# shadow our passthrough route and serve the proxy's own schema to RemoteTool.
app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
STATE = {
    "upstream": "http://127.0.0.1:16181",
    "mode": "passthrough",            # global default mode
    "per_tool_modes": {},              # {tool_name: mode}
    "mask": None,                      # None = all tools; list = allowed subset
    "unavailable_tools": [],           # tools routed to 'unavailable' (no API key)
    "p_fail": 0.1,                     # for unreliable mode
    "seed": 0,
    "log_path": "proxy_calls.jsonl",
    "run_meta": {},                    # free-form, echoed into every log line
    "phi_toolmeta": None,              # path to a Φ-variant toolmeta.json:
                                       # its descriptions overwrite openapi
                                       # summaries (the agent-visible text)
    # --- TACO (tool-attribute causal optimization) output intervention layer ---
    # {"default": {...spec...}, "per_tool": {tool: {...spec...}}, "exclude": [...]}
    # A spec is {"format": "F0".."F6",
    #            "length": {"level": "L0".."L5", "mechanism": ...} | null,
    #            "position": "front"|"middle"|"back" | null}.
    # None (default) => byte-identical behaviour to the pre-TACO proxy.
    "taco_spec": None,
    "taco_log_path": None,             # jsonl transformation log (Phase 1)
    # Per-task attribution fallback. The X-GTA-Task-Id header does not survive
    # the RemoteTool -> lagent wrapping in the chat() path (pre-existing: prior
    # runs' proxy logs also carry an empty task_id), so the agent additionally
    # publishes the current task to GTA_CURTASK_FILE. Evaluation is sequential
    # (--max-num-workers 1), which makes a single pointer file unambiguous.
    "curtask_file": None,
}
_openapi_cache = {"raw": None, "tool_output_types": {}}
_log_lock = threading.Lock()
_call_counter = defaultdict(int)       # per (task_id, tool) turn counter
_unreliable_counter = defaultdict(int) # per tool, for deterministic Bernoulli


def _load_config(path: str):
    with open(path) as f:
        cfg = json.load(f)
    STATE.update({k: v for k, v in cfg.items() if k in STATE})


def _log(rec: dict):
    rec["ts"] = time.time()
    rec["run_meta"] = STATE["run_meta"]
    with _log_lock:
        with open(STATE["log_path"], "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


async def _fetch_openapi() -> dict:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(STATE["upstream"].rstrip("/") + "/openapi.json")
    spec = r.json()
    # Map tool name -> output media/type hint from the response schema.
    types = {}
    for path, ops in spec.get("paths", {}).items():
        tool = path.strip("/")
        for op in ops.values():
            try:
                resp = op["responses"]["200"]["content"]["application/json"]["schema"]
                types[tool] = json.dumps(resp)
            except (KeyError, TypeError):
                types[tool] = ""
    _openapi_cache["raw"] = spec
    _openapi_cache["tool_output_types"] = types
    return spec


def _is_file_output(tool: str) -> bool:
    """True if the tool's declared 200-response contains base64 file content."""
    schema = _openapi_cache["tool_output_types"].get(tool, "")
    return "base64" in schema or "binary" in schema


def _mode_for(tool: str) -> str:
    if tool in STATE["unavailable_tools"]:
        return "unavailable"
    return STATE["per_tool_modes"].get(tool, STATE["mode"])


# ---------------- corruption helpers ----------------

def _black_image_like(b64: str) -> str:
    from PIL import Image
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    fmt = img.format or "PNG"
    black = Image.new(img.mode if img.mode in ("RGB", "L", "RGBA") else "RGB", img.size, 0)
    buf = io.BytesIO()
    black.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _shuffle_text(text: str, rng: random.Random) -> str:
    toks = text.split(" ")
    if len(toks) > 1:
        rng.shuffle(toks)
        return " ".join(toks)
    return text[::-1] if len(text) > 3 else text


def _corrupt_json_value(v, rng: random.Random):
    if isinstance(v, bool):
        return rng.random() < 0.5
    if isinstance(v, int):
        return rng.randint(0, max(10, abs(v) * 2 + 1))
    if isinstance(v, float):
        return rng.uniform(0, max(1.0, abs(v) * 2))
    if isinstance(v, str):
        return _shuffle_text(v, rng)
    if isinstance(v, list):
        return [_corrupt_json_value(x, rng) for x in v]
    if isinstance(v, dict):
        return {k: _corrupt_json_value(x, rng) for k, x in v.items()}
    return v


def _corrupt_payload(tool: str, body: bytes, rng: random.Random) -> bytes:
    """Corrupt a JSON response body, keeping it format-valid."""
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body  # not JSON; leave protocol intact
    if isinstance(payload, str):
        if _is_file_output(tool):
            # try image corruption; fall back to leaving bytes valid but blank
            try:
                payload = _black_image_like(payload)
            except Exception:
                try:
                    n = len(base64.b64decode(payload))
                    payload = base64.b64encode(b"\x00" * n).decode()
                except Exception:
                    pass
        else:
            # text payloads often embed numbers/coords: shuffle tokens AND
            # replace digit-runs with random same-length digits
            s = _shuffle_text(payload, rng)
            payload = re.sub(r"\d+", lambda m: str(rng.randint(0, 10 ** len(m.group()) - 1)), s)
    else:
        payload = _corrupt_json_value(payload, rng)
    return json.dumps(payload).encode()


# ---------------- endpoints ----------------

@app.get("/openapi.json")
async def openapi_passthrough():
    spec = await _fetch_openapi()
    spec = json.loads(json.dumps(spec))  # deep copy; never mutate the cache
    if STATE["mask"] is not None:
        allowed = set(STATE["mask"])
        spec["paths"] = {p: v for p, v in spec["paths"].items() if p.strip("/") in allowed}
    if STATE["phi_toolmeta"]:
        # Φ (schema rewrite): agent-visible tool descriptions come from the
        # openapi operation summary, so rewrite them here from the variant file.
        phi = json.load(open(STATE["phi_toolmeta"]))
        for p, ops in spec["paths"].items():
            tool = p.strip("/")
            if tool in phi:
                new_desc = phi[tool].get("description") or ""
                for op in ops.values():
                    if isinstance(op, dict):
                        op["summary"] = new_desc
                        if "description" in op:
                            op["description"] = new_desc
    return JSONResponse(spec)


@app.get("/proxy_config")
async def get_config():
    return {k: v for k, v in STATE.items()}


@app.post("/proxy_config")
async def set_config(request: Request):
    body = await request.json()
    STATE.update({k: v for k, v in body.items() if k in STATE})
    _log({"event": "config_update", "new": body, "tool": None, "mode": None})
    return {"ok": True, "state": {k: v for k, v in STATE.items()}}


@app.get("/proxy_health")
async def health():
    return {"ok": True}


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(path: str, request: Request):
    tool = path.strip("/").split("/")[0]
    task_id = request.headers.get("x-gta-task-id", "")
    if not task_id and STATE.get("curtask_file"):
        try:
            task_id = open(STATE["curtask_file"]).read().strip()
        except OSError:
            pass
    mode = _mode_for(tool)
    if STATE["mask"] is not None and tool not in STATE["mask"]:
        _log({"task_id": task_id, "tool": tool, "mode": "masked", "status": 404})
        return JSONResponse({"detail": f"Tool {tool} not found"}, status_code=404)

    key = (task_id, tool)
    _call_counter[key] += 1
    turn = _call_counter[key]
    t0 = time.time()

    # summarize args without dumping file bytes
    args_summary = dict(request.query_params)
    ctype = request.headers.get("content-type", "")
    body = await request.body()
    if "multipart" in ctype:
        args_summary["_multipart_fields"] = re.findall(rb'name="([^"]+)"', body)[:20]
        args_summary["_multipart_fields"] = [f.decode() for f in args_summary["_multipart_fields"]]
    elif body and "json" in ctype:
        try:
            args_summary["_json"] = json.loads(body)
        except json.JSONDecodeError:
            pass

    base_rec = {"task_id": task_id, "turn": turn, "tool": tool, "mode": mode,
                "args": args_summary, "body_bytes": len(body)}

    if mode in ("tool_free", "unavailable"):
        _log({**base_rec, "status": 503, "latency_s": round(time.time() - t0, 3)})
        # 503 + detail: RemoteTool raises RuntimeError containing this message,
        # lagent surfaces it to the agent as the tool response text.
        return JSONResponse({"detail": UNAVAILABLE_MSG}, status_code=503)

    if mode == "unreliable":
        _unreliable_counter[tool] += 1
        draw_rng = random.Random(f'{STATE["seed"]}:{tool}:{_unreliable_counter[tool]}')
        if draw_rng.random() < STATE["p_fail"]:
            _log({**base_rec, "status": 503, "seed_draw": _unreliable_counter[tool],
                  "failed": True, "latency_s": round(time.time() - t0, 3)})
            return JSONResponse({"detail": FAIL_MSG}, status_code=503)

    # forward to upstream
    fwd_headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in ("host", "content-length", "x-gta-task-id")}
    url = STATE["upstream"].rstrip("/") + "/" + path
    async with httpx.AsyncClient(timeout=600) as client:
        upstream = await client.request(request.method, url, params=request.query_params,
                                        content=body, headers=fwd_headers)
    out = upstream.content
    rec = {**base_rec, "status": upstream.status_code, "raw_return_hash": _sha(out),
           "raw_return_bytes": len(out)}

    # TACO: attribute intervention on the tool's *output* (format / length /
    # evidence position). Applied only on a successful passthrough return, so
    # it composes with, and never masks, the failure/corruption modes above.
    if (STATE.get("taco_spec") and upstream.status_code == 200
            and mode not in ("corrupt_output", "format_only")):
        tspec = _taco.spec_for(tool, STATE["taco_spec"])
        if tspec:
            try:
                payload = json.loads(out)
            except json.JSONDecodeError:
                payload = None
            if payload is not None:
                new_payload, trec = _taco.transform(tool, payload, tspec)
                if trec.get("applied"):
                    out = json.dumps(new_payload, ensure_ascii=False).encode()
                    rec["taco_applied"] = True
                    rec["taco_tokens_before"] = trec.get("tokens_before")
                    rec["taco_tokens_after"] = trec.get("tokens_after")
                    rec["taco_evidence_density"] = trec.get("evidence_density")
                    rec["taco_preserved"] = trec.get("preservation", {}).get("units_present")
                else:
                    rec["taco_applied"] = False
                    rec["taco_skip"] = trec.get("reason")
                tlog = STATE.get("taco_log_path")
                if tlog:
                    trec.update(task_id=task_id, turn=turn,
                                run_meta=STATE["run_meta"], ts=time.time())
                    with _log_lock:
                        with open(tlog, "a") as f:
                            f.write(json.dumps(trec, default=str) + "\n")

    if mode in ("corrupt_output", "format_only") and upstream.status_code == 200:
        if not _openapi_cache["tool_output_types"]:
            await _fetch_openapi()
        rng = random.Random(f'{STATE["seed"]}:{tool}:{task_id}:{turn}')
        out = _corrupt_payload(tool, out, rng)
        rec["corrupted_return_hash"] = _sha(out)
        rec["corrupted"] = True

    rec["latency_s"] = round(time.time() - t0, 3)
    _log(rec)
    resp_headers = {k: v for k, v in upstream.headers.items()
                    if k.lower() in ("content-type",)}
    return Response(content=out, status_code=upstream.status_code, headers=resp_headers)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.getenv("GTA_PROXY_PORT", 16281)))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--upstream", default=os.getenv("GTA_TOOLSERVER_URL", "http://127.0.0.1:16181"))
    ap.add_argument("--config", default=os.getenv("GTA_PROXY_CONFIG"))
    ap.add_argument("--log", default=os.getenv("GTA_PROXY_LOG"))
    ap.add_argument("--mode", default=None, help="override global mode")
    args = ap.parse_args()

    STATE["upstream"] = args.upstream
    if args.config:
        _load_config(args.config)
    if args.log:
        STATE["log_path"] = args.log
    if args.mode:
        STATE["mode"] = args.mode
    os.makedirs(os.path.dirname(os.path.abspath(STATE["log_path"])), exist_ok=True)
    _log({"event": "proxy_start", "state": {k: v for k, v in STATE.items()}})
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
