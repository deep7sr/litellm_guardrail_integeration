"""Minimal in-memory Langfuse API stub for end-to-end pipeline tests.

Implements just enough of the Langfuse public API for:
  * LiteLLM's `langfuse` success_callback (SDK v2 batch ingestion), and
  * the eval worker's REST usage (list traces, get trace, create score).

Also exposes /debug/events so tests can inspect exactly what the LiteLLM
callback sent.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request

app = FastAPI()

EVENTS: list[dict] = []          # raw ingestion events, in arrival order
TRACES: dict[str, dict] = {}     # trace_id -> materialized trace
TRACE_ORDER: list[str] = []
SCORES: list[dict] = []


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_basic_auth(request: Request) -> None:
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("basic "):
        raise HTTPException(status_code=401, detail="missing basic auth")
    try:
        base64.b64decode(header.split(" ", 1)[1])
    except Exception:
        raise HTTPException(status_code=401, detail="bad basic auth")


def _merge_trace(trace_id: str, body: dict) -> None:
    trace = TRACES.get(trace_id)
    if trace is None:
        trace = {
            "id": trace_id,
            "timestamp": body.get("timestamp") or _now_iso(),
            "name": None,
            "input": None,
            "output": None,
            "metadata": None,
            "tags": [],
            "scores": [],
            "observations": [],
        }
        TRACES[trace_id] = trace
        TRACE_ORDER.append(trace_id)
    for key in ("name", "input", "output", "metadata", "tags", "userId", "sessionId"):
        value = body.get(key)
        if value is not None:
            trace[key] = value


@app.post("/api/public/ingestion")
async def ingestion(request: Request):
    _require_basic_auth(request)
    payload = await request.json()
    batch = payload.get("batch", [])
    successes = []
    for event in batch:
        EVENTS.append(event)
        etype = event.get("type", "")
        body = event.get("body", {}) or {}
        if etype.startswith("trace"):
            trace_id = body.get("id")
            if trace_id:
                _merge_trace(trace_id, body)
        elif etype.startswith("generation") or etype.startswith("observation") or etype.startswith("span"):
            trace_id = body.get("traceId")
            if trace_id:
                _merge_trace(trace_id, {"timestamp": body.get("startTime")})
                obs = {
                    "id": body.get("id"),
                    "type": "GENERATION",
                    "model": body.get("model"),
                    "input": body.get("input"),
                    "output": body.get("output"),
                    "metadata": body.get("metadata"),
                }
                trace = TRACES[trace_id]
                existing = next(
                    (o for o in trace["observations"] if o["id"] == obs["id"]), None
                )
                if existing:
                    for k, v in obs.items():
                        if v is not None:
                            existing[k] = v
                else:
                    trace["observations"].append(obs)
                # Mirror generation input/output onto the trace if the trace
                # events didn't carry them (mirrors Langfuse UI behaviour of
                # LiteLLM's integration, which sets these on the trace too).
                if trace.get("input") is None and obs["input"] is not None:
                    trace["input"] = obs["input"]
                if obs["output"] is not None:
                    trace["output"] = obs["output"]
        elif etype.startswith("score"):
            SCORES.append(body)
            trace_id = body.get("traceId")
            if trace_id in TRACES:
                TRACES[trace_id]["scores"].append(body)
        successes.append({"id": event.get("id"), "status": 201})
    return {"successes": successes, "errors": []}


@app.get("/api/public/traces")
async def list_traces(request: Request, page: int = 1, limit: int = 50):
    _require_basic_auth(request)
    from_ts = request.query_params.get("fromTimestamp")
    traces = [TRACES[tid] for tid in TRACE_ORDER]
    if from_ts:
        from_dt = datetime.fromisoformat(from_ts.replace("Z", "+00:00"))
        def after(t: dict) -> bool:
            try:
                ts = datetime.fromisoformat(str(t["timestamp"]).replace("Z", "+00:00"))
            except ValueError:
                return True
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts >= from_dt
        traces = [t for t in traces if after(t)]
    total = len(traces)
    start = (page - 1) * limit
    page_items = traces[start : start + limit]
    summaries = [{k: v for k, v in t.items() if k != "observations"} for t in page_items]
    return {
        "data": summaries,
        "meta": {
            "page": page,
            "limit": limit,
            "totalItems": total,
            "totalPages": max((total + limit - 1) // limit, 1),
        },
    }


@app.get("/api/public/traces/{trace_id}")
async def get_trace(trace_id: str, request: Request):
    _require_basic_auth(request)
    trace = TRACES.get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return trace


@app.post("/api/public/scores")
async def create_score(request: Request):
    _require_basic_auth(request)
    body = await request.json()
    body.setdefault("id", f"score-{len(SCORES)+1}")
    SCORES.append(body)
    trace_id = body.get("traceId")
    if trace_id in TRACES:
        TRACES[trace_id]["scores"].append(body)
    return body


@app.get("/debug/events")
async def debug_events():
    return {"events": EVENTS, "n_traces": len(TRACES), "scores": SCORES}


@app.post("/debug/reset")
async def debug_reset():
    EVENTS.clear()
    TRACES.clear()
    TRACE_ORDER.clear()
    SCORES.clear()
    return {"ok": True}


# The langfuse SDK may probe health/auth endpoints on startup.
@app.get("/api/public/health")
async def health():
    return {"status": "OK", "version": "stub"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=9002, log_level="warning")
