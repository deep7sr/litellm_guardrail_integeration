"""The async external eval pipeline (one cycle = run_once):

  fetch recent Langfuse traces
    -> drop internal/judge traffic          (recursion guard)
    -> drop already-scored traces           (idempotency, SQLite)
    -> deterministic sampling per trace id  (RAG vs non-RAG rates)
    -> cap at max_traces_per_run            (bounded judge spend per cycle)
    -> extract question/context/answer      (guardrail's message contract)
    -> DeepEval metrics via the proxy judge
    -> push scores back to Langfuse (name, value, reason comment)

Per-trace failures are counted and logged, never fatal — one malformed
trace or one judge hiccup must not kill the monitoring loop.
"""

import hashlib
import logging
from datetime import datetime, timedelta, timezone

from .config import EvalConfig
from .extraction import _get, build_case
from .state import ScoreState

logger = logging.getLogger("eval_pipeline")

_PAGE_SIZE = 50
_MAX_PAGES = 20  # safety bound on Langfuse pagination per cycle


def sampled(trace_id: str, rate: float) -> bool:
    """Deterministic per-trace sampling: same trace id always gets the same
    decision, so retries/re-runs are stable and idempotency stays cheap."""
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    h = int(hashlib.sha256(trace_id.encode()).hexdigest()[:8], 16)
    return (h / 0xFFFFFFFF) < rate


def _trace_model(trace) -> str:
    """Best-effort model name — LiteLLM versions put it in different spots."""
    inp = _get(trace, "input")
    if isinstance(inp, dict) and isinstance(inp.get("model"), str):
        return inp["model"]
    md = _get(trace, "metadata") or {}
    if isinstance(md, dict):
        for key in ("model", "requested_model", "model_group", "ls_model_name"):
            if isinstance(md.get(key), str):
                return md[key]
    return ""


def should_skip(trace, cfg: EvalConfig) -> bool:
    """Recursion/exclusion guard: never evaluate the judges' own traffic,
    anything tagged internal, or the guardrail's regeneration calls."""
    tags = _get(trace, "tags") or []
    if cfg.internal_tag in tags:
        return True
    md = _get(trace, "metadata") or {}
    if isinstance(md, dict):
        if md.get("eval_internal") or md.get("ragas_internal_retry"):
            return True
        rm = md.get("requester_metadata")
        if isinstance(rm, dict) and (rm.get("eval_internal") or rm.get("ragas_internal_retry")):
            return True
    return _trace_model(trace) in cfg.skip_models


def fetch_recent_traces(langfuse, cfg: EvalConfig, now: datetime | None = None) -> list:
    now = now or datetime.now(timezone.utc)
    since = now - timedelta(minutes=cfg.lookback_minutes)
    traces: list = []
    for page in range(1, _MAX_PAGES + 1):
        resp = langfuse.fetch_traces(
            from_timestamp=since, to_timestamp=now, page=page, limit=_PAGE_SIZE
        )
        data = _get(resp, "data") or []
        traces.extend(data)
        if len(data) < _PAGE_SIZE:
            break
    return traces


def run_once(cfg: EvalConfig, langfuse, metric_factory, state: ScoreState) -> dict:
    """One pipeline cycle. Returns counters for logging/monitoring."""
    stats = {"fetched": 0, "skipped_internal": 0, "already_scored": 0,
             "not_sampled": 0, "unextractable": 0, "scored": 0,
             "errors": 0, "capped": 0}

    candidates = []
    for trace in fetch_recent_traces(langfuse, cfg):
        stats["fetched"] += 1
        trace_id = _get(trace, "id")
        if not trace_id:
            continue
        if should_skip(trace, cfg):
            stats["skipped_internal"] += 1
            continue
        if state.is_scored(trace_id):
            stats["already_scored"] += 1
            continue
        candidates.append((trace_id, trace))

    for trace_id, trace in candidates:
        if stats["scored"] + stats["errors"] >= cfg.max_traces_per_run:
            stats["capped"] += 1
            continue

        case = build_case(trace, cfg.evidence_marker)
        if case is None:
            stats["unextractable"] += 1
            # Mark it so we don't re-parse the same malformed trace forever.
            state.mark_scored(trace_id, ["unextractable"])
            continue

        rate = cfg.sample_rate_rag if case.is_rag else cfg.sample_rate_non_rag
        if not sampled(trace_id, rate):
            stats["not_sampled"] += 1
            state.mark_scored(trace_id, ["not_sampled"])
            continue

        pushed = []
        try:
            for score_name, measure in metric_factory(case):
                value, reason = measure()
                langfuse.score(
                    # Deterministic id => Langfuse upserts on retry instead
                    # of stacking duplicate scores on the trace.
                    id=f"deepeval-{trace_id}-{score_name}",
                    trace_id=trace_id,
                    name=score_name,
                    value=value,
                    comment=reason,
                )
                pushed.append(score_name)
            state.mark_scored(trace_id, pushed)
            stats["scored"] += 1
            logger.info("scored trace=%s rag=%s metrics=%s",
                        trace_id, case.is_rag, ",".join(pushed))
        except Exception:
            # Partial scores may already be in Langfuse; not marking the
            # trace means the next cycle retries the whole set — the
            # deterministic score ids make that an upsert, not a duplicate.
            stats["errors"] += 1
            logger.exception("failed scoring trace=%s (pushed=%s)", trace_id, pushed)

    logger.info("cycle done: %s", stats)
    return stats
