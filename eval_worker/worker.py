"""Online eval worker: continuously scores LiteLLM proxy traffic.

Loop:
  1. Fetch recent traces from Langfuse (populated by LiteLLM's `langfuse`
     success_callback).
  2. Filter out internal traffic (judge-model calls, guardrail retries) and
     apply deterministic sampling.
  3. Extract question / retrieval context / answer.
  4. Score with DeepEval metrics; the judge LLM is called through the
     LiteLLM proxy itself.
  5. Push scores back onto the trace in Langfuse.

Idempotency: a trace is skipped if it already carries our score name, so
restarts never double-score. Sampling is a deterministic hash of the trace
id, so the keep/skip decision for a given trace never changes.
"""

from __future__ import annotations

import hashlib
import logging
import os
import signal
import time
from datetime import datetime, timedelta, timezone

# Must be set before importing deepeval: the worker is a long-running batch
# service and must not depend on outbound telemetry endpoints.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("ERROR_REPORTING", "NO")

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from extraction import (
    _messages_from_trace_input,
    extract_answer,
    extract_context,
    extract_question,
)
from judge import ProxyJudge
from langfuse_client import LangfuseClient

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("eval_worker")

FAITHFULNESS_SCORE_NAME = "deepeval_faithfulness"
RELEVANCY_SCORE_NAME = "deepeval_answer_relevancy"
SCORING_ERROR_SCORE_NAME = "deepeval_scoring_error"


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        logger.warning("invalid %s, using default %s", name, default)
        return default


SAMPLE_RATE = min(max(env_float("EVAL_SAMPLE_RATE", 1.0), 0.0), 1.0)
POLL_INTERVAL = env_float("EVAL_POLL_INTERVAL_SECONDS", 30.0)
LOOKBACK_MINUTES = env_float("EVAL_LOOKBACK_MINUTES", 60.0)
ENABLE_RELEVANCY = os.environ.get("EVAL_ENABLE_RELEVANCY", "true").lower() == "true"
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "judge-model")
RUN_ONCE = os.environ.get("EVAL_RUN_ONCE", "false").lower() == "true"

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    logger.info("received signal %s, shutting down after current trace", signum)
    _shutdown = True


def sampled_in(trace_id: str) -> bool:
    """Deterministic sampling: same trace id -> same decision, forever."""
    digest = hashlib.sha256(trace_id.encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
    return bucket < SAMPLE_RATE


def _generation_observations(trace_detail: dict) -> list[dict]:
    obs = trace_detail.get("observations") or []
    return [o for o in obs if isinstance(o, dict)]


def trace_models(trace_detail: dict) -> set[str]:
    """All model names/aliases involved in this trace. LiteLLM logs the
    underlying model on the generation and the public alias (the proxy
    model_name) as metadata.model_group."""
    models: set[str] = set()
    for o in _generation_observations(trace_detail):
        if o.get("model"):
            models.add(str(o["model"]))
        md = o.get("metadata") or {}
        if isinstance(md, dict) and md.get("model_group"):
            models.add(str(md["model_group"]))
    return models


def is_internal_traffic(trace_detail: dict) -> bool:
    """Never score the judge's own calls or the guardrail's regeneration
    calls — mirrors the recursion guards inside the guardrail itself."""
    if JUDGE_MODEL in trace_models(trace_detail):
        return True
    metadata = request_metadata(trace_detail)
    if metadata.get("ragas_internal_retry"):
        return True
    return False


def already_scored(trace_detail: dict) -> bool:
    for score in trace_detail.get("scores") or []:
        if isinstance(score, dict) and score.get("name") in (
            FAITHFULNESS_SCORE_NAME,
            SCORING_ERROR_SCORE_NAME,
        ):
            return True
    return False


def request_metadata(trace_detail: dict) -> dict:
    """Collect the calling application's request metadata (the `metadata`
    the client sent via extra_body). LiteLLM's Langfuse integration logs it
    on the generation observation under metadata.requester_metadata; older
    versions put it on the trace metadata. Merge all locations."""
    merged: dict = {}
    trace_md = trace_detail.get("metadata")
    if isinstance(trace_md, dict):
        merged.update(trace_md)
        if isinstance(trace_md.get("requester_metadata"), dict):
            merged.update(trace_md["requester_metadata"])
    for o in _generation_observations(trace_detail):
        obs_md = o.get("metadata") or {}
        if isinstance(obs_md, dict) and isinstance(obs_md.get("requester_metadata"), dict):
            merged.update(obs_md["requester_metadata"])
    return merged


class EvalRunner:
    def __init__(self):
        self.langfuse = LangfuseClient(
            host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        )
        self.judge = ProxyJudge()
        # async_mode=False: worker is a sequential batch process; keeping the
        # metric synchronous makes failures attributable and logs ordered.
        self.faithfulness = FaithfulnessMetric(
            model=self.judge, include_reason=True, async_mode=False
        )
        self.relevancy = (
            AnswerRelevancyMetric(model=self.judge, include_reason=True, async_mode=False)
            if ENABLE_RELEVANCY
            else None
        )

    def score_trace(self, trace: dict) -> None:
        trace_id = trace["id"]
        detail = self.langfuse.get_trace(trace_id)
        if is_internal_traffic(detail):
            logger.debug("trace %s is internal traffic, skipping", trace_id)
            return
        if already_scored(detail):
            logger.debug("trace %s already scored, skipping", trace_id)
            return

        messages = _messages_from_trace_input(detail.get("input"))
        metadata = request_metadata(detail)
        context = extract_context(metadata, messages)
        question = extract_question(messages)
        answer = extract_answer(detail.get("output"))

        if not answer or not question:
            logger.info("trace %s not scorable (missing question/answer), skipping", trace_id)
            return
        if not context:
            logger.info(
                "trace %s has no retrieval context (neither metadata.context "
                "nor system messages) — faithfulness not applicable, skipping",
                trace_id,
            )
            return

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=context,
        )

        try:
            self.faithfulness.measure(test_case)
            self.langfuse.create_score(
                trace_id=trace_id,
                name=FAITHFULNESS_SCORE_NAME,
                value=float(self.faithfulness.score),
                comment=self.faithfulness.reason,
            )
            logger.info(
                "trace %s %s=%.3f",
                trace_id,
                FAITHFULNESS_SCORE_NAME,
                self.faithfulness.score,
            )
            if self.relevancy is not None:
                self.relevancy.measure(test_case)
                self.langfuse.create_score(
                    trace_id=trace_id,
                    name=RELEVANCY_SCORE_NAME,
                    value=float(self.relevancy.score),
                    comment=self.relevancy.reason,
                )
                logger.info(
                    "trace %s %s=%.3f",
                    trace_id,
                    RELEVANCY_SCORE_NAME,
                    self.relevancy.score,
                )
        except Exception as exc:  # judge outage, malformed JSON, etc.
            logger.warning("trace %s scoring failed: %s", trace_id, exc)
            # Mark the trace so it is not retried forever on a permanent
            # failure; the score name makes these visible on the dashboard.
            try:
                self.langfuse.create_score(
                    trace_id=trace_id,
                    name=SCORING_ERROR_SCORE_NAME,
                    value=1.0,
                    comment=str(exc)[:500],
                )
            except Exception as exc2:
                logger.warning("trace %s could not record scoring error: %s", trace_id, exc2)

    def run_cycle(self) -> int:
        """Score one batch of recent traces. Returns number scored/attempted."""
        from_ts = datetime.now(timezone.utc) - timedelta(minutes=LOOKBACK_MINUTES)
        attempted = 0
        page = 1
        while not _shutdown:
            try:
                body = self.langfuse.list_traces(from_timestamp=from_ts, page=page)
            except Exception as exc:
                logger.warning("failed to list traces (page %d): %s", page, exc)
                return attempted
            traces = body.get("data") or []
            if not traces:
                break
            for trace in traces:
                if _shutdown:
                    break
                trace_id = trace.get("id")
                if not trace_id:
                    continue
                if not sampled_in(trace_id):
                    logger.debug("trace %s sampled out", trace_id)
                    continue
                attempted += 1
                try:
                    self.score_trace(trace)
                except Exception as exc:
                    logger.warning("trace %s unexpected error: %s", trace_id, exc)
            meta = body.get("meta") or {}
            total_pages = meta.get("totalPages")
            if total_pages is not None and page >= int(total_pages):
                break
            page += 1
        return attempted

    def run_forever(self) -> None:
        logger.info(
            "eval worker started: sample_rate=%.2f poll=%.0fs lookback=%.0fmin "
            "judge=%s relevancy=%s",
            SAMPLE_RATE, POLL_INTERVAL, LOOKBACK_MINUTES, JUDGE_MODEL, ENABLE_RELEVANCY,
        )
        while not _shutdown:
            started = time.monotonic()
            attempted = self.run_cycle()
            logger.info("cycle done: %d trace(s) attempted", attempted)
            if RUN_ONCE:
                break
            elapsed = time.monotonic() - started
            sleep_for = max(POLL_INTERVAL - elapsed, 1.0)
            deadline = time.monotonic() + sleep_for
            while not _shutdown and time.monotonic() < deadline:
                time.sleep(0.5)
        self.langfuse.close()
        logger.info("eval worker stopped")


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    EvalRunner().run_forever()


if __name__ == "__main__":
    main()
