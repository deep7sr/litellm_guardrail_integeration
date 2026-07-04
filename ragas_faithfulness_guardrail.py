"""RAGAS faithfulness guardrail for the LiteLLM proxy.

Fully transparent middleware: applications send normal OpenAI-format
requests, and every non-internal chat completion is scored against the
conversation the model was shown (see context_contract.py). Ungrounded
answers are regenerated with corrective feedback; an unverified low-scoring
answer is never served.

Fail-closed by default:
  - scorer error/timeout   -> fallback text served (configurable)
  - retry exhaustion       -> fallback text served
  - streaming responses    -> buffered and scored before release (no bypass)

Judge calls go directly to the judge endpoint (vLLM or any OpenAI-compatible
server), NOT through this proxy, so they can never re-trigger the guardrail.
Regeneration calls go through the in-process LiteLLM router, which also
bypasses the proxy's guardrail layer; the metadata flag check is kept as a
second line of defense.

Observability: every attempt logs the full picture to Postgres — the answer
received from the generation model, the grounding context used, the
claim-by-claim judge verdicts WITH the judge's stated reason for each, the
score, timing, and any error — so the dashboard can show exactly why each
decision was made.
"""

import asyncio
import copy
import json
import logging
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, List, Optional

import asyncpg
from openai import AsyncOpenAI
from ragas.dataset_schema import SingleTurnSample  # noqa: F401  (public API kept importable)
from ragas.llms import llm_factory
# NOTE: this must be ragas.metrics.collections.Faithfulness (the modern,
# instructor-native implementation), NOT ragas.metrics.Faithfulness (the
# legacy PydanticPrompt-based one). The legacy metric's internal
# is_langchain_llm() heuristic (hasattr(llm, "agenerate") and not
# hasattr(llm, "run_config")) misidentifies our llm_factory(...) judge as a
# LangChain LLM and calls agenerate_prompt() on it, which doesn't exist on
# that object -> AttributeError at scoring time, not at import time. The
# collections version is built specifically to pair with llm_factory and
# has the identical claim schema (statement/verdict/reason).
from ragas.metrics.collections import Faithfulness

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth

from context_contract import (
    INTERNAL_FLAG,
    extract_question,
    is_internal_call,
    prompt_grounding_context,
)

logger = logging.getLogger("ragas_faithfulness_guardrail")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Configuration (env-driven so ops can tune without code changes)
# ---------------------------------------------------------------------------

PASS_THRESHOLD = float(os.environ.get("RAGAS_PASS_THRESHOLD", "0.7"))
MAX_RETRIES = int(os.environ.get("RAGAS_MAX_RETRIES", "2"))
SCORING_TIMEOUT_S = float(os.environ.get("RAGAS_SCORING_TIMEOUT_S", "60"))
MAX_CONCURRENT_SCORING = int(os.environ.get("RAGAS_MAX_CONCURRENT_SCORING", "8"))

# "block" (serve fallback text) or "allow" (serve the unverified answer) when
# the judge errors out or times out. "block" = fail-closed.
ON_SCORER_ERROR = os.environ.get("RAGAS_ON_SCORER_ERROR", "block").lower()

JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "http://vllm-judge:8000/v1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", "not-needed-for-local-vllm")

RETRY_FEEDBACK = (
    "That answer was not fully grounded in the provided context. Regenerate it "
    "using ONLY facts stated in the context. If the context does not contain "
    "the answer, say so explicitly instead of guessing."
)
FALLBACK_TEXT = (
    "I wasn't able to find a fully supported answer to this in the available documents."
)

# Caps for what we persist per event (keep rows bounded, but big enough to
# actually diagnose from the dashboard).
Q_CAP, A_CAP, CTX_CAP = 1000, 4000, 8000

# ---------------------------------------------------------------------------
# Judge / scorer — talks directly to the judge endpoint, never via the proxy
# ---------------------------------------------------------------------------

_judge_client = AsyncOpenAI(base_url=JUDGE_BASE_URL, api_key=JUDGE_API_KEY)
_judge_llm = llm_factory(JUDGE_MODEL, client=_judge_client)
_scorer = Faithfulness(llm=_judge_llm)
_scoring_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCORING)


@dataclass
class ScoreResult:
    """Everything the judge produced for one answer, for decisions AND logging.

    score: fraction of claims supported; NaN if the answer contained no
    checkable claims; None if the judge errored/timed out (see .error).
    claims: [{"statement": str, "verdict": 0|1, "reason": str}, ...] — the
    judge's per-claim decision and its stated reason.
    """
    score: Optional[float]
    claims: List[dict] = field(default_factory=list)
    error: str = ""
    duration_ms: int = 0


async def score_answer(question: str, contexts: List[str], answer: str) -> ScoreResult:
    question = question or "N/A"
    context_str = "\n".join(contexts)
    started = time.monotonic()

    async def _run():
        # Call the metric's two stages directly (instead of ascore) so we can
        # capture the claim-level breakdown for the dashboard.
        statements = await _scorer._create_statements(question, answer)
        if not statements:
            return float("nan"), []
        verdicts = await _scorer._create_verdicts(statements, context_str)
        claims = [
            {"statement": s.statement, "verdict": int(s.verdict), "reason": s.reason}
            for s in verdicts.statements
        ]
        if not claims:
            return float("nan"), []
        score = sum(c["verdict"] for c in claims) / len(claims)
        return score, claims

    try:
        async with _scoring_semaphore:
            score, claims = await asyncio.wait_for(_run(), timeout=SCORING_TIMEOUT_S)
        return ScoreResult(
            score=score, claims=claims,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
    except asyncio.TimeoutError:
        msg = f"judge scoring timed out after {SCORING_TIMEOUT_S:.0f}s"
        logger.warning("[ragas] %s", msg)
        return ScoreResult(score=None, error=msg,
                           duration_ms=int((time.monotonic() - started) * 1000))
    except Exception as e:  # noqa: BLE001 — any judge failure is handled by policy
        logger.warning("[ragas] scoring failed: %s", e)
        return ScoreResult(score=None, error=f"{type(e).__name__}: {e}",
                           duration_ms=int((time.monotonic() - started) * 1000))


# ---------------------------------------------------------------------------
# Observability — Postgres event log (best-effort, never blocks the request)
# ---------------------------------------------------------------------------

_db_pool = None
_db_lock = asyncio.Lock()

_TABLE_DDL = """
    CREATE TABLE IF NOT EXISTS ragas_events (
        id SERIAL PRIMARY KEY,
        req_id TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        attempt INT NOT NULL,
        score DOUBLE PRECISION,
        verdict TEXT NOT NULL,
        target_model TEXT,
        question TEXT,
        answer_snippet TEXT
    )
"""

# Columns added over time; applied idempotently so existing deployments
# migrate on startup without manual steps.
_MIGRATIONS = [
    "ALTER TABLE ragas_events ADD COLUMN IF NOT EXISTS context TEXT",
    "ALTER TABLE ragas_events ADD COLUMN IF NOT EXISTS claims JSONB",
    "ALTER TABLE ragas_events ADD COLUMN IF NOT EXISTS error TEXT",
    "ALTER TABLE ragas_events ADD COLUMN IF NOT EXISTS duration_ms INT",
    "ALTER TABLE ragas_events ADD COLUMN IF NOT EXISTS threshold DOUBLE PRECISION",
]


async def get_db_pool():
    global _db_pool
    if _db_pool is not None:
        return _db_pool
    async with _db_lock:
        if _db_pool is not None:
            return _db_pool
        dsn = os.environ.get("DATABASE_URL")
        if dsn and "?" in dsn:
            dsn = dsn.split("?")[0]
        if not dsn:
            logger.warning("[ragas] DATABASE_URL not set, event logging disabled")
            return None
        try:
            pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
            async with pool.acquire() as conn:
                await conn.execute(_TABLE_DDL)
                for stmt in _MIGRATIONS:
                    await conn.execute(stmt)
            logger.info("[ragas] connected to event DB, table ready")
            _db_pool = pool
        except Exception as e:
            logger.warning("[ragas] could not connect to event DB: %s", e)
            _db_pool = None
    return _db_pool


async def log_event(
    req_id, attempt, score, verdict, target_model, question, answer,
    context=None, claims=None, error=None, duration_ms=None,
):
    pool = await get_db_pool()
    if pool is None:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO ragas_events
                   (req_id, attempt, score, verdict, target_model, question,
                    answer_snippet, context, claims, error, duration_ms, threshold)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10,$11,$12)""",
                req_id, attempt, score, verdict, target_model,
                (question or "")[:Q_CAP], (answer or "")[:A_CAP],
                (context or "")[:CTX_CAP] or None,
                json.dumps(claims) if claims else None,
                error or None, duration_ms, PASS_THRESHOLD,
            )
    except Exception as e:
        logger.warning("[ragas] req=%s failed to log event: %s", req_id, e)


# ---------------------------------------------------------------------------
# Guardrail
# ---------------------------------------------------------------------------

@dataclass
class Outcome:
    action: str  # "allow" | "replace"
    final_text: str = ""


class RagasFaithfulnessGuardrail(CustomGuardrail):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info(
            "[ragas] guardrail initialized: threshold=%.2f retries=%d "
            "on_scorer_error=%s judge=%s @ %s",
            PASS_THRESHOLD, MAX_RETRIES, ON_SCORER_ERROR,
            JUDGE_MODEL, JUDGE_BASE_URL,
        )

    # ---- non-streaming ----------------------------------------------------

    async def async_post_call_success_hook(
        self, data: dict, user_api_key_dict: UserAPIKeyAuth, response
    ):
        if not isinstance(response, litellm.ModelResponse):
            return response
        if is_internal_call(data):
            return response
        try:
            answer = response.choices[0].message.content
        except (AttributeError, IndexError):
            return response
        if not isinstance(answer, str) or not answer.strip():
            # Tool calls / empty content: nothing to fact-check.
            return response

        outcome = await self._evaluate(data, answer)
        if outcome.final_text != answer:
            response.choices[0].message.content = outcome.final_text
        return response

    # ---- streaming ---------------------------------------------------------
    # Without this hook, streamed completions would bypass the guardrail
    # entirely. We buffer the full stream, score it, and only then release it
    # (or substitute the fallback). Grounding cannot be verified on a partial
    # answer, so buffering is unavoidable — same tradeoff as Bedrock
    # guardrails on streaming.

    async def async_post_call_streaming_iterator_hook(
        self, user_api_key_dict: UserAPIKeyAuth, response: Any, request_data: dict
    ) -> AsyncGenerator[Any, None]:
        if is_internal_call(request_data):
            async for chunk in response:
                yield chunk
            return

        chunks = []
        async for chunk in response:
            chunks.append(chunk)

        answer = ""
        for c in chunks:
            try:
                delta = c.choices[0].delta
                if delta is not None and isinstance(delta.content, str):
                    answer += delta.content
            except (AttributeError, IndexError):
                continue

        if not answer.strip():
            for c in chunks:
                yield c
            return

        outcome = await self._evaluate(request_data, answer)

        if outcome.action == "allow" and outcome.final_text == answer:
            for c in chunks:
                yield c
            return

        replacement = outcome.final_text
        if not chunks:
            return
        try:
            head = copy.deepcopy(chunks[0])
            head.choices[0].delta.content = replacement
            head.choices[0].finish_reason = None
            tail = copy.deepcopy(chunks[-1])
            tail.choices[0].delta.content = ""
            tail.choices[0].finish_reason = "stop"
            yield head
            yield tail
        except (AttributeError, IndexError):
            # Cannot synthesize a replacement chunk: fail closed by ending the
            # stream rather than releasing an unverified answer.
            logger.error("[ragas] could not synthesize replacement stream chunk")
            return

    # ---- shared evaluation core --------------------------------------------

    async def _evaluate(self, data: dict, answer: str) -> Outcome:
        req_id = str(data.get("litellm_call_id") or uuid.uuid4())[:8]
        target_model = data.get("model")
        question = extract_question(data)
        messages = list(data.get("messages") or [])

        contexts = prompt_grounding_context(messages)
        if contexts is None:
            # Empty / non-text conversation: nothing exists to verify against.
            await log_event(req_id, 0, None, "skipped_no_context", target_model, question, answer)
            return Outcome("allow", final_text=answer)
        context_str = "\n".join(contexts)

        current = answer
        attempt = 0

        while True:
            result = await score_answer(question, contexts, current)
            score = result.score

            if score is None:
                verdict = "scorer_error_allowed" if ON_SCORER_ERROR == "allow" else "scorer_error_blocked"
                logger.warning("[ragas] req=%s scorer error (%s): %s", req_id, verdict, result.error)
                await log_event(req_id, attempt, None, verdict, target_model, question, current,
                                context=context_str, error=result.error, duration_ms=result.duration_ms)
                if ON_SCORER_ERROR == "allow":
                    return Outcome("allow", final_text=current)
                return Outcome("replace", final_text=FALLBACK_TEXT)

            if math.isnan(score):
                # No verifiable claims (e.g. an honest refusal): nothing to
                # hallucinate. Logged with its own verdict, never as a fake 1.0.
                logger.info("[ragas] req=%s attempt=%d no checkable claims, passing", req_id, attempt)
                await log_event(req_id, attempt, None, "no_claims", target_model, question, current,
                                context=context_str, duration_ms=result.duration_ms)
                return Outcome("allow", final_text=current)

            logger.info("[ragas] req=%s attempt=%d score=%.3f (%d/%d claims supported, %dms)",
                        req_id, attempt, score,
                        sum(c["verdict"] for c in result.claims), len(result.claims),
                        result.duration_ms)

            if score >= PASS_THRESHOLD:
                await log_event(req_id, attempt, score, "passed", target_model, question, current,
                                context=context_str, claims=result.claims, duration_ms=result.duration_ms)
                return Outcome("allow", final_text=current)

            if attempt >= MAX_RETRIES:
                logger.warning(
                    "[ragas] req=%s EXHAUSTED %d retries, final score=%.3f, serving fallback",
                    req_id, MAX_RETRIES, score,
                )
                await log_event(req_id, attempt, score, "exhausted", target_model, question, current,
                                context=context_str, claims=result.claims, duration_ms=result.duration_ms)
                return Outcome("replace", final_text=FALLBACK_TEXT)

            await log_event(req_id, attempt, score, "retrying", target_model, question, current,
                            context=context_str, claims=result.claims, duration_ms=result.duration_ms)
            attempt += 1
            messages = messages + [
                {"role": "assistant", "content": current},
                {"role": "user", "content": RETRY_FEEDBACK},
            ]
            regenerated = await self._regenerate(target_model, messages, req_id)
            if regenerated is None:
                await log_event(req_id, attempt, None, "regen_error", target_model, question, current,
                                context=context_str,
                                error="regeneration call failed or returned empty (see proxy logs)")
                return Outcome("replace", final_text=FALLBACK_TEXT)
            current = regenerated

    async def _regenerate(self, model: str, messages: list, req_id: str) -> Optional[str]:
        """Regenerate through the in-process router (bypasses the proxy's HTTP
        layer and guardrail hooks — no self-triggering, no master-key use)."""
        try:
            llm_router = None
            try:
                from litellm.proxy.proxy_server import llm_router as _router
                llm_router = _router
            except Exception:
                pass
            kwargs = dict(
                model=model,
                messages=messages,
                temperature=0.0,
                metadata={INTERNAL_FLAG: True},
            )
            if llm_router is not None:
                resp = await llm_router.acompletion(**kwargs)
            else:
                resp = await litellm.acompletion(**kwargs)
            content = resp.choices[0].message.content
            if isinstance(content, str) and content.strip():
                return content
            return None
        except Exception as e:
            logger.warning("[ragas] req=%s regeneration failed: %s", req_id, e)
            return None
