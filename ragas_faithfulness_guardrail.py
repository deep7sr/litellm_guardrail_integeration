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
"""

import asyncio
import copy
import logging
import math
import os
import uuid
from dataclasses import dataclass
from typing import Any, AsyncGenerator, List, Optional

import asyncpg
from openai import AsyncOpenAI
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics import Faithfulness

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
# ---------------------------------------------------------------------------
# Judge / scorer — talks directly to the judge endpoint, never via the proxy
# ---------------------------------------------------------------------------

_judge_client = AsyncOpenAI(base_url=JUDGE_BASE_URL, api_key=JUDGE_API_KEY)
_judge_llm = llm_factory(JUDGE_MODEL, client=_judge_client)
_scorer = Faithfulness(llm=_judge_llm)
_scoring_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCORING)


async def score_answer(question: str, contexts: List[str], answer: str) -> Optional[float]:
    """Return the faithfulness score in [0,1], NaN if the answer contained no
    checkable claims, or None on scorer error/timeout."""
    sample = SingleTurnSample(
        user_input=question or "N/A",
        response=answer,
        retrieved_contexts=contexts,
    )
    try:
        async with _scoring_semaphore:
            score = await asyncio.wait_for(
                _scorer.single_turn_ascore(sample), timeout=SCORING_TIMEOUT_S
            )
        return float(score)
    except asyncio.TimeoutError:
        logger.warning("[ragas] scoring timed out after %.0fs", SCORING_TIMEOUT_S)
        return None
    except Exception as e:  # noqa: BLE001 — any judge failure is handled by policy
        logger.warning("[ragas] scoring failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Observability — Postgres event log (best-effort, never blocks the request)
# ---------------------------------------------------------------------------

_db_pool = None
_db_lock = asyncio.Lock()


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
                await conn.execute(
                    """
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
                )
            logger.info("[ragas] connected to event DB, table ready")
            _db_pool = pool
        except Exception as e:
            logger.warning("[ragas] could not connect to event DB: %s", e)
            _db_pool = None
    return _db_pool


async def log_event(req_id, attempt, score, verdict, target_model, question, answer):
    pool = await get_db_pool()
    if pool is None:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO ragas_events
                   (req_id, attempt, score, verdict, target_model, question, answer_snippet)
                   VALUES ($1, $2, $3, $4, $5, $6, $7)""",
                req_id, attempt, score, verdict, target_model,
                (question or "")[:500], (answer or "")[:500],
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

        current = answer
        attempt = 0

        while True:
            score = await score_answer(question, contexts, current)

            if score is None:
                if ON_SCORER_ERROR == "allow":
                    logger.warning("[ragas] req=%s scorer error, allowing (fail-open configured)", req_id)
                    await log_event(req_id, attempt, None, "scorer_error_allowed", target_model, question, current)
                    return Outcome("allow", final_text=current)
                logger.warning("[ragas] req=%s scorer error, serving fallback (fail-closed)", req_id)
                await log_event(req_id, attempt, None, "scorer_error_blocked", target_model, question, current)
                return Outcome("replace", final_text=FALLBACK_TEXT)

            if math.isnan(score):
                # RAGAS returns NaN when the answer contains no verifiable
                # claims (e.g. an honest refusal). Nothing to hallucinate.
                logger.info("[ragas] req=%s attempt=%d no checkable claims, passing", req_id, attempt)
                await log_event(req_id, attempt, None, "no_claims", target_model, question, current)
                return Outcome("allow", final_text=current)

            logger.info("[ragas] req=%s attempt=%d score=%.3f", req_id, attempt, score)

            if score >= PASS_THRESHOLD:
                await log_event(req_id, attempt, score, "passed", target_model, question, current)
                return Outcome("allow", final_text=current)

            if attempt >= MAX_RETRIES:
                logger.warning(
                    "[ragas] req=%s EXHAUSTED %d retries, final score=%.3f, serving fallback",
                    req_id, MAX_RETRIES, score,
                )
                await log_event(req_id, attempt, score, "exhausted", target_model, question, current)
                return Outcome("replace", final_text=FALLBACK_TEXT)

            await log_event(req_id, attempt, score, "retrying", target_model, question, current)
            attempt += 1
            messages = messages + [
                {"role": "assistant", "content": current},
                {"role": "user", "content": RETRY_FEEDBACK},
            ]
            regenerated = await self._regenerate(target_model, messages, req_id)
            if regenerated is None:
                await log_event(req_id, attempt, None, "regen_error", target_model, question, current)
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
