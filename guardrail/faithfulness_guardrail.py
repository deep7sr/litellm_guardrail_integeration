"""
LiteLLM post-call faithfulness guardrail (in-process RAGAS).

Registered in litellm_config.yaml as a custom guardrail firing on
async_post_call_success_hook (mode: post_call). All RAGAS + judge-model
logic runs inside this LiteLLM process — no separate scoring service.

  1. Reads context + question from the explicit request contract
     (metadata.guardrail_context / metadata.guardrail_question) — it never
     infers context from message roles, which was a confirmed
     prompt-injection vulnerability.
  2. Scores the answer with RAGAS Faithfulness against that context, judged
     by JUDGE_MODEL (reached back through this same proxy).
  3. Applies the verdict per the configured fail mode:
       block (default) — raise a structured 400 error back to the caller
       retry           — regenerate with corrective feedback up to
                         RAGAS_MAX_RETRIES, then serve FALLBACK_TEXT
  4. Logs every scoring attempt to Postgres (ragas_events) for the dashboard.

Integration contract (see docs/INTEGRATION.md): RAG apps must send
  extra_body={"metadata": {
      "guardrail_context": ["chunk 1", "chunk 2", ...],   # required to be scored
      "guardrail_question": "...",                        # optional, recommended
      "guardrail_on_fail": "block" | "retry",             # optional override
      "guardrail_threshold": 0.7,                         # optional override
  }}
Requests without guardrail_context are passed through unscored (and logged
with verdict="unscored") — the guardrail cannot judge groundedness without
knowing what the grounding material was.
"""

import asyncio
import logging
import os
import uuid

import asyncpg
import numpy as np
from fastapi import HTTPException
from openai import AsyncOpenAI
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics import Faithfulness

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth

logger = logging.getLogger("faithfulness_guardrail")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

PASS_THRESHOLD = float(os.environ.get("RAGAS_PASS_THRESHOLD", "0.7"))
MAX_RETRIES = int(os.environ.get("RAGAS_MAX_RETRIES", "3"))
DEFAULT_ON_FAIL = os.environ.get("RAGAS_ON_FAIL", "block")  # "block" | "retry"
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "judge-model")
PROXY_BASE_URL = os.environ.get("PROXY_BASE_URL", "http://localhost:4000/v1")

RETRY_FEEDBACK = (
    "That answer was not fully grounded in the context. Regenerate it using ONLY "
    "the context above. If the context doesn't contain the answer, say so explicitly."
)
FALLBACK_TEXT = "I wasn't able to find a fully supported answer to this in the available documents."

_proxy_client = AsyncOpenAI(
    base_url=PROXY_BASE_URL,
    api_key=os.environ.get("LITELLM_MASTER_KEY", "sk-1234"),
)
_judge_llm = llm_factory(JUDGE_MODEL, client=_proxy_client)
_scorer = Faithfulness(llm=_judge_llm)

_db_pool: asyncpg.Pool | None = None
_db_lock = asyncio.Lock()


async def get_db_pool() -> asyncpg.Pool | None:
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
            logger.warning("[ragas] DATABASE_URL not set, dashboard logging disabled")
            return None
        try:
            pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
            async with pool.acquire() as conn:
                await conn.execute("""
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
                """)
            logger.info("[ragas] connected to dashboard DB, table ready")
            _db_pool = pool
        except Exception as e:
            logger.warning(f"[ragas] could not connect to dashboard DB: {e}")
            _db_pool = None
    return _db_pool


async def _log_event(req_id, attempt, score, verdict, target_model, question, answer):
    pool = await get_db_pool()
    if pool is None:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO ragas_events (req_id, attempt, score, verdict, target_model, question, answer_snippet)
                   VALUES ($1, $2, $3, $4, $5, $6, $7)""",
                req_id, attempt, score, verdict, target_model,
                (question or "")[:500], (answer or "")[:500],
            )
    except Exception as e:
        logger.warning(f"[ragas] req={req_id} failed to log event to DB: {e}")


def _request_metadata(data: dict) -> dict:
    """Caller-supplied metadata; LiteLLM nests it under requester_metadata in
    some versions and merges it into metadata directly in others."""
    md = data.get("metadata") or {}
    requester = md.get("requester_metadata") or {}
    return {**md, **requester}


def _extract_contexts(meta: dict) -> list[str] | None:
    raw = meta.get("guardrail_context")
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = [raw]
    contexts = [c for c in raw if isinstance(c, str) and c.strip()]
    return contexts or None


def _fallback_question(messages: list) -> str:
    # Best-effort only, used when guardrail_question isn't supplied.
    return next(
        (m["content"] for m in reversed(messages)
         if m.get("role") == "user" and isinstance(m.get("content"), str)),
        "",
    )


async def _score(question: str, contexts: list[str], answer: str) -> float | None:
    sample = SingleTurnSample(user_input=question, response=answer, retrieved_contexts=contexts)
    try:
        raw = await _scorer.single_turn_ascore(sample)
    except Exception as e:
        logger.warning(f"[ragas] scoring failed: {e}")
        return None
    # RAGAS returns NaN when the answer contains no checkable claims
    # (e.g. an honest "I don't know") — treat that as fully grounded.
    return 1.0 if np.isnan(raw) else float(raw)


class RagasFaithfulnessGuardrail(CustomGuardrail):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def async_post_call_success_hook(self, data: dict, user_api_key_dict: UserAPIKeyAuth, response):
        # Never re-score the judge's own decompose/verify calls.
        if data.get("model") == JUDGE_MODEL:
            return response

        meta = _request_metadata(data)

        # Never re-score our own regeneration calls.
        if meta.get("ragas_internal_retry"):
            return response

        if not isinstance(response, litellm.ModelResponse):
            return response

        try:
            answer = response.choices[0].message.content
        except (AttributeError, IndexError):
            return response
        if not answer:
            return response

        req_id = str(uuid.uuid4())[:8]
        messages = data.get("messages", [])
        target_model = data.get("model")

        contexts = _extract_contexts(meta)
        question = meta.get("guardrail_question") or _fallback_question(messages)

        if contexts is None:
            # Contract not adopted by this caller: pass through, but record it
            # so unguarded traffic is visible on the dashboard.
            logger.info(f"[ragas] req={req_id} no guardrail_context supplied, passing unscored")
            await _log_event(req_id, 0, None, "unscored", target_model, question, answer)
            return response

        try:
            threshold = float(meta.get("guardrail_threshold", PASS_THRESHOLD))
        except (TypeError, ValueError):
            threshold = PASS_THRESHOLD
        on_fail = meta.get("guardrail_on_fail") or DEFAULT_ON_FAIL
        if on_fail not in ("block", "retry"):
            on_fail = DEFAULT_ON_FAIL

        score = await _score(question, contexts, answer)
        if score is None:
            # Fail open on judge/scoring errors so the proxy never hard-fails
            # traffic because of a scoring-infrastructure problem.
            await _log_event(req_id, 0, None, "error", target_model, question, answer)
            return response

        logger.info(f"[ragas] req={req_id} attempt=0 score={score:.3f} threshold={threshold} on_fail={on_fail}")

        if score >= threshold:
            await _log_event(req_id, 0, score, "passed", target_model, question, answer)
            return response

        if on_fail == "block":
            await _log_event(req_id, 0, score, "blocked", target_model, question, answer)
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Response failed faithfulness guardrail",
                    "guardrail": "ragas-faithfulness",
                    "score": round(score, 3),
                    "threshold": threshold,
                },
            )

        # on_fail == "retry": regenerate with corrective feedback.
        attempt_messages = list(messages)
        attempt = 0
        while True:
            if attempt >= MAX_RETRIES:
                logger.warning(f"[ragas] req={req_id} EXHAUSTED {MAX_RETRIES} retries, final score={score:.3f}")
                await _log_event(req_id, attempt, score, "exhausted", target_model, question, answer)
                response.choices[0].message.content = FALLBACK_TEXT
                return response

            await _log_event(req_id, attempt, score, "retrying", target_model, question, answer)
            attempt += 1
            attempt_messages = attempt_messages + [
                {"role": "assistant", "content": answer},
                {"role": "user", "content": RETRY_FEEDBACK},
            ]
            try:
                regen = await _proxy_client.chat.completions.create(
                    model=target_model, messages=attempt_messages, temperature=0,
                    extra_body={"metadata": {"ragas_internal_retry": True}},
                )
                answer = regen.choices[0].message.content
            except Exception as e:
                logger.warning(f"[ragas] req={req_id} regeneration call failed: {e}")
                await _log_event(req_id, attempt, None, "error", target_model, question, answer)
                response.choices[0].message.content = FALLBACK_TEXT
                return response

            score = await _score(question, contexts, answer)
            if score is None:
                await _log_event(req_id, attempt, None, "error", target_model, question, answer)
                response.choices[0].message.content = answer
                return response
            logger.info(f"[ragas] req={req_id} attempt={attempt} score={score:.3f}")

            if score >= threshold:
                logger.info(f"[ragas] req={req_id} PASSED at attempt={attempt} score={score:.3f}")
                await _log_event(req_id, attempt, score, "passed", target_model, question, answer)
                response.choices[0].message.content = answer
                return response
