"""
LiteLLM post-call faithfulness guardrail (in-process RAGAS).

Production v1. Context and question are read from the OpenAI RAG message
structure — there is NO caller-facing metadata contract:

  - Retrieved context lives in an assistant-role message formatted per
    OpenAI's own RAG example:
        context_str = "--- Retrieved Evidence ---\\n" + "\\n\\n".join(chunks)
    The guardrail identifies context as the LAST assistant-role message
    carrying the evidence marker. The marker distinguishes injected context
    from prior model answers in multi-turn conversations (both are
    assistant-role, so role alone cannot disambiguate).
  - The question is the last user-role message.

Flow:
  1. Extract context/question from the message structure.
  2. Strip any <think>...</think> reasoning block from the answer, so both
     the served response and the judge's scoring see only the final answer.
  3. Score with RAGAS Faithfulness via JUDGE_MODEL (through this same
     proxy), bounded by JUDGE_TIMEOUT_SECONDS.
  4. Enforce per GUARDRAIL_MODE:
       retry (default) — regenerate with corrective feedback up to
                         MAX_ATTEMPTS, then serve FALLBACK_TEXT
       block           — raise a structured 400 back to the caller
  5. Log every attempt (user message, context, answer, score, verdict) to
     Postgres for the dashboard.

Requests with no evidence-marked assistant message pass through unscored
(verdict="unscored") — the guardrail cannot judge groundedness without
knowing the grounding material, and non-RAG traffic must never break.

Security invariant: context is only ever read from assistant-role messages,
never user-role content — a user cannot inject fabricated "context", even
by typing the evidence marker into their own message.

Recursion guards (both required — removing either reinfects the proxy with
self-scoring loops):
  - requests whose model is JUDGE_MODEL are never scored (RAGAS's own
    decompose/verify calls re-enter this proxy);
  - requests tagged ragas_internal_retry are never scored (our own
    regeneration calls re-enter this proxy). This tag is internal
    machinery, not a developer-facing contract.
"""

import asyncio
import logging
import os
import re
import uuid

import asyncpg
import numpy as np
from fastapi import HTTPException
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth

logger = logging.getLogger("faithfulness_guardrail")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# All configuration is central (env vars) — callers cannot override per
# request. Reuses the deployment's existing variable names.
PASS_THRESHOLD = float(os.environ.get("GUARDRAIL_THRESHOLD", "0.7"))
MAX_RETRIES = int(os.environ.get("MAX_ATTEMPTS", "3"))
DEFAULT_ON_FAIL = os.environ.get("GUARDRAIL_MODE", "retry")  # "retry" | "block"
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "judge-model")
PROXY_BASE_URL = os.environ.get("PROXY_BASE_URL", "http://localhost:4000/v1")
# A single faithfulness score fans out into several judge calls; this bounds
# the whole scoring step so a hung judge cannot stall the proxy request.
JUDGE_TIMEOUT_SECONDS = float(os.environ.get("JUDGE_TIMEOUT_SECONDS", "60"))
# Part of the OpenAI RAG contract — the header apps prepend to the
# retrieved-context assistant message.
EVIDENCE_MARKER = os.environ.get("EVIDENCE_MARKER", "--- Retrieved Evidence ---")
# RAGAS's InstructorModelArgs defaults max_tokens to 1024 for the judge's
# internal claim-decomposition/verification JSON. Answers with several
# claims can overflow that, truncating the JSON mid-generation — seen live
# as a deterministic "Failed to validate JSON" error that no amount of
# retrying fixes, since the token budget is the same on every attempt.
JUDGE_MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", "4096"))
# Separately, judge calls can still fail for genuinely transient reasons
# (network hiccups, rate limits). Retrying the scoring call itself — not
# just the outer generate/regenerate loop — absorbs those without falling
# through to the fail-open path.
SCORE_RETRY_ATTEMPTS = int(os.environ.get("SCORE_RETRY_ATTEMPTS", "2"))

RETRY_FEEDBACK = (
    "That answer was not fully grounded in the context. Regenerate it using ONLY "
    "the context above. If the context doesn't contain the answer, say so explicitly."
)
FALLBACK_TEXT = "I wasn't able to find a fully supported answer to this in the available documents."

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks some reasoning models emit, so the
    judge and the caller both see only the actual final answer."""
    if not text:
        return text
    return _THINK_BLOCK_RE.sub("", text).strip()


_proxy_client = AsyncOpenAI(
    base_url=PROXY_BASE_URL,
    api_key=os.environ.get("LITELLM_MASTER_KEY", "sk-1234"),
)
_judge_llm = llm_factory(JUDGE_MODEL, client=_proxy_client, max_tokens=JUDGE_MAX_TOKENS)
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
            logger.warning("[faithfulness] DATABASE_URL not set, event logging disabled")
            return None
        try:
            pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
            async with pool.acquire() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS faithfulness_events (
                        id SERIAL PRIMARY KEY,
                        req_id TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        attempt INT NOT NULL,
                        score DOUBLE PRECISION,
                        verdict TEXT NOT NULL,
                        target_model TEXT,
                        answer_snippet TEXT
                    )
                """)
                # Older deployments may have the table without newer columns.
                await conn.execute("ALTER TABLE faithfulness_events ADD COLUMN IF NOT EXISTS context_snippet TEXT")
                await conn.execute("ALTER TABLE faithfulness_events ADD COLUMN IF NOT EXISTS user_message TEXT")
                await conn.execute("ALTER TABLE faithfulness_events ADD COLUMN IF NOT EXISTS reason TEXT")
            logger.info("[faithfulness] connected to events DB, table ready")
            _db_pool = pool
        except Exception as e:
            logger.warning(f"[faithfulness] could not connect to events DB: {e}")
            _db_pool = None
    return _db_pool


async def _log_event(req_id, attempt, score, verdict, target_model,
                     user_message, context, answer, reason=None):
    pool = await get_db_pool()
    if pool is None:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO faithfulness_events
                   (req_id, attempt, score, verdict, target_model, user_message,
                    context_snippet, answer_snippet, reason)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)""",
                req_id, attempt, score, verdict, target_model,
                (user_message or "")[:1000],
                (context or "")[:2000], (answer or "")[:1000],
                (reason or "")[:1000] or None,
            )
    except Exception as e:
        logger.warning(f"[faithfulness] req={req_id} failed to log event to DB: {e}")


def _is_internal_retry(data: dict) -> bool:
    """True for the guardrail's own regeneration calls (recursion guard 2).
    LiteLLM nests caller metadata under requester_metadata in some versions
    and merges it directly in others, so check both."""
    md = data.get("metadata") or {}
    return bool(md.get("ragas_internal_retry")
                or (md.get("requester_metadata") or {}).get("ragas_internal_retry"))


def _extract_contexts(messages: list) -> list[str] | None:
    """Context = the LAST assistant-role message carrying the evidence
    marker. Assistant-role-only is the security invariant: user-role content
    is never treated as context, so a user typing the marker into their own
    message cannot inject fabricated context."""
    mk = EVIDENCE_MARKER.lower()
    for m in reversed(messages):
        if m.get("role") == "assistant" and isinstance(m.get("content"), str):
            content = m["content"]
            low = content.lower()
            if mk in low:
                idx = low.index(mk)
                body = content[idx + len(EVIDENCE_MARKER):].strip()
                if body:
                    return [body]
    return None


def _raw_user_message(messages: list) -> str:
    return next(
        (m["content"] for m in reversed(messages)
         if m.get("role") == "user" and isinstance(m.get("content"), str)),
        "",
    )


async def _score(user_message: str, contexts: list[str], answer: str) -> float | None:
    last_error = None
    for attempt in range(SCORE_RETRY_ATTEMPTS + 1):
        try:
            result = await asyncio.wait_for(
                _scorer.ascore(user_input=user_message, response=answer, retrieved_contexts=contexts),
                timeout=JUDGE_TIMEOUT_SECONDS,
            )
            raw = result.value
            # RAGAS returns NaN when the answer contains no checkable claims
            # (e.g. an honest "I don't know") — treat that as fully grounded.
            return 1.0 if np.isnan(raw) else float(raw)
        except asyncio.TimeoutError:
            last_error = f"timed out after {JUDGE_TIMEOUT_SECONDS}s"
        except Exception as e:
            last_error = str(e)
        if attempt < SCORE_RETRY_ATTEMPTS:
            logger.warning(f"[faithfulness] scoring attempt {attempt + 1} failed ({last_error}), retrying")
    logger.warning(f"[faithfulness] scoring failed after {SCORE_RETRY_ATTEMPTS + 1} attempt(s): {last_error}")
    return None


class FaithfulnessGuardrail(CustomGuardrail):
    """Faithfulness guardrail. Subclasses may override _failure_reason() to
    attach a human-readable explanation to blocked/exhausted outcomes (see
    faithfulness_guardrail_reasoning.FaithfulnessReasoningGuardrail)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def _failure_reason(self, user_message: str, contexts: list[str],
                              answer: str, score: float) -> str | None:
        """Optional explanation of WHY an answer failed. Base returns None
        (no reason generated, no extra judge calls)."""
        return None

    async def async_post_call_success_hook(self, data: dict, user_api_key_dict: UserAPIKeyAuth, response):
        # Recursion guard 1: never re-score the judge's own decompose/verify
        # calls (they re-enter this proxy).
        if data.get("model") == JUDGE_MODEL:
            return response

        # Recursion guard 2: never re-score our own regeneration calls.
        if _is_internal_retry(data):
            return response

        if not isinstance(response, litellm.ModelResponse):
            return response

        try:
            answer = response.choices[0].message.content
        except (AttributeError, IndexError):
            return response
        if not answer:
            return response

        # Strip any reasoning trace before it's used for anything — the
        # judge and the caller must both see only the actual final answer.
        answer = _strip_reasoning(answer)
        response.choices[0].message.content = answer
        if not answer:
            return response

        req_id = str(uuid.uuid4())[:8]
        messages = data.get("messages", [])
        target_model = data.get("model")

        contexts = _extract_contexts(messages)
        user_message = _raw_user_message(messages)
        context_text = "\n---\n".join(contexts) if contexts else None

        if contexts is None:
            # No evidence-marked context: non-RAG traffic (or a RAG app not
            # yet on the message structure). Pass through, but record it so
            # unguarded traffic is visible on the dashboard.
            logger.info(f"[faithfulness] req={req_id} no evidence-marked context found, passing unscored")
            await _log_event(req_id, 0, None, "unscored", target_model,
                             user_message, context_text, answer)
            return response

        threshold = PASS_THRESHOLD
        on_fail = DEFAULT_ON_FAIL if DEFAULT_ON_FAIL in ("block", "retry") else "retry"

        score = await _score(user_message, contexts, answer)
        if score is None:
            # Fail open on judge/scoring errors so the proxy never hard-fails
            # traffic because of a scoring-infrastructure problem. This is a
            # deliberate policy choice — flip here if fail-closed is required.
            await _log_event(req_id, 0, None, "error", target_model,
                             user_message, context_text, answer)
            return response

        logger.info(f"[faithfulness] req={req_id} attempt=0 score={score:.3f} threshold={threshold} on_fail={on_fail}")

        if score >= threshold:
            await _log_event(req_id, 0, score, "passed", target_model,
                             user_message, context_text, answer)
            return response

        if on_fail == "block":
            reason = await self._failure_reason(user_message, contexts, answer, score)
            await _log_event(req_id, 0, score, "blocked", target_model,
                             user_message, context_text, answer, reason)
            detail = {
                "error": "Response failed faithfulness guardrail",
                "guardrail": "faithfulness-guardrail",
                "score": round(score, 3),
                "threshold": threshold,
            }
            if reason:
                detail["reason"] = reason
            raise HTTPException(status_code=400, detail=detail)

        # on_fail == "retry": regenerate with corrective feedback.
        attempt_messages = list(messages)
        attempt = 0
        while True:
            if attempt >= MAX_RETRIES:
                logger.warning(f"[faithfulness] req={req_id} EXHAUSTED {MAX_RETRIES} retries, final score={score:.3f}")
                reason = await self._failure_reason(user_message, contexts, answer, score)
                await _log_event(req_id, attempt, score, "exhausted", target_model,
                                 user_message, context_text, answer, reason)
                fallback = FALLBACK_TEXT
                if reason:
                    fallback = f"{FALLBACK_TEXT} ({reason})"
                response.choices[0].message.content = fallback
                return response

            reason = await self._failure_reason(user_message, contexts, answer, score)
            await _log_event(req_id, attempt, score, "retrying", target_model,
                             user_message, context_text, answer, reason)
            attempt += 1
            feedback = RETRY_FEEDBACK
            if reason:
                feedback = f"{RETRY_FEEDBACK} Specifically: {reason}"
            attempt_messages = attempt_messages + [
                {"role": "assistant", "content": answer},
                {"role": "user", "content": feedback},
            ]
            try:
                regen = await _proxy_client.chat.completions.create(
                    model=target_model, messages=attempt_messages, temperature=0,
                    extra_body={"metadata": {"ragas_internal_retry": True}},
                )
                answer = _strip_reasoning(regen.choices[0].message.content)
            except Exception as e:
                logger.warning(f"[faithfulness] req={req_id} regeneration call failed: {e}")
                await _log_event(req_id, attempt, None, "error", target_model,
                                 user_message, context_text, answer)
                response.choices[0].message.content = FALLBACK_TEXT
                return response

            score = await _score(user_message, contexts, answer)
            if score is None:
                await _log_event(req_id, attempt, None, "error", target_model,
                                 user_message, context_text, answer)
                response.choices[0].message.content = answer
                return response
            logger.info(f"[faithfulness] req={req_id} attempt={attempt} score={score:.3f}")

            if score >= threshold:
                logger.info(f"[faithfulness] req={req_id} PASSED at attempt={attempt} score={score:.3f}")
                await _log_event(req_id, attempt, score, "passed", target_model,
                                 user_message, context_text, answer)
                response.choices[0].message.content = answer
                return response
