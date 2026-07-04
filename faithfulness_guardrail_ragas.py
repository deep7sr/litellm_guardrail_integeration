import os
import logging
import uuid
import asyncio
import asyncpg
import numpy as np
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics import Faithfulness
from ragas.dataset_schema import SingleTurnSample
import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth

logger = logging.getLogger("faithfulness_guardrail_ragas")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

_proxy_client = AsyncOpenAI(
    base_url="http://localhost:4000/v1",
    api_key=os.environ.get("LITELLM_MASTER_KEY", "sk-1234"),
)
_judge_llm = llm_factory("judge-model", client=_proxy_client)
_scorer = Faithfulness(llm=_judge_llm)

PASS_THRESHOLD = 0.7
MAX_RETRIES = 3

RETRY_FEEDBACK = (
    "That answer was not fully grounded in the context. Regenerate it using ONLY "
    "the context above. If the context doesn't contain the answer, say so explicitly."
)
FALLBACK_TEXT = "I wasn't able to find a fully supported answer to this in the available documents."

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


async def log_event(req_id, attempt, score, verdict, target_model, question, answer):
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


async def score_pair(question, context, answer):
    sample = SingleTurnSample(user_input=question, response=answer, retrieved_contexts=[context])
    try:
        score = await _scorer.single_turn_ascore(sample)
    except Exception as e:
        logger.warning(f"[ragas] scoring failed: {e}")
        return None
    return 1.0 if np.isnan(score) else float(score)


def build_context(messages):
    # KNOWN ISSUE: system-role only. See "Open problems #1" in the project brief.
    return "\n".join(
        m["content"] for m in messages
        if m.get("role") == "system" and isinstance(m.get("content"), str)
    )


class RagasFaithfulnessGuardrail(CustomGuardrail):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def async_post_call_success_hook(self, data: dict, user_api_key_dict: UserAPIKeyAuth, response):
        # Never re-score the judge's own decompose/verify calls.
        if data.get("model") == "judge-model":
            return response

        # Never re-score our own regeneration calls.
        if data.get("metadata", {}).get("ragas_internal_retry"):
            return response

        if not isinstance(response, litellm.ModelResponse):
            return response

        try:
            answer = response.choices[0].message.content
        except (AttributeError, IndexError):
            return response

        req_id = str(uuid.uuid4())[:8]
        messages = data.get("messages", [])
        target_model = data.get("model")
        context_blob = build_context(messages)
        # KNOWN ISSUE: only grabs the most recent user message. See "Open problems #2".
        question = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")

        score = await score_pair(question, context_blob, answer)
        if score is None:
            return response

        logger.info(f"[ragas] req={req_id} attempt=0 score={score:.3f}")
        attempt_messages = list(messages)
        attempt = 0

        while True:
            if score >= PASS_THRESHOLD:
                logger.info(f"[ragas] req={req_id} PASSED at attempt={attempt} score={score:.3f}")
                await log_event(req_id, attempt, score, "passed", target_model, question, answer)
                if attempt > 0:
                    response.choices[0].message.content = answer
                return response

            if attempt >= MAX_RETRIES:
                logger.warning(f"[ragas] req={req_id} EXHAUSTED {MAX_RETRIES} retries, final score={score:.3f}")
                await log_event(req_id, attempt, score, "exhausted", target_model, question, answer)
                response.choices[0].message.content = FALLBACK_TEXT
                return response

            await log_event(req_id, attempt, score, "retrying", target_model, question, answer)

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
                await log_event(req_id, attempt, None, "error", target_model, question, answer)
                response.choices[0].message.content = FALLBACK_TEXT
                return response

            score = await score_pair(question, context_blob, answer)
            if score is None:
                response.choices[0].message.content = answer
                return response
            logger.info(f"[ragas] req={req_id} attempt={attempt} score={score:.3f}")
