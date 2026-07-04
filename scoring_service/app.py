"""
Standalone faithfulness scoring service.

Owns all RAGAS + judge-model logic so the LiteLLM proxy stays thin.
The proxy's guardrail hook calls:

  POST /score   {question, contexts, answer}          -> {score}
  POST /events  {req_id, attempt, score, verdict, ...} -> {ok}
  GET  /health

The judge model is reached back through the LiteLLM proxy itself
(JUDGE_MODEL routed via LITELLM_BASE_URL), so all model traffic stays
centrally configured. The guardrail skips scoring judge-model traffic,
so this does not recurse.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

import asyncpg
import numpy as np
from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics import Faithfulness

logger = logging.getLogger("scoring_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://litellm:4000/v1")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "sk-1234")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "judge-model")

_judge_client = AsyncOpenAI(base_url=LITELLM_BASE_URL, api_key=LITELLM_API_KEY)
_judge_llm = llm_factory(JUDGE_MODEL, client=_judge_client)
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
            logger.warning("DATABASE_URL not set, event logging disabled")
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
            logger.info("connected to events DB, table ready")
            _db_pool = pool
        except Exception as e:
            logger.warning(f"could not connect to events DB: {e}")
            _db_pool = None
    return _db_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    await get_db_pool()
    yield
    if _db_pool is not None:
        await _db_pool.close()


app = FastAPI(title="Faithfulness Scoring Service", lifespan=lifespan)


class ScoreRequest(BaseModel):
    question: str
    contexts: list[str] = Field(min_length=1)
    answer: str


class ScoreResponse(BaseModel):
    score: float


class EventRequest(BaseModel):
    req_id: str
    attempt: int
    score: float | None = None
    verdict: str
    target_model: str | None = None
    question: str | None = None
    answer_snippet: str | None = None


@app.get("/health")
async def health():
    return {"status": "ok", "judge_model": JUDGE_MODEL}


@app.post("/score", response_model=ScoreResponse)
async def score(req: ScoreRequest):
    sample = SingleTurnSample(
        user_input=req.question,
        response=req.answer,
        retrieved_contexts=req.contexts,
    )
    try:
        raw = await _scorer.single_turn_ascore(sample)
    except Exception as e:
        logger.warning(f"scoring failed: {e}")
        raise HTTPException(status_code=502, detail=f"judge scoring failed: {e}")
    # RAGAS returns NaN when the answer contains no checkable claims
    # (e.g. "I don't know") — treat that as fully grounded.
    value = 1.0 if np.isnan(raw) else float(raw)
    return ScoreResponse(score=value)


@app.post("/events")
async def record_event(event: EventRequest):
    pool = await get_db_pool()
    if pool is None:
        return {"ok": False, "reason": "db unavailable"}
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO ragas_events (req_id, attempt, score, verdict, target_model, question, answer_snippet)
                   VALUES ($1, $2, $3, $4, $5, $6, $7)""",
                event.req_id, event.attempt, event.score, event.verdict,
                event.target_model,
                (event.question or "")[:500], (event.answer_snippet or "")[:500],
            )
    except Exception as e:
        logger.warning(f"req={event.req_id} failed to log event: {e}")
        return {"ok": False, "reason": str(e)}
    return {"ok": True}
