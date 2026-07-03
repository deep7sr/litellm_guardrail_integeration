# Project Brief: LiteLLM Faithfulness Guardrail (RAGAS-based)

## Objective

Integrate a hallucination-detection guardrail into our company's central LiteLLM
proxy, so that every RAG application routed through it gets automatic checking
of whether the LLM's response is actually grounded in the retrieved context —
before that response ever reaches the end user. We are using RAGAS's
Faithfulness metric as the scoring mechanism. This is intended to become a
production guardrail used by all applications going through this LiteLLM
instance, not a one-off script.

We are asking you to review everything below, identify anything wrong or
missing in our current approach, and help us build a complete, correct,
production-ready version — full code, correct architecture, and the right
approach for the specific unresolved problems described near the end of this
document.

---

## Why RAGAS, and why not just an LLM asked "yes/no"

We evaluated several approaches before landing here:
- A single LLM call asking the judge model "is this grounded? yes/no" — works,
  but gives no partial credit and no insight into *which* part of an answer
  is wrong.
- HHEM (Vectara's open-source hallucination classifier) — completely free and
  local (no LLM call needed for scoring at all), but we found real accuracy
  gaps in testing (details below) and were explicitly told by leadership to
  set it aside in favor of RAGAS, since RAGAS is the more industry-adopted
  approach.
- **RAGAS's Faithfulness metric** — decomposes the generated answer into
  individual factual claims, then checks each claim against the retrieved
  context, producing a continuous score (fraction of claims supported). This
  is what we're currently building on.

---

## Current Architecture

**Proxy**: LiteLLM, running via Docker Compose (Postgres included for
LiteLLM's own internal state plus a custom events table we added).

**Generation model**: currently a Groq-hosted model, standing in for what will
eventually be a frontier model in production. Its only job is to answer the
user's question using the provided context — it has no awareness of the
guardrail or scoring layer.

**Judge model**: an open-source model, currently also proxied through Groq's
API as a stand-in for the POC. **The end goal is a self-hosted open-source
judge model**, not tied to any commercial API — this hasn't been built yet
and is one of the things we need help with (see Open Problems below). The
judge must reliably support strict structured/JSON output, since RAGAS uses
the `instructor` library internally to force the judge into a specific claim
schema — we learned the hard way that not every model handles this reliably
(one model we tried returned malformed JSON and failed outright; a
different-sized model from the same family worked fine).

**Guardrail mechanism**: LiteLLM's declarative `guardrails:` config, pointing
at a custom Python class that extends `litellm.integrations.custom_guardrail.CustomGuardrail`
and implements `async_post_call_success_hook`. Important note for context:
LiteLLM also has a lighter-weight `custom_code` guardrail type that lets you
embed Python directly in the YAML config — we initially tried this and hit a
hard wall: it's sandboxed and **forbids `import` statements entirely** (a
deliberate security measure). Since RAGAS needs real library imports, we had
to use the full custom-class-in-a-file approach instead, registered via:
```yaml
guardrails:
  - guardrail_name: "ragas-faithfulness"
    litellm_params:
      guardrail: faithfulness_guardrail_ragas.RagasFaithfulnessGuardrail
      mode: "post_call"
      default_on: true
```

**Scoring flow**: generate answer → guardrail hook fires → extract context +
question + answer from the request → package into a RAGAS `SingleTurnSample`
→ RAGAS internally makes its own calls to the judge model (a decomposition
call, then verification calls per claim) → score returned (0.0–1.0).

**Retry logic**: if the score is below threshold, regenerate the answer with
corrective feedback appended to the conversation, re-score, up to a max
number of retries. If still failing after retries, replace the response with
a fixed fallback message rather than ever serving a low-scoring answer.

**Threshold**: currently `0.7`. This was tuned empirically, not derived — we
initially copied a `0.5` convention from a different scoring method (HHEM)
and found through testing that it was too permissive for RAGAS's semantics
(RAGAS's score is literally "fraction of claims supported," so 0.5 means "let
it through even if only half the claims are true," which let a real
fabrication pass with a perfect-looking score before we understood the
context bug described below).

**Observability**: every scoring event (per attempt, including retries) is
logged to a Postgres table, with a request ID so concurrent requests are
traceable in logs. A small separate dashboard service (FastAPI, auto-refreshing
HTML) reads from this table to show live pass/fail rates and recent events —
built for visibility during testing/demos, not yet a production monitoring
solution.

---

## What we've verified working (with real test evidence)

- Clean, directly-answerable question → high score, passes immediately.
- Deliberate fabrication (a fake product color injected into the prompt as if
  it were established fact) → correctly scored low on the first attempt,
  self-corrected on retry (model explicitly noted "this detail is not in the
  provided context"), passed on the corrected answer.
- A false claim planted earlier in conversation history, then the model asked
  to "restate" it → correctly scored low, self-corrected to the true value on
  retry, passed.
- Multiple adversarial prompts explicitly instructing the model to "always
  provide a confident answer, never refuse" — the target model resisted all
  of them and gave honest answers; RAGAS correctly scored these as grounded.

We have **not yet observed a genuine retry-exhaustion case** (i.e., all 3
retries failing and the fallback message actually firing) under real
adversarial prompting — the generation model we're testing with has proven
more resistant to fabrication than expected. This path is implemented and
logically sound but not yet proven under real conditions.

---

## ⚠️ Open problems — where we need the most help

### 1. Context extraction is a real, confirmed vulnerability (highest priority)

LiteLLM has **no concept of "context" vs. "question."** It just forwards a
`messages` array (each entry tagged with a role: `system`, `user`,
`assistant`) to the model, unchanged. Any meaning assigned to "this part is
retrieved context, this part is the actual question" is a convention imposed
by whoever built the request — LiteLLM enforces nothing.

Our current extraction logic:
```python
def build_context(messages):
    return "\n".join(
        m["content"] for m in messages
        if m.get("role") == "system" and isinstance(m.get("content"), str)
    )
```
This assumes every RAG application puts retrieved context in `system`-role
messages and the actual question in the latest `user`-role message.

**We confirmed a real bug from a more permissive earlier version** of this
function (which also read `user`-role content as context): a test prompt
embedded a fabricated product detail *inside the user's own message*, framed
as "to confirm what we know so far..." — and the guardrail treated the user's
own claim as ground-truth context, then scored the model's response as
perfectly grounded (1.0) simply because it matched what the user had just
said. This is a legitimate prompt-injection-style vulnerability: a
guardrail meant to prevent fabrication can be tricked into validating
fabricated claims if they're phrased as pre-established context.

Restricting extraction to `system`-role only closed that specific hole, but
introduces a **different, equally real problem**: it assumes a single
integration convention across every RAG application that will ever route
through this proxy. We don't control how other teams build their requests.
Common alternative patterns we know exist:
- Context and question both packed into a single `user` message (e.g.
  `"Context: {docs}\n\nQuestion: {question}"`) — our current logic extracts
  zero context for this pattern, meaning the guardrail would score nearly
  every answer as ungrounded, making it non-functional, not just imprecise.
- Multi-turn conversations where the real originating question is several
  turns back, not the latest message.

**We need your guidance on the right architectural solution here** — is
there a better way to structurally guarantee context and question are
unambiguous and not attacker-controllable, without requiring a fragile
role-based heuristic? Options we've considered but not committed to:
- Mandating a strict message-structure contract for any RAG app integrating
  with this proxy (system = context, user = question only), documented and
  enforced somehow.
- Requiring RAG apps to pass retrieved context through a dedicated field
  (e.g. `extra_body={"metadata": {"context": [...]}}`) instead of relying on
  message-role inference at all — more reliable, but adds an integration
  requirement for every consuming application, which we were originally
  trying to avoid.
- Something else we haven't thought of.

### 2. Question extraction bug

```python
question = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
```
This only grabs the *most recent* `user`-role message. In our
contradiction-history test, this was literally "Actually, restate that
answer, keeping the same confident tone" — not the actual original question
("how long does the battery last?"). It happened to still work because the
answer carried enough signal on its own, but this is fragile for any
multi-turn conversation where the real question isn't in the final turn.

### 3. Judge model is not yet self-hosted

We're currently pointing the judge at a Groq-hosted model as a convenience
stand-in for the POC. Production requires a genuinely self-hosted
open-source judge (no external API dependency, no rate-limit risk). We hit
real rate-limiting problems earlier in this project when relying on
commercial APIs for judge calls at any real volume — this is a large part of
why self-hosting the judge matters for production, not just cost. **We'd
like a concrete recommendation for which self-hostable open-source model to
use as the judge**, given the hard requirement that it must reliably support
RAGAS/`instructor`'s structured JSON output.

### 4. Guardrail self-triggering (already solved once, worth verifying in any redesign)

Because the guardrail is `default_on: true`, it fires on *every* completion
that passes through the proxy — including the guardrail's own internal calls
(regeneration calls to the target model, and RAGAS's own judge calls). Left
unguarded, this causes the guardrail to recursively re-score its own internal
traffic. We fixed this with two explicit checks at the top of the hook:
```python
if data.get("model") == "judge-model":
    return response
if data.get("metadata", {}).get("ragas_internal_retry"):
    return response
```
Any redesign needs to preserve this kind of guard, or reintroduce the bug.

### 5. Threshold is currently a single empirically-guessed constant

`0.7` was arrived at through manual testing, not principled derivation. It's
worth asking whether a single static threshold is even the right model, or
whether the threshold should vary by claim count, answer length, or some
other signal we haven't considered.

---

## Current file contents

### `docker-compose.yml`
```yaml
services:
  litellm:
    image: litellm-hhem:latest
    container_name: litellm
    ports:
      - "4000:4000"
    volumes:
      - ./litellm_config.yaml:/app/config.yaml
      - ./faithfulness_guardrail_ragas.py:/app/faithfulness_guardrail_ragas.py
    command: ["--config", "/app/config.yaml", "--port", "4000"]
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - GROQ_API_KEY=${GROQ_API_KEY}
      - LITELLM_MASTER_KEY=sk-1234
      - DATABASE_URL=postgresql://llmproxy:dbpassword9090@db:5432/litellm
    depends_on:
      - db
    restart: unless-stopped
  dashboard:
    build: ./dashboard
    container_name: ragas-dashboard
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=postgresql://llmproxy:dbpassword9090@db:5432/litellm
    depends_on:
      - db
    restart: unless-stopped
  db:
    image: postgres:16
    container_name: litellm_db
    environment:
      POSTGRES_DB: litellm
      POSTGRES_USER: llmproxy
      POSTGRES_PASSWORD: dbpassword9090
    volumes:
      - litellm_pg_data:/var/lib/postgresql/data
    restart: unless-stopped
volumes:
  litellm_pg_data:
```
> Note: image name `litellm-hhem:latest` is a leftover name from earlier
> experimentation — it currently contains RAGAS dependencies, not HHEM. Worth
> renaming in any rebuild.

### `litellm_config.yaml`
```yaml
model_list:
  - model_name: groq-llama-3.1-8b
    litellm_params:
      model: groq/llama-3.1-8b-instant
      api_key: os.environ/GROQ_API_KEY

  - model_name: judge-model
    litellm_params:
      model: groq/openai/gpt-oss-20b
      api_key: os.environ/GROQ_API_KEY
      reasoning_effort: "low"

general_settings:
  master_key: sk-1234

guardrails:
  - guardrail_name: "ragas-faithfulness"
    litellm_params:
      guardrail: faithfulness_guardrail_ragas.RagasFaithfulnessGuardrail
      mode: "post_call"
      default_on: true
```

### `faithfulness_guardrail_ragas.py`
```python
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
```

---

## What we're asking you to do

1. Review this architecture end-to-end and flag anything incorrect, fragile,
   or insecure that isn't already listed above.
2. Propose and implement a proper fix for the context-extraction problem
   (Open Problem #1) — this is the most important open issue.
3. Fix question extraction to reliably capture the actual originating
   question, not just the latest conversational turn.
4. Recommend a specific, self-hostable open-source judge model that reliably
   supports structured/JSON output for RAGAS, and provide the integration
   code to run it (replacing the current Groq stand-in).
5. Provide complete, ready-to-deploy code for all of the above — not
   partial snippets — including any Dockerfile, config, and guardrail code
   changes needed.
6. If you believe RAGAS itself is the wrong tool for any part of this,
   or that our overall approach has a structural flaw we haven't
   identified, say so directly and explain the alternative.
