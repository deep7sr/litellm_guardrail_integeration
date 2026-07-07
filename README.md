# LiteLLM Faithfulness Guardrails (RAGAS-based)

Hallucination-detection guardrails for the central LiteLLM proxy. Every RAG
application routed through the proxy gets its responses verified for
groundedness against the retrieved context — scored claim-by-claim by a
separate judge model (RAGAS Faithfulness) — before the response reaches the
end user.

**Branch purpose (`VM_faithfulness_Guardrail`):** the production line of the
guardrail as deployed and validated on the office VM. Two variants ship here:

| File | Class | What it adds |
|---|---|---|
| `guardrail/faithfulness_guardrail.py` | `FaithfulnessGuardrail` | v1 — detection + retry/block enforcement |
| `guardrail/faithfulness_guardrail_reasoning.py` | `FaithfulnessReasoningGuardrail` | v2 — v1 plus a judge-written explanation of WHY a failed answer was blocked/replaced (returned to the caller, used to sharpen retry feedback, logged for the dashboard) |

v2 subclasses v1 — same detection and enforcement, one extension point
(`_failure_reason`). **Enable exactly one of them** in the LiteLLM config;
both fire on `post_call`, so enabling both double-scores every request.

---

## The contract: standard OpenAI RAG message structure (no metadata)

The guardrail is transparent to developers. There is **no metadata field, no
SDK, no custom parameter** — apps format requests the standard OpenAI way:

```python
context_str = "--- Retrieved Evidence ---\n" + "\n\n".join(retrieved_chunks)

messages = [
    {"role": "system",    "content": "You are a factual assistant. Use ONLY the retrieved context..."},
    {"role": "assistant", "content": context_str},   # retrieved context
    {"role": "user",      "content": user_question}, # the question
]
```

The guardrail reads:
- **Context** = the last `assistant`-role message carrying the
  `--- Retrieved Evidence ---` marker (the marker is OpenAI's own documented
  context format, not something we invented). The marker is what
  distinguishes injected context from prior model answers in multi-turn
  conversations — both are `assistant`-role, so role alone can't.
- **Question** = the last `user`-role message.

**Security invariant:** context is only ever read from `assistant`-role
messages. A user typing the marker into their own message cannot inject
fabricated context — validated by test and live on the VM.

Requests with no marked evidence message pass through **unscored** (logged,
never blocked) — non-RAG apps need to do nothing and are never broken.

> Historical note: an earlier iteration used an explicit
> `metadata.guardrail_context` contract. That mechanism was removed after
> team review in favor of the message structure above (zero developer
> friction). The security principle — context is app-controlled, never
> inferred from or trusted out of user text — is unchanged; see
> `docs/WHY_CONTEXT_CONTRACT.md` for the original evidence and industry
> survey behind it.

## Enforcement flow

```
answer generated
  → strip <think>…</think> reasoning trace (some models emit it)
  → extract context/question from messages
  → no marked context?  → serve unscored (logged)
  → RAGAS Faithfulness score via JUDGE_MODEL (bounded by JUDGE_TIMEOUT_SECONDS)
  → score ≥ threshold?  → serve answer                    [passed]
  → GUARDRAIL_MODE=block → structured 400 w/ score (+reason in v2)  [blocked]
  → GUARDRAIL_MODE=retry → regenerate with corrective feedback,
       re-score, up to MAX_ATTEMPTS                        [retrying…]
     → recovers → serve corrected answer                   [passed]
     → still failing → serve fixed fallback text (+reason in v2)  [exhausted]
```

Recursion guards (both load-bearing — the judge's scoring calls and our own
regeneration calls re-enter this same proxy):
1. requests whose `model == JUDGE_MODEL` are never scored;
2. requests tagged `ragas_internal_retry` (our own regenerations) are never
   scored.

## Configuration (central, env vars — no per-request overrides)

| Variable | Default | Meaning |
|---|---|---|
| `GUARDRAIL_THRESHOLD` | `0.7` | Minimum faithfulness score (fraction of answer claims supported by context). |
| `GUARDRAIL_MODE` | `retry` | `retry` (regenerate then fallback) or `block` (fail fast with 400). |
| `MAX_ATTEMPTS` | `3` | Regeneration budget in retry mode. |
| `JUDGE_MODEL` | `judge-model` | Model-list name of the judge; its traffic is never re-scored. |
| `JUDGE_TIMEOUT_SECONDS` | `60` | Bound on each scoring/reason step — a hung judge cannot stall the proxy. |
| `PROXY_BASE_URL` | `http://localhost:4000/v1` | How the guardrail calls back into this proxy (judge + regenerations). |
| `EVIDENCE_MARKER` | `--- Retrieved Evidence ---` | Context marker; change only org-wide, in lockstep with app templates. |
| `DATABASE_URL` | — | Postgres for the `faithfulness_events` audit table (optional; logging disabled if unset). |
| `GIT_PYTHON_REFRESH` | set to `quiet` | Required: RAGAS transitively imports GitPython, which errors without a git binary. |

## Observability

Every attempt is logged to Postgres (`faithfulness_events`): request id,
attempt number, score, verdict, model, raw user message, scoring question,
retrieved context, generated answer, and (v2) the judge's failure reason.
The `dashboard/` service renders it live at `:8080`.

Verdicts: `passed` · `blocked` · `retrying` · `exhausted` ·
`unscored` (no marked context — expected for non-RAG traffic) ·
`error` (scoring infrastructure failure — **fails open by design**; flip in
`_score`-error handling if policy requires fail-closed).

## Test status — read this before deploying

| What | How verified | Status |
|---|---|---|
| Context extraction, marker rules, injection safety, multi-turn disambiguation | 23-case pytest suite (`tests/`), real `litellm.ModelResponse` objects, mocked judge | ✅ passing |
| Enforcement paths: pass / block / retry-correct / exhaustion-fallback / fail-open / both recursion guards | same suite | ✅ passing |
| v2 reasoning: reason in 400 detail, reason in fallback, specific retry feedback, graceful degradation when reason step fails | same suite | ✅ passing |
| Live end-to-end scoring (real judge, real Groq models) — v1 behavior | validated on the office VM across three domains (drone/SaaS/insurance): clean pass 1.000, injection neutralized, forced hallucination 0.000→caught→self-corrected, multi-turn marker disambiguation, unscored passthrough | ✅ done (pre-merge form of this code) |
| Live end-to-end — this exact merged code + v2 reasoning variant | **pending — run `scripts/demo_test.py` on the VM after deploying this branch** | ⚠️ required before production cutover |
| Threshold calibration against labeled data | not done — `0.7` is empirical | ⚠️ open |

Run the unit tests: `pip install pytest pytest-asyncio && python -m pytest tests/`

## Deploying on the VM

1. Copy both `guardrail/*.py` files into the directory mounted at
   `/app/litellm` (or adjust the compose mounts as in `docker-compose.yml`).
2. In the LiteLLM config's `guardrails:` section, register **one** class —
   see `litellm_config.yaml` for both stanzas (v2 enabled by default there).
3. Ensure the runtime image has `ragas==0.4.3`, `langchain-community==0.4.1`,
   `asyncpg` installed (the production image was built by layering these on
   the LiteLLM image actually running against the existing database — do not
   change LiteLLM versions casually; mismatched Prisma migrations against a
   shared Postgres was a real failure we hit).
4. Restart the litellm service, confirm registration via
   `GET /guardrails/list`, then run `scripts/demo_test.py`.

## Progress log

- **v0 (POC)** — metadata-based contract, in-process RAGAS, retry loop,
  recursion guards; validated on a laptop stack.
- **Fixes along the way** — RAGAS 0.4 API migration
  (`ragas.metrics.collections.Faithfulness`, `ascore(...)→MetricResult`),
  GitPython import crash (`GIT_PYTHON_REFRESH=quiet`), corporate-TLS
  workarounds, LiteLLM image/DB-migration version pinning, reasoning-model
  `<think>` leakage stripped from scoring and serving.
- **Contract pivot** — after team review: metadata contract removed
  entirely; OpenAI message structure with the evidence marker is the sole
  mechanism. Validated live on the VM (single-turn, multi-turn, marker
  injection).
- **This branch** — production v1 + v2(reasoning), judge-call timeouts,
  full event logging incl. raw user message and failure reason, dashboard,
  23-case test suite, docs rewritten for the message-structure contract,
  all metadata-era files removed.

## Known limitations / next steps

- **One VM smoke run required** for this exact merged code (see test matrix).
- **Threshold is uncalibrated** — derive it from a labeled golden dataset
  (an internal DeepEval golden-dataset harness likely already exists; use it).
- **Fail-open on scoring errors** is a deliberate but unratified policy —
  get an explicit fail-open vs. fail-closed decision for production.
- **No streaming support** — guarded routes must be non-streaming.
- **v2 cost**: one extra judge call per *failed* attempt (not per request).
- **Internal tag is not authenticated** — a caller inside the org could set
  `ragas_internal_retry` metadata to skip scoring; equivalent trust level to
  simply omitting the evidence marker, but worth knowing.
- **No alerting** — dashboard only; wire `error`/`unscored`/`exhausted`
  rates into real alerting before broad rollout.
- **Data retention** — full user messages/context/answers are logged to
  Postgres indefinitely; set a retention/redaction policy (GDPR/EU-AI-Act
  traffic already flows through this proxy).
- **Judge is Groq-hosted** — self-host for rate-limit and residency reasons.

The original project brief (full history and the problems this design
answers) is `ragas-guardrail-project-brief.md`.
