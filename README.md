# LiteLLM Faithfulness Guardrail (RAGAS-based)

A hallucination-detection guardrail for a central LiteLLM proxy. Every RAG
application routed through the proxy gets its responses checked for
groundedness against the retrieved context (RAGAS Faithfulness, scored by a
judge model) before the response reaches the end user.

The architecture mirrors the gateway-plus-guardrail-service pattern
(gateway orchestrates, a separate service evaluates, results land in a DB
behind a dashboard):

```
                       ┌─────────────────────┐
 RAG app ── request ──►│   LiteLLM proxy     │──► target model (Groq/…)
 (sends context +      │  (guardrail hook)   │
  question in          └─────┬──────▲────────┘
  metadata)                  │      │ judge calls routed
                     POST /score    │ back through proxy
                             │      │
                       ┌─────▼──────┴───────┐        ┌───────────┐
                       │  Scoring service    │───────►│ Postgres  │◄── Dashboard
                       │  (RAGAS + judge)    │ events │ (events)  │    (:8080)
                       └─────────────────────┘        └───────────┘
```

## Components

| Service     | What it does |
|-------------|--------------|
| `litellm`   | The gateway. Runs the thin `RagasFaithfulnessGuardrail` post-call hook (`guardrail/faithfulness_guardrail.py`) — extracts the contract fields, calls the scoring service, enforces the verdict. |
| `scoring`   | Standalone FastAPI service (`scoring_service/`) owning all RAGAS + judge-model logic. `POST /score` returns a 0.0–1.0 faithfulness score; `POST /events` records verdicts. |
| `dashboard` | Auto-refreshing view of pass/block/retry/unscored rates and recent events (testing/demo use). |
| `db`        | Postgres — LiteLLM's own state plus the `ragas_events` table. |

## The integration contract (important)

The guardrail does **not** infer context from message roles. Inferring
context from the `messages` array is unsafe (a user can phrase fabricated
claims as "established context" inside their own message and the guardrail
would validate the fabrication against itself — a confirmed vulnerability in
an earlier version) and unreliable (apps pack context into system messages,
user messages, or single blended strings — no heuristic covers all of them).

Instead, RAG apps **must pass the retrieved context and originating question
explicitly** via request metadata:

```python
client.chat.completions.create(
    model="groq-llama-3.1-8b",
    messages=[...],                     # prompt however you like
    extra_body={"metadata": {
        "guardrail_context": ["chunk 1", "chunk 2"],  # required for scoring
        "guardrail_question": "the user's question",  # recommended
        "guardrail_on_fail": "block" | "retry",       # optional override
        "guardrail_threshold": 0.7,                   # optional override
    }},
)
```

Requests **without** `guardrail_context` pass through unscored and are logged
with verdict `unscored`, so unguarded traffic is visible on the dashboard and
can be chased down per team.

See `examples/rag_client_example.py` for a complete raw-format client, or —
preferred — use the internal helper in `sdk/company_llm.py`
(`guarded_completion`), which builds the metadata and converts guardrail
blocks into a typed `GuardrailBlockedError`.

Developer-facing docs:
- `docs/INTEGRATION.md` — the one-pager to hand to application teams.
- `docs/WHY_CONTEXT_CONTRACT.md` — full justification for the contract, with
  our incident evidence and industry citations (AWS/Azure/NVIDIA/OWASP).

## Fail behavior: `block` vs `retry`

Configured globally via `RAGAS_ON_FAIL` (default `block`), overridable
per-request via `guardrail_on_fail`:

- **`block`** — score below threshold ⇒ the request fails immediately with a
  structured 400 error (`{"error": ..., "score": ..., "threshold": ...}`).
  Cheap, fast, predictable; the calling app decides how to present it.
- **`retry`** — regenerate with corrective feedback and re-score, up to
  `RAGAS_MAX_RETRIES` times; if still failing, a fixed fallback message is
  served instead of the low-scoring answer. Better UX when it works, but
  worst case adds several generations + scorings of latency.

## Configuration

Environment variables on the `litellm` service (see `docker-compose.yml`):

| Variable | Default | Meaning |
|---|---|---|
| `SCORING_SERVICE_URL` | `http://scoring:8000` | Where the guardrail sends `/score` and `/events`. |
| `RAGAS_PASS_THRESHOLD` | `0.7` | Minimum faithfulness score to pass. |
| `RAGAS_ON_FAIL` | `block` | Default fail behavior (`block`/`retry`). |
| `RAGAS_MAX_RETRIES` | `3` | Retry budget when in `retry` mode. |
| `JUDGE_MODEL` | `judge-model` | Model-list name of the judge; guardrail skips scoring its traffic (prevents recursion). |

On the `scoring` service: `LITELLM_BASE_URL`, `LITELLM_API_KEY`,
`JUDGE_MODEL`, `DATABASE_URL`.

The judge is defined once in `litellm_config.yaml` as `judge-model`; to move
to a self-hosted judge later, only that model entry changes — the guardrail
and scoring service are untouched.

## Running

```bash
export GROQ_API_KEY=...          # current POC judge + target model host
docker compose up --build
```

- Proxy: `http://localhost:4000` (master key from `LITELLM_MASTER_KEY`, default `sk-1234`)
- Dashboard: `http://localhost:8080`
- Scoring service health: `http://scoring:8000/health` (inside the compose network)

## Event verdicts

Each scoring attempt is logged to `ragas_events`:

| Verdict | Meaning |
|---|---|
| `passed` | Score ≥ threshold, response served. |
| `blocked` | `block` mode, score below threshold, 400 returned. |
| `retrying` | `retry` mode, attempt failed, regeneration issued. |
| `exhausted` | `retry` mode, retries used up, fallback text served. |
| `unscored` | Caller didn't supply `guardrail_context`. |
| `error` | Scoring or regeneration infrastructure failure (fails open). |

## Design notes

- **Fail-open on infrastructure errors**: if the scoring service is down, the
  proxy serves responses unscored (logged as `error`) rather than taking all
  traffic down with it. Flip this deliberately if policy requires fail-closed.
- **Recursion guards**: the guardrail skips (a) any request whose model is the
  judge, and (b) any request tagged `ragas_internal_retry` (its own
  regeneration calls). Preserve both in any refactor.
- **NaN scores** from RAGAS (answers with no checkable claims, e.g. honest
  "I don't know") are treated as fully grounded (1.0).

The original project brief, including the full history and open problems this
design addresses, is in `ragas-guardrail-project-brief.md`.
