# LiteLLM Eval Harness — Setup & Integration Guide

This adds a **continuous evaluation layer** around the existing LiteLLM proxy
(and its RAGAS faithfulness guardrail):

```
End user → RAG app → LiteLLM proxy ──► LLM provider
                        │   (inline RAGAS guardrail, unchanged)
                        │
                        └─ langfuse callback (async, non-blocking)
                                 │
                                 ▼
                      Langfuse (self-hosted)  ◄── scores pushed back ──┐
                        traces of ALL traffic                          │
                                 │                                     │
                                 └── eval-worker: samples traces, ─────┘
                                     scores with DeepEval
                                     (judge = "judge-model" via the proxy)

Separately, before any change ships:
  golden dataset (evals/golden_dataset.json)
    → DeepEval in CI (.github/workflows/evals.yml)
    → PR blocked if faithfulness drops below threshold
```

Everything is free / open source: self-hosted Langfuse (MIT-licensed core),
DeepEval (Apache 2.0). No Confident AI account is used anywhere.

## Components in this repo

| Path | What it is |
|---|---|
| `docker-compose.yml` | Full stack: LiteLLM + its Postgres + Langfuse (web, worker, ClickHouse, Redis, MinIO, Postgres) + eval-worker |
| `litellm_config.yaml` | Proxy config: models, guardrail, `success_callback: ["langfuse"]` |
| `litellm_image/Dockerfile` | LiteLLM image + ragas/langfuse deps (replaces `litellm-hhem:latest`) |
| `eval_worker/` | The online eval service (DeepEval + Langfuse REST) |
| `evals/golden_dataset.json` | Regression test cases (grow this from real traffic) |
| `evals/test_golden_dataset.py` | DeepEval CI gate, answers every case through the proxy |
| `.github/workflows/evals.yml` | CI: deterministic pipeline tests + real-model golden evals |
| `tests/` | Deterministic local verification harness (mock LLM, Langfuse stub, e2e tests) |

## Running it on your machine

```bash
git clone <this repo> && cd litellm_guardrail_integeration
cp .env.example .env        # put your real GROQ_API_KEY in .env
docker compose up -d --build
```

First boot takes a few minutes (ClickHouse migrations). Then:

1. **Langfuse UI**: http://localhost:3000 — log in with
   `admin@example.com` / `changeme123` (from `.env`). The project and API
   keys already exist (headless init).
2. **Send traffic through the proxy**:
   ```bash
   curl http://localhost:4000/v1/chat/completions \
     -H "Authorization: Bearer sk-1234" -H "Content-Type: application/json" \
     -d '{
       "model": "groq-llama-3.1-8b",
       "messages": [
         {"role": "system", "content": "Context: The AeroBook X200 battery lasts 12 hours."},
         {"role": "user", "content": "How long does the battery last?"}
       ],
       "metadata": {"context": ["The AeroBook X200 battery lasts 12 hours."]}
     }'
   ```
3. **Watch the trace appear** in Langfuse (Tracing → Traces) within seconds.
4. **Watch the score appear**: within `EVAL_POLL_INTERVAL_SECONDS` the
   eval-worker attaches `deepeval_faithfulness` (and
   `deepeval_answer_relevancy`) scores to sampled traces. For a demo, set
   `EVAL_SAMPLE_RATE=1.0` in `.env` so every trace is scored:
   `docker compose up -d eval-worker` after editing.
5. `docker compose logs -f eval-worker` shows each scoring decision.

## How consuming apps integrate

Nothing changes for app teams — they call the proxy as before. Two optional
conventions make their eval scores more accurate:

* **Pass retrieval context explicitly** (preferred; also the planned fix for
  the guardrail's context-extraction problem):
  ```python
  client.chat.completions.create(
      model="groq-llama-3.1-8b",
      messages=[...],
      extra_body={"metadata": {"context": ["chunk 1", "chunk 2"]}},
  )
  ```
  The worker prefers `metadata.context`; otherwise it falls back to
  system-role messages. Traces with neither are skipped (never scored
  against empty context).
* **One LiteLLM virtual key per team/app**, so Langfuse dashboards can be
  filtered per team.

## Eval worker behaviour (eval_worker/worker.py)

* Polls Langfuse for recent traces (`EVAL_LOOKBACK_MINUTES` window).
* **Sampling**: deterministic hash of trace id vs `EVAL_SAMPLE_RATE` —
  judge cost scales linearly with this; the keep/skip decision for a trace
  never changes across restarts.
* **Never scores internal traffic**: judge-model calls (matched via the
  generation's model/`model_group`) and guardrail regeneration calls
  (`metadata.ragas_internal_retry`) — the same recursion guards as the
  guardrail.
* **Idempotent**: a trace already carrying `deepeval_faithfulness` (or the
  error marker `deepeval_scoring_error`) is never re-scored.
* Judge calls go through the LiteLLM proxy (`judge-model` route), so
  swapping to a self-hosted judge later is a proxy-config change only.

## CI gate

`.github/workflows/evals.yml` has two jobs:

* **pipeline-e2e** — runs on every PR with zero secrets. Spins up the
  deterministic mock stack and proves the whole pipeline: capture → worker →
  scoring → score push, correct skip logic, idempotency, sampling, and that
  the golden-dataset gate fails exactly the deliberately-fabricated cases.
* **golden-evals** — runs when the `GROQ_API_KEY` repo secret is set:
  starts the proxy with real Groq models and runs
  `evals/test_golden_dataset.py`; any case below the faithfulness threshold
  (0.7, same as the guardrail) fails the build.

Grow `evals/golden_dataset.json` over time: every hallucination found in
Langfuse (low `deepeval_faithfulness`) or reported by a user becomes a new
row. That is the flywheel: production failures become permanent regression
tests.

## Local verification without Docker (what CI's first job does)

```bash
python3 -m venv venv-proxy  && venv-proxy/bin/pip install 'litellm[proxy]' 'langfuse>=2,<3'
python3 -m venv venv-worker && venv-worker/bin/pip install -r eval_worker/requirements.txt pytest
PROXY_VENV=$PWD/venv-proxy ./tests/run_stack.sh /tmp/stack
WORKER_PYTHON=$PWD/venv-worker/bin/python venv-worker/bin/python -m pytest -q tests/test_e2e_pipeline.py
WORKER_PYTHON=$PWD/venv-worker/bin/python bash tests/check_golden_harness.sh
```

## Production hand-off checklist (for the infra team)

* Rotate every secret in `.env` (Langfuse keys, `SALT`, `ENCRYPTION_KEY`
  via `openssl rand -hex 32`, `NEXTAUTH_SECRET`, admin password, master key).
* `EVAL_SAMPLE_RATE`: start at 0.1–0.2; each sampled trace costs ~6–8 judge
  calls across the two metrics.
* The old FastAPI dashboard reading `ragas_events` still works (the
  guardrail is unchanged); Langfuse dashboards supersede it for eval scores.
* Known limitations carried over from the guardrail (documented in the
  project brief): context extraction convention and multi-turn question
  extraction. The `metadata.context` contract in this harness is the
  migration path for both.
* Not yet wired (recommended follow-up): pushing the guardrail's own RAGAS
  scores onto the same Langfuse trace (name it `ragas_faithfulness_inline`)
  so inline and async scores are comparable on one screen.
