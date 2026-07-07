# Langfuse + Async Eval Pipeline — Setup & Runbook

Implements Loop B of `docs/EVALS_ARCHITECTURE_PLAN.md`: every request through
the proxy is traced to **self-hosted Langfuse OSS**; a sidecar **eval runner**
samples recent traces, scores them with **DeepEval** (open-source, local —
never Confident AI), using an **open-source judge routed through the proxy
itself**, and pushes the scores back onto the Langfuse traces.

```
apps ──▶ LiteLLM proxy ──▶ frontier models
              │  success/failure_callback: ["langfuse"]
              ▼
        Langfuse (self-hosted: web+worker+PG+ClickHouse+Redis+MinIO)
              ▲                    │ fetch_traces (lookback window)
              │ scores             ▼
              └──────────── eval-runner (deepeval)
                                   │ judge calls: litellm_proxy/eval-judge
                                   └────────▶ back through the proxy
```

## Components added

| Piece | Where |
|---|---|
| Langfuse v3 stack + eval-runner service | `docker-compose.langfuse.yml` (overlay) |
| Proxy → Langfuse tracing + `eval-judge` model | `litellm_config.yaml` (`litellm_settings`, `model_list`) |
| Eval pipeline code | `evals/` (`config` / `extraction` / `state` / `judge` / `pipeline` / `run_pipeline`) |
| Guardrail skip-set for eval-judge traffic | `GUARDRAIL_SKIP_MODELS` env (guardrail change) |
| Unit tests (fakes, no network) | `tests/test_eval_pipeline.py` |
| Secrets template | `.env.example` |

## Bringing it up

1. **Secrets.** `cp .env.example .env`, fill everything in (`openssl rand
   -hex 32` for salt/encryption/nextauth). The dev defaults boot on a laptop
   but must not reach the VM.

2. **LiteLLM image.** The proxy's `langfuse` callback needs the Langfuse v2
   SDK inside the LiteLLM image. Rebuild the production image with one more
   layer: `pip install "langfuse>=2.54,<3"` (same major the eval runner
   pins — do not mix v2/v3 SDKs across the stack).

3. **Start.**

   ```bash
   docker compose -f docker-compose.yml -f docker-compose.langfuse.yml up -d
   ```

   First boot provisions Langfuse headlessly (`LANGFUSE_INIT_*`): org,
   project `proxy-evals`, your UI login, and the API keypair the proxy and
   eval runner already reference. UI: `http://<host>:3000`.

4. **Budgeted virtual key for the eval runner** (do NOT run it on the master
   key — this is the cost fuse):

   ```bash
   curl -X POST http://localhost:4000/key/generate \
     -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
     -H "Content-Type: application/json" \
     -d '{"key_alias": "eval-runner", "models": ["eval-judge"],
          "max_budget": 10.0, "budget_duration": "30d", "rpm_limit": 30}'
   ```

   Put the returned key in `.env` as `EVAL_PROXY_API_KEY` and
   `docker compose ... up -d eval-runner` to restart it.

5. **Verify end-to-end.**
   - Send a normal RAG request through the proxy (e.g.
     `examples/rag_client_example.py`).
   - Langfuse UI → Traces: the request appears.
   - Within `EVAL_INTERVAL_SECONDS` (default 300) + sampling, the trace gains
     `deepeval_faithfulness` / `deepeval_answer_relevancy` scores with the
     judge's reasoning as the score comment. (For a quick check, temporarily
     set `EVAL_SAMPLE_RATE_RAG=1.0` and `docker compose ... restart eval-runner`,
     or run one cycle by hand: `docker compose ... exec eval-runner python -m
     evals.run_pipeline --once`.)
   - `docker logs eval-runner` shows per-cycle counters:
     `{'fetched': …, 'skipped_internal': …, 'scored': …}`.

## How the pipeline decides what to score

- **Exclusions (never scored):** traces tagged `eval-internal` (the runner's
  own judge calls), guardrail regenerations (`ragas_internal_retry`), and
  any model in `EVAL_SKIP_MODELS` (default `judge-model,eval-judge`).
  Mirror-image guard on the proxy side: `GUARDRAIL_SKIP_MODELS=eval-judge`
  keeps the faithfulness guardrail from scoring eval-judge traffic.
- **Idempotency:** scored trace ids are recorded in SQLite
  (`eval_runner_state` volume); deterministic score ids make retries upsert
  in Langfuse instead of duplicating.
- **Sampling:** deterministic per trace id. RAG traffic (evidence-marker
  contract, same extraction rules and injection-safety invariant as the
  guardrail) sampled at `EVAL_SAMPLE_RATE_RAG` (0.25); everything else at
  `EVAL_SAMPLE_RATE_NON_RAG` (0.05).
- **Spend bounds:** `EVAL_MAX_TRACES_PER_RUN` (25/cycle) caps each cycle;
  the virtual key's `max_budget` caps the month; the metrics run only on
  the sampled remainder.
- **Metrics:** RAG traces get DeepEval `FaithfulnessMetric` +
  `AnswerRelevancyMetric`; non-RAG traces get `AnswerRelevancyMetric` only
  (faithfulness needs retrieval context).

## Operational notes

- **Never on the request path.** Langfuse callback failures are
  non-blocking in LiteLLM, and the eval runner only reads traces after the
  fact. Business traffic is unaffected by Langfuse or the runner being down.
- **Judge quality.** `eval-judge` must stay a ≥20B-class model
  (`groq/openai/gpt-oss-120b` today): DeepEval metrics require
  schema-conforming JSON and small models silently corrupt scores. Before
  trusting dashboards, validate the judge against the 64-row labeled golden
  dataset (plan Phase 2).
- **Data residency/retention.** Full prompts/responses now live in Langfuse
  (ClickHouse + MinIO) in addition to `faithfulness_events` — the retention
  policy gap flagged in the README now covers both stores.
- **Dashboards/alerts.** Langfuse UI charts scores natively (Scores →
  filter by name, group by trace metadata). Wire alerting on
  `deepeval_faithfulness` moving averages before broad rollout.
