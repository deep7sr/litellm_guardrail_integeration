# DeepEval + LiteLLM Proxy — Evaluation Architecture Plan

Status: **PLAN — for team review before implementation.**
Owner branch: `liteLLM_evals`.

## 1. Goal and constraints

Set up LLM evaluation for the org's central LiteLLM proxy (the gateway all
business AI applications call through), using **DeepEval open source** as the
metric/eval framework.

Hard constraints:

- **Zero paid services.** No Confident AI, no hosted eval platforms.
  Everything self-hosted or free-tier.
- **Open-source judge model.** All LLM-as-judge scoring runs on an
  open-source model (today: Groq-hosted `gpt-oss`/Llama via our proxy;
  later: self-hosted vLLM — swappable in `litellm_config.yaml` without
  touching eval code).
- **Production-grade.** The proxy is production infrastructure; evals must
  never add latency to business traffic, never eval-loop on their own
  traffic, and must be budget-capped.

## 2. The critical clarification: two *different* "deepeval + litellm" integrations

The docs are confusing because there are two unrelated integrations. Only
one of them is usable for us:

| Integration | Direction | What it does | Usable? |
|---|---|---|---|
| LiteLLM's "DeepEval" **logging callback** (`success_callback: ["deepeval"]`, needs `CONFIDENT_API_KEY`) | proxy → deepeval | Ships every request trace to **Confident AI's cloud platform** for tracing/monitoring | ❌ **No** — it is the paid hosted platform (plus data residency: full prompts/responses leave our infra) |
| DeepEval's **`LiteLLMModel`** judge (`LITELLM_PROXY_API_BASE` / `LITELLM_PROXY_API_KEY`) | deepeval → proxy | Lets deepeval metrics use **any model behind our own proxy as the judge** | ✅ **Yes** — this is our path |

So the answer to "how do I integrate deepeval with the proxy" is:
**deepeval is not a proxy plugin.** It is an eval harness that runs *beside*
the proxy, and the proxy plays two roles for it:

1. **Judge gateway** — deepeval's metrics call the judge model *through the
   proxy* (`LiteLLMModel(model="litellm_proxy/eval-judge", base_url=...)`),
   so judge choice, keys, budgets, rate limits and later self-hosting are all
   managed centrally in `litellm_config.yaml`, exactly like the guardrail's
   `judge-model` today.
2. **Data source** — the proxy already sees every production request/response
   (and our guardrail already logs RAG traffic to Postgres
   `faithfulness_events`). Evals consume *logged traces*, asynchronously —
   never inline on the request path.

## 3. Reference architecture

```
                              ┌──────────────────────────────────────────┐
 business apps ──────────────▶│  LiteLLM PROXY (production gateway)      │──▶ frontier models
 (RAG, chat, agents)          │  • guardrails (faithfulness, post_call)  │
                              │  • model_list incl. `eval-judge`         │
                              │  • logging callback → trace store        │
                              └───────┬──────────────────────▲───────────┘
                                      │ traces (async)       │ judge calls
                                      ▼                      │ (OSS model)
                          ┌──────────────────────┐   ┌───────┴───────────────┐
                          │  TRACE STORE          │   │  EVAL RUNNER          │
                          │  Postgres (existing)  │──▶│  deepeval (OSS, local │
                          │  [opt: Langfuse OSS]  │   │  results — no cloud)  │
                          └──────────────────────┘   └───────┬───────────────┘
                                                             │ scores
                                                             ▼
                                              ┌──────────────────────────────┐
                                              │ eval_results (Postgres)      │
                                              │ → existing dashboard + alerts│
                                              └──────────────────────────────┘

  CI loop (pre-production): golden datasets ──▶ deepeval + pytest ──▶ gate merges
```

Three eval loops, built in this order:

### Loop A — Offline / CI evals on golden datasets (build first)

- Reuse and extend `datasets/golden_faithfulness_dataset.json` into deepeval
  `EvaluationDataset` goldens (per use-case datasets over time: RAG QA,
  summarization, etc.).
- Metrics: start with deepeval `FaithfulnessMetric`, `AnswerRelevancyMetric`,
  `ContextualRelevancyMetric` (RAG triad); add `GEval` for use-case-specific
  rubrics.
- Runner: `deepeval test run` (pytest-based, `assert_test` per golden) —
  results are printed/stored **locally** (`.deepeval/` cache + JSON export);
  no Confident AI login is ever run.
- Generation under test goes **through the proxy** (apps' real path), judge
  goes through the proxy's `eval-judge` model.
- Wire into CI so any change to prompts, guardrail code, `litellm_config.yaml`
  (model swaps!) or judge model must pass threshold gates before deploy.
  This also finally closes the README's open item: **threshold calibration**
  for the guardrail becomes a deepeval run over the labeled dataset.

### Loop B — Online production monitoring (sampled, async)

- **Never inline.** A scheduled sidecar job (cron/container, same host as the
  dashboard) pulls a **sample** of recent traces and scores them with
  deepeval metrics, writing scores to a new Postgres table `eval_results`
  keyed by request id.
- Trace capture options:
  - **Phase 1 (zero new infra):** extend the pattern we already have — a
    LiteLLM custom logging callback writing request/response/metadata to
    Postgres (the guardrail's `faithfulness_events` already covers RAG
    traffic; add a generic `llm_traces` callback for the rest, with sampling
    at write time).
  - **Phase 2 (recommended industry pattern):** self-host **Langfuse OSS**
    (free, Docker, Postgres) and point the proxy's built-in
    `success_callback: ["langfuse"]` at it. Langfuse becomes the trace UI +
    queue; the deepeval runner fetches unscored traces via Langfuse's API
    and pushes scores back as Langfuse `scores` — this "external evaluation
    pipeline" pattern is Langfuse's documented cookbook and is what
    LiteLLM+Langfuse shops (Lemonade, Fletch.ai, and others) run in
    production. Our existing dashboard keeps working; Langfuse adds
    per-trace drill-down.
- Sampling policy (initial): 100% of guardrail-**failed** traffic, ~5% of
  passed/unscored traffic, per-app caps. Tune by judge budget.
- Aggregate metrics land on the dashboard; alert on drops (e.g., 7-day
  faithfulness moving average, per-app).

### Loop C — Regression harness for the platform itself

- Before any proxy change (LiteLLM version bump, model swap, guardrail
  threshold change): replay the golden datasets through a staging proxy and
  diff deepeval scores vs. the last accepted baseline. This is Loop A run
  against staging with a stored baseline, not a new system.

## 4. Judge configuration (the deepeval ↔ proxy contract)

`litellm_config.yaml` gains one entry (separate from the guardrail's
`judge-model` so eval traffic is independently budgeted/observable):

```yaml
model_list:
  - model_name: eval-judge
    litellm_params:
      model: groq/openai/gpt-oss-120b     # OSS judge; swap to self-hosted vLLM later
      api_key: os.environ/GROQ_API_KEY
```

Eval runner side (env or code):

```python
from deepeval.models import LiteLLMModel

judge = LiteLLMModel(
    model="litellm_proxy/eval-judge",
    base_url="http://<proxy-host>:4000",   # or LITELLM_PROXY_API_BASE
    api_key=os.environ["EVAL_VIRTUAL_KEY"],  # dedicated LiteLLM virtual key
)
metric = FaithfulnessMetric(model=judge, threshold=0.7)
```

Platform guards (all enforced in the proxy, not in eval code):

- **Dedicated virtual key** for the eval runner with a **max budget** and
  rate limit (LiteLLM key/team budgets) — a runaway eval job cannot burn
  money or starve production.
- **No recursion:** eval-judge traffic must be excluded from guardrail
  scoring and from eval sampling. The guardrail already skips
  `model == JUDGE_MODEL`; add `eval-judge` to the same skip set, and tag
  eval-runner requests (`metadata.eval_internal`) so the trace callback
  drops them.
- **Judge quality gotcha:** deepeval metrics require the judge to emit
  schema-conforming JSON. Small OSS models (≤8B) fail this often, which
  shows up as metric errors or silently wrong scores. Use ≥20B-class
  (gpt-oss-120b / Llama-3.3-70B on Groq free tier now; vLLM-hosted
  equivalent later) and validate the judge itself once against the labeled
  golden dataset (judge-vs-human-label agreement) before trusting Loop B
  numbers.

## 5. What other orgs do (research summary)

- The dominant production pattern for "LiteLLM proxy as central gateway +
  evals" is **proxy → self-hosted trace store (Langfuse OSS) → async
  external eval pipeline → scores pushed back**: Langfuse documents it as a
  first-class cookbook, and LiteLLM+Langfuse is a publicly cited stack at
  Lemonade (self-hosted for compliance) and Fletch.ai; Netflix and
  Rocket Money run LiteLLM as their gateway.
- Nobody credible runs LLM-as-judge evals **inline** on the gateway for all
  traffic — inline judging is reserved for guardrails (which we already
  have); evals are sampled and asynchronous.
- Teams avoiding paid platforms run deepeval purely as a library + pytest in
  CI with results kept local, exactly Loop A.

## 6. Delivery plan

| Phase | Deliverable | Depends on |
|---|---|---|
| 1 | `evals/` package: deepeval dataset loader for our golden JSON, `LiteLLMModel` judge wiring, `deepeval test run` config; run against staging proxy | proxy reachable, `eval-judge` in config, virtual key |
| 2 | Judge validation report: deepeval FaithfulnessMetric scores vs. the 64 human labels; pick judge + calibrate guardrail threshold (closes README open item) | Phase 1 |
| 3 | CI gate: golden evals on PRs touching prompts/guardrail/config | Phase 1 |
| 4 | Online sampler v1: cron job scoring sampled `faithfulness_events` + new `llm_traces` callback; `eval_results` table; dashboard panels + alert thresholds | Phase 2 |
| 5 | (Optional, recommended) Langfuse OSS deployment; switch trace capture to `success_callback: ["langfuse"]`; eval runner reads/writes Langfuse scores | Phase 4 experience |

## 7. Open decisions for the team

1. **Langfuse now or later?** Phase-1-on-Postgres works with zero new infra;
   Langfuse is the better long-term trace store but is one more service to
   operate.
2. **Judge hosting:** stay on Groq free tier (rate limits, data leaves VPC)
   vs. accelerate self-hosted vLLM. Eval volume makes this decision sooner
   than the guardrail did.
3. **Sampling rates & retention** for `llm_traces` (same GDPR/retention gap
   the README flags for `faithfulness_events` — fix both together).
4. **Which non-RAG use cases** get golden datasets first (needs app-team
   input; RAG QA is covered by the existing 64 rows).
