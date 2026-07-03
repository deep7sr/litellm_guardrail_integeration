# LiteLLM RAGAS Faithfulness Guardrail

A production-oriented hallucination guardrail for a central LiteLLM proxy:
every response is decomposed into factual claims and verified against its
grounding source (RAGAS Faithfulness) by a self-hosted judge model before
it reaches the end user. Ungrounded answers are regenerated with corrective
feedback; persistent failures are replaced with a safe fallback. The
pipeline is **fail-closed**: judge errors, timeouts, and retry exhaustion
all block rather than pass.

The proxy is **invisible middleware**: applications send completely normal
OpenAI-format requests, and every answer is verified against the full
conversation the model was shown (the model's own earlier replies excluded).
No app or end-user changes are needed. See `docs/INTEGRATION.md` for the
precise guarantee and its limits.

- **`docs/REVIEW.md`** — full architecture review: what was wrong with the
  previous iteration, every decision made here, judge-model recommendation,
  threshold calibration procedure, and honest accuracy limits.
- **`docs/INTEGRATION.md`** — the contract consuming RAG apps must follow
  (one `extra_body` field).
- **`ragas-guardrail-project-brief.md`** — the original brief this rebuild
  responds to.

## Components

| File | Role |
|---|---|
| `context_contract.py` | Context/question extraction contract (stdlib-only, unit-tested) |
| `ragas_faithfulness_guardrail.py` | The guardrail: scoring, retries, streaming support, fail-closed policies, event logging |
| `litellm_config.yaml` | Proxy config registering the guardrail (`default_on: true`) |
| `Dockerfile` / `requirements.txt` | Proxy image with RAGAS deps baked in |
| `docker-compose.yml` | Proxy + Postgres + self-hosted vLLM judge + dashboard |
| `dashboard/` | Live pass/fail event viewer (demo tool) on :8080 |
| `tests/` | Unit tests for the extraction contract |

## Quick start

```bash
cp .env.example .env        # set LITELLM_MASTER_KEY, POSTGRES_PASSWORD, GROQ_API_KEY
docker compose up --build -d
```

The judge defaults to `Qwen/Qwen2.5-7B-Instruct` on the local vLLM service
(needs an NVIDIA GPU, ~24 GB VRAM; first start downloads the weights). For
production accuracy use `JUDGE_MODEL=Qwen/Qwen2.5-32B-Instruct` with
appropriate GPUs. No GPU available for dev? Comment out `vllm-judge` in the
compose file and point `JUDGE_BASE_URL`/`JUDGE_MODEL`/`JUDGE_API_KEY` at any
OpenAI-compatible endpoint (see `.env.example` for the Groq settings).

Any chat completion through the proxy is guarded automatically — no special
request format needed:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H "Content-Type: application/json" \
  -d '{
    "model": "groq-llama-3.1-8b",
    "messages": [
      {"role": "system", "content": "Answer using the context.\n\nThe X100 battery lasts 10 hours."},
      {"role": "user", "content": "How long does the battery last?"}
    ]
  }'
```

The answer is scored against the conversation; if the model adds anything
it wasn't told, the guardrail retries with corrective feedback and, failing
that, returns the fallback message instead.

## Tests

```bash
pip install pytest && pytest tests/ -v
```

## Key configuration (env)

| Variable | Default | Meaning |
|---|---|---|
| `RAGAS_PASS_THRESHOLD` | `0.7` | Min fraction of supported claims (calibrate per `docs/REVIEW.md` §7) |
| `RAGAS_MAX_RETRIES` | `2` | Regeneration attempts before fallback |
| `RAGAS_ON_SCORER_ERROR` | `block` | `block` (fallback text) or `allow` |
| `RAGAS_SCORING_TIMEOUT_S` | `60` | Judge scoring timeout per attempt |
| `JUDGE_BASE_URL` / `JUDGE_MODEL` / `JUDGE_API_KEY` | local vLLM | Any OpenAI-compatible judge endpoint |
