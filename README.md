# litellm_guardrail_integeration

Central LiteLLM proxy with:

1. **Inline hallucination guardrail** — RAGAS Faithfulness scoring of every
   RAG response before it reaches the user
   (`faithfulness_guardrail_ragas.py`, see `ragas-guardrail-project-brief.md`).
2. **Continuous eval harness** — all proxy traffic captured to self-hosted
   Langfuse and scored asynchronously with DeepEval; golden-dataset
   regression evals gate every PR in CI.

**Start here: [docs/EVALS_SETUP.md](docs/EVALS_SETUP.md)** — architecture,
quick start (`docker compose up`), integration contract for consuming apps,
and the production hand-off checklist.
