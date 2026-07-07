"""Central configuration for the eval runner. Everything is env-driven —
same policy as the guardrail: no per-request overrides, one place to audit."""

import os
from dataclasses import dataclass, field


def _csv_set(raw: str) -> frozenset[str]:
    return frozenset(x.strip() for x in raw.split(",") if x.strip())


@dataclass(frozen=True)
class EvalConfig:
    # Langfuse (trace source + score sink)
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # LiteLLM proxy (judge gateway). judge_model is the model_list name;
    # the runner calls it as litellm_proxy/<judge_model>.
    proxy_base_url: str = "http://localhost:4000"
    proxy_api_key: str = ""
    judge_model: str = "eval-judge"

    # Pipeline behavior
    lookback_minutes: int = 60
    max_traces_per_run: int = 25  # hard cap on judge spend per cycle
    sample_rate_rag: float = 0.25
    sample_rate_non_rag: float = 0.05
    interval_seconds: int = 300

    # Metric thresholds (Langfuse stores raw scores; thresholds only drive
    # the pass/fail comment on the score)
    faithfulness_threshold: float = 0.7
    relevancy_threshold: float = 0.5

    # Exclusions — the recursion guards of the eval world. Traffic from
    # these models, or tagged as internal, is never sampled.
    skip_models: frozenset[str] = field(
        default_factory=lambda: frozenset({"judge-model", "eval-judge"})
    )
    internal_tag: str = "eval-internal"

    # Same org-wide marker contract as the guardrail
    evidence_marker: str = "--- Retrieved Evidence ---"

    state_path: str = "/data/eval_state.sqlite"

    @classmethod
    def from_env(cls) -> "EvalConfig":
        e = os.environ.get
        return cls(
            langfuse_host=e("LANGFUSE_HOST", cls.langfuse_host),
            langfuse_public_key=e("LANGFUSE_PUBLIC_KEY", ""),
            langfuse_secret_key=e("LANGFUSE_SECRET_KEY", ""),
            proxy_base_url=e("EVAL_PROXY_BASE_URL", cls.proxy_base_url),
            proxy_api_key=e("EVAL_PROXY_API_KEY", ""),
            judge_model=e("EVAL_JUDGE_MODEL", cls.judge_model),
            lookback_minutes=int(e("EVAL_LOOKBACK_MINUTES", "60")),
            max_traces_per_run=int(e("EVAL_MAX_TRACES_PER_RUN", "25")),
            sample_rate_rag=float(e("EVAL_SAMPLE_RATE_RAG", "0.25")),
            sample_rate_non_rag=float(e("EVAL_SAMPLE_RATE_NON_RAG", "0.05")),
            interval_seconds=int(e("EVAL_INTERVAL_SECONDS", "300")),
            faithfulness_threshold=float(e("EVAL_FAITHFULNESS_THRESHOLD", "0.7")),
            relevancy_threshold=float(e("EVAL_RELEVANCY_THRESHOLD", "0.5")),
            skip_models=_csv_set(e("EVAL_SKIP_MODELS", "judge-model,eval-judge")),
            internal_tag=e("EVAL_INTERNAL_TAG", cls.internal_tag),
            evidence_marker=e("EVIDENCE_MARKER", cls.evidence_marker),
            state_path=e("EVAL_STATE_PATH", cls.state_path),
        )
