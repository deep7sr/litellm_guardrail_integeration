"""DeepEval judge wiring: the judge is whatever open-source model sits
behind the proxy's `eval-judge` model_list entry — swap it in
litellm_config.yaml (Groq today, self-hosted vLLM later) without touching
this code.

Judge calls carry the `eval-internal` tag so (a) the pipeline never samples
its own judge traffic out of Langfuse and (b) the traffic is identifiable
for budget review. DeepEval imports are kept inside the function so unit
tests with injected fakes don't need deepeval installed.
"""

from .config import EvalConfig


def build_judge(cfg: EvalConfig):
    from deepeval.models import LiteLLMModel

    return LiteLLMModel(
        model=f"litellm_proxy/{cfg.judge_model}",
        base_url=cfg.proxy_base_url,
        api_key=cfg.proxy_api_key,
        temperature=0,
        # extra_body merges into the request JSON the proxy actually sees —
        # a bare `metadata` kwarg is consumed client-side by the litellm SDK
        # and never reaches the proxy (verified E2E), which would leave judge
        # traffic untagged and samplable.
        generation_kwargs={
            "extra_body": {
                "metadata": {"tags": [cfg.internal_tag], "eval_internal": True},
            },
        },
    )


def default_metric_factory(cfg: EvalConfig):
    """Returns factory(case) -> list of (score_name, measure_fn) where
    measure_fn() -> (score: float, reason: str|None). Faithfulness only for
    RAG cases (it needs retrieval context); relevancy for everything."""
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    judge = build_judge(cfg)

    def factory(case):
        def _measure(metric, test_case):
            def run():
                metric.measure(test_case)
                return float(metric.score), getattr(metric, "reason", None)
            return run

        entries = []
        if case.is_rag:
            tc = LLMTestCase(
                input=case.question,
                actual_output=case.answer,
                retrieval_context=case.contexts,
            )
            entries.append((
                "deepeval_faithfulness",
                _measure(
                    FaithfulnessMetric(
                        threshold=cfg.faithfulness_threshold,
                        model=judge,
                        include_reason=True,
                        async_mode=False,
                    ),
                    tc,
                ),
            ))
        tc_rel = LLMTestCase(input=case.question, actual_output=case.answer)
        entries.append((
            "deepeval_answer_relevancy",
            _measure(
                AnswerRelevancyMetric(
                    threshold=cfg.relevancy_threshold,
                    model=judge,
                    include_reason=True,
                    async_mode=False,
                ),
                tc_rel,
            ),
        ))
        return entries

    return factory
