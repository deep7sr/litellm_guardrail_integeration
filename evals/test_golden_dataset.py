"""Offline / CI evaluation: golden dataset vs. the live LiteLLM proxy.

Every case in golden_dataset.json is answered by the target model THROUGH the
proxy (so guardrails, routing, and prompts are exercised exactly as in
production) and scored with DeepEval. A score below threshold fails the test,
which fails the pipeline — quality regressions block the merge.

Environment:
  EVAL_PROXY_URL      base URL of the LiteLLM proxy   (default http://127.0.0.1:4010)
  EVAL_PROXY_API_KEY  a LiteLLM virtual key            (default sk-1234)
  EVAL_TARGET_MODEL   model_name to evaluate           (default rag-model)
  JUDGE_MODEL         judge model_name on the proxy    (default judge-model)
  EVAL_FAITHFULNESS_THRESHOLD (default 0.7 — matches the inline guardrail)

Run:  deepeval test run evals/test_golden_dataset.py    (or plain pytest)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("ERROR_REPORTING", "NO")

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval_worker"))
from judge import ProxyJudge  # noqa: E402

PROXY_URL = os.environ.get("EVAL_PROXY_URL", "http://127.0.0.1:4010")
PROXY_KEY = os.environ.get("EVAL_PROXY_API_KEY", "sk-1234")
TARGET_MODEL = os.environ.get("EVAL_TARGET_MODEL", "rag-model")
FAITHFULNESS_THRESHOLD = float(os.environ.get("EVAL_FAITHFULNESS_THRESHOLD", "0.7"))
RELEVANCY_THRESHOLD = float(os.environ.get("EVAL_RELEVANCY_THRESHOLD", "0.5"))

DATASET = json.loads(
    (Path(__file__).resolve().parent / "golden_dataset.json").read_text()
)

_client = OpenAI(base_url=f"{PROXY_URL}/v1", api_key=PROXY_KEY, timeout=120)
_judge = ProxyJudge(base_url=f"{PROXY_URL}/v1", api_key=PROXY_KEY)


def answer_through_proxy(question: str, retrieval_context: list[str]) -> str:
    resp = _client.chat.completions.create(
        model=TARGET_MODEL,
        messages=[
            {"role": "system", "content": "\n\n".join(retrieval_context)},
            {"role": "user", "content": question},
        ],
        temperature=0,
        extra_body={"metadata": {"context": retrieval_context, "source": "golden-dataset-ci"}},
    )
    return resp.choices[0].message.content or ""


@pytest.mark.parametrize("case", DATASET, ids=[c["id"] for c in DATASET])
def test_golden_case(case: dict):
    actual_output = answer_through_proxy(case["question"], case["retrieval_context"])
    test_case = LLMTestCase(
        input=case["question"],
        actual_output=actual_output,
        retrieval_context=case["retrieval_context"],
    )
    assert_test(
        test_case,
        [
            FaithfulnessMetric(
                threshold=FAITHFULNESS_THRESHOLD, model=_judge, async_mode=False
            ),
            AnswerRelevancyMetric(
                threshold=RELEVANCY_THRESHOLD, model=_judge, async_mode=False
            ),
        ],
        run_async=False,
    )
