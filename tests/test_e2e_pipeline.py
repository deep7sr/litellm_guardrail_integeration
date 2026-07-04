"""End-to-end pipeline test.

Requires the local stack to be running (tests/run_stack.sh):
  * LiteLLM proxy on :4010 with `success_callback: ["langfuse"]`
  * deterministic mock LLM on :9001
  * Langfuse API stub on :9002

Verifies, over the real LiteLLM proxy and real DeepEval:
  1. Proxy traffic is captured as Langfuse traces (callback fires).
  2. The worker scores a grounded answer 1.0 and a half-fabricated answer 0.5
     (deterministic via the mock judge).
  3. metadata.context contract takes priority; system-role fallback works.
  4. Judge-model traffic and traces without context are never scored.
  5. Re-running the worker never double-scores (idempotency).
  6. EVAL_SAMPLE_RATE=0.0 scores nothing (sampling).

Run:  pytest -x -q tests/test_e2e_pipeline.py
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import httpx
import pytest

PROXY = os.environ.get("TEST_PROXY_URL", "http://127.0.0.1:4010")
STUB = os.environ.get("TEST_LANGFUSE_URL", "http://127.0.0.1:9002")
MASTER_KEY = "sk-1234"
WORKER_DIR = Path(__file__).resolve().parent.parent / "eval_worker"
WORKER_PYTHON = os.environ.get("WORKER_PYTHON", "/home/user/venvs/worker/bin/python")

KB = (
    "Product KB: The AeroBook X200 battery lasts 12 hours under normal use. "
    "It fully charges in 90 minutes with the included charger."
)

client = httpx.Client(timeout=60)


def chat(messages, model="rag-model", metadata=None):
    body = {"model": model, "messages": messages}
    if metadata is not None:
        body["metadata"] = metadata
    r = client.post(
        f"{PROXY}/v1/chat/completions",
        headers={"Authorization": f"Bearer {MASTER_KEY}"},
        json=body,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def stub_state():
    r = client.get(f"{STUB}/debug/events")
    r.raise_for_status()
    return r.json()


def scores_by_trace():
    state = stub_state()
    out: dict[str, dict[str, float]] = {}
    for s in state["scores"]:
        out.setdefault(s["traceId"], {})[s["name"]] = s["value"]
    return out


def find_trace_id(question_substring: str) -> str:
    state = stub_state()
    for e in state["events"]:
        if e["type"] != "trace-create":
            continue
        msgs = (e["body"].get("input") or {}).get("messages") or []
        if any(question_substring in str(m.get("content", "")) for m in msgs):
            return e["body"]["id"]
    raise AssertionError(f"no trace found containing {question_substring!r}")


def run_worker(sample_rate: str = "1.0") -> str:
    env = dict(os.environ)
    env.update(
        LANGFUSE_HOST=STUB,
        LANGFUSE_PUBLIC_KEY="pk-lf-test",
        LANGFUSE_SECRET_KEY="sk-lf-test",
        LITELLM_BASE_URL=f"{PROXY}/v1",
        LITELLM_API_KEY=MASTER_KEY,
        JUDGE_MODEL="judge-model",
        EVAL_RUN_ONCE="true",
        EVAL_SAMPLE_RATE=sample_rate,
        DEEPEVAL_TELEMETRY_OPT_OUT="YES",
    )
    proc = subprocess.run(
        [WORKER_PYTHON, "worker.py"],
        cwd=WORKER_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"worker failed:\n{proc.stdout}\n{proc.stderr}"
    return proc.stderr + proc.stdout


@pytest.fixture(scope="module", autouse=True)
def fresh_stub_and_traffic():
    client.post(f"{STUB}/debug/reset")

    # 1. Grounded answer; context via explicit metadata contract.
    answer = chat(
        [
            {"role": "system", "content": KB},
            {"role": "user", "content": "How long does the battery last?"},
        ],
        metadata={"context": [KB], "team": "team-a"},
    )
    assert "12 hours" in answer

    # 2. Half-fabricated answer; context via system-role fallback only.
    answer = chat(
        [
            {"role": "system", "content": KB},
            {"role": "user", "content": "What color options does the AeroBook X200 come in?"},
        ]
    )
    assert "XyloBrite" in answer  # the mock's deterministic fabrication

    # 3. Direct judge-model call: internal traffic, must never be scored.
    chat([{"role": "user", "content": "internal judge probe"}], model="judge-model")

    # 4. No context anywhere: not scorable, must be skipped.
    chat([{"role": "user", "content": "Hello, what can you do?"}])

    # Wait for the langfuse callback to flush all 4 traces to the stub.
    deadline = time.time() + 30
    while time.time() < deadline:
        if stub_state()["n_traces"] >= 4:
            break
        time.sleep(1)
    assert stub_state()["n_traces"] >= 4, "langfuse callback did not deliver traces"

    run_worker()
    yield


def test_grounded_trace_scores_one():
    trace_id = find_trace_id("How long does the battery last?")
    scores = scores_by_trace().get(trace_id, {})
    assert scores.get("deepeval_faithfulness") == 1.0
    assert scores.get("deepeval_answer_relevancy") == 1.0


def test_fabricated_trace_scores_half():
    trace_id = find_trace_id("What color options")
    scores = scores_by_trace().get(trace_id, {})
    # 2 claims, 1 fabricated (XyloBrite) -> exactly 0.5
    assert scores.get("deepeval_faithfulness") == 0.5


def test_judge_traffic_never_scored():
    trace_id = find_trace_id("internal judge probe")
    assert trace_id not in scores_by_trace()


def test_no_context_trace_skipped():
    trace_id = find_trace_id("Hello, what can you do?")
    assert trace_id not in scores_by_trace()


def test_rerun_is_idempotent():
    before = sum(len(v) for v in scores_by_trace().values())
    run_worker()
    after = sum(len(v) for v in scores_by_trace().values())
    assert before == after, "second worker run must not add duplicate scores"


def test_sample_rate_zero_scores_nothing():
    chat(
        [
            {"role": "system", "content": KB},
            {"role": "user", "content": "Does the charger come included?"},
        ]
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            trace_id = find_trace_id("Does the charger come included?")
            break
        except AssertionError:
            time.sleep(1)
    else:
        raise AssertionError("trace for sampling test never arrived")
    run_worker(sample_rate="0.0")
    assert trace_id not in scores_by_trace()
