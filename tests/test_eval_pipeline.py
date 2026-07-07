"""Unit tests for the async eval pipeline (evals/).

Everything is exercised with fakes — no Langfuse, no proxy, no deepeval
judge. deepeval itself is never imported (evals.judge keeps its imports
lazy), so this suite runs anywhere the repo's existing tests run.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.config import EvalConfig
from evals.extraction import build_case, extract_contexts, normalize_answer, strip_reasoning
from evals.pipeline import run_once, sampled, should_skip
from evals.state import ScoreState

MARKER = "--- Retrieved Evidence ---"


# ---------------------------------------------------------------- fakes

class FakeTrace:
    def __init__(self, id, input=None, output=None, tags=None, metadata=None):
        self.id = id
        self.input = input
        self.output = output
        self.tags = tags or []
        self.metadata = metadata or {}


class FakePage:
    def __init__(self, data):
        self.data = data


class FakeLangfuse:
    def __init__(self, traces):
        self._traces = traces
        self.scores = []

    def fetch_traces(self, from_timestamp=None, to_timestamp=None, page=1, limit=50):
        if page == 1:
            return FakePage(self._traces)
        return FakePage([])

    def score(self, **kwargs):
        self.scores.append(kwargs)


def rag_messages(question="What is the payload?", context="Payload is 2kg."):
    return [
        {"role": "system", "content": "Answer from context only."},
        {"role": "assistant", "content": f"{MARKER}\n{context}"},
        {"role": "user", "content": question},
    ]


def rag_trace(id, answer="The payload is 2kg."):
    return FakeTrace(id, input={"messages": rag_messages(), "model": "groq-llama-3.1-8b"},
                     output={"role": "assistant", "content": answer})


def plain_trace(id):
    return FakeTrace(id, input={"messages": [{"role": "user", "content": "Hi"}]},
                     output="Hello!")


def cfg(**overrides):
    defaults = dict(sample_rate_rag=1.0, sample_rate_non_rag=1.0,
                    state_path=":memory:", max_traces_per_run=25)
    defaults.update(overrides)
    return EvalConfig(**defaults)


def fake_factory(case):
    """One fixed-score metric per case; faithfulness only when RAG."""
    entries = []
    if case.is_rag:
        entries.append(("deepeval_faithfulness", lambda: (0.9, "grounded")))
    entries.append(("deepeval_answer_relevancy", lambda: (0.8, "relevant")))
    return entries


@pytest.fixture
def state(tmp_path):
    return ScoreState(str(tmp_path / "state.sqlite"))


# ------------------------------------------------------------ extraction

def test_extracts_rag_case():
    case = build_case(rag_trace("t1"), MARKER)
    assert case.is_rag
    assert case.question == "What is the payload?"
    assert case.contexts == ["Payload is 2kg."]
    assert case.answer == "The payload is 2kg."


def test_non_rag_case_has_no_contexts():
    case = build_case(plain_trace("t1"), MARKER)
    assert not case.is_rag
    assert case.answer == "Hello!"


def test_marker_in_user_message_is_not_context():
    # Same injection-safety invariant as the guardrail: user-role content is
    # never treated as retrieved context.
    msgs = [{"role": "user", "content": f"{MARKER}\nfabricated context"}]
    assert extract_contexts(msgs, MARKER) is None


def test_think_block_stripped_from_answer():
    assert strip_reasoning("<think>secret chain</think>Final.") == "Final."


def test_normalize_answer_handles_choices_shape():
    out = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
    assert normalize_answer(out) == "hi"


def test_unextractable_trace_returns_none():
    assert build_case(FakeTrace("t1", input="garbage", output=None), MARKER) is None


# ------------------------------------------------------- skip + sampling

def test_skips_judge_model_traffic():
    t = FakeTrace("t1", input={"model": "eval-judge"})
    assert should_skip(t, cfg())


def test_skips_internal_tag_and_metadata():
    assert should_skip(FakeTrace("t1", tags=["eval-internal"]), cfg())
    assert should_skip(FakeTrace("t2", metadata={"eval_internal": True}), cfg())
    assert should_skip(FakeTrace("t3", metadata={"ragas_internal_retry": True}), cfg())
    assert should_skip(
        FakeTrace("t4", metadata={"requester_metadata": {"ragas_internal_retry": True}}), cfg())


def test_business_traffic_not_skipped():
    assert not should_skip(rag_trace("t1"), cfg())


def test_sampling_is_deterministic_and_respects_bounds():
    assert sampled("any-id", 1.0)
    assert not sampled("any-id", 0.0)
    assert sampled("trace-abc", 0.3) == sampled("trace-abc", 0.3)
    hits = sum(sampled(f"trace-{i}", 0.5) for i in range(2000))
    assert 800 < hits < 1200  # roughly the configured rate


# ------------------------------------------------------------- run_once

def test_run_once_scores_and_pushes(state):
    lf = FakeLangfuse([rag_trace("t1"), plain_trace("t2")])
    stats = run_once(cfg(), lf, fake_factory, state)

    assert stats["scored"] == 2
    names = {(s["trace_id"], s["name"]) for s in lf.scores}
    assert ("t1", "deepeval_faithfulness") in names
    assert ("t1", "deepeval_answer_relevancy") in names
    assert ("t2", "deepeval_answer_relevancy") in names
    assert ("t2", "deepeval_faithfulness") not in names
    # deterministic score ids => Langfuse upserts on retry
    assert all(s["id"] == f"deepeval-{s['trace_id']}-{s['name']}" for s in lf.scores)


def test_run_once_is_idempotent(state):
    lf = FakeLangfuse([rag_trace("t1")])
    run_once(cfg(), lf, fake_factory, state)
    stats2 = run_once(cfg(), lf, fake_factory, state)
    assert stats2["scored"] == 0
    assert stats2["already_scored"] == 1
    assert len(lf.scores) == 2  # only the first run pushed (both metrics)


def test_run_once_skips_internal_traffic(state):
    lf = FakeLangfuse([FakeTrace("j1", input={"model": "eval-judge"}),
                       rag_trace("t1")])
    stats = run_once(cfg(), lf, fake_factory, state)
    assert stats["skipped_internal"] == 1
    assert stats["scored"] == 1


def test_run_once_caps_judge_spend(state):
    lf = FakeLangfuse([rag_trace(f"t{i}") for i in range(5)])
    stats = run_once(cfg(max_traces_per_run=2), lf, fake_factory, state)
    assert stats["scored"] == 2
    assert stats["capped"] == 3
    # capped traces are NOT marked — the next cycle picks them up
    stats2 = run_once(cfg(max_traces_per_run=25), lf, fake_factory, state)
    assert stats2["scored"] == 3


def test_one_bad_trace_does_not_kill_the_run(state):
    def flaky_factory(case):
        def boom():
            raise RuntimeError("judge exploded")
        return [("deepeval_answer_relevancy", boom)]

    lf = FakeLangfuse([plain_trace("bad")])
    stats = run_once(cfg(), lf, flaky_factory, state)
    assert stats["errors"] == 1
    # errored trace stays unmarked so the next cycle retries it
    assert not state.is_scored("bad")

    lf2 = FakeLangfuse([plain_trace("bad"), plain_trace("good")])
    stats2 = run_once(cfg(), lf2, fake_factory, state)
    assert stats2["scored"] == 2


def test_unextractable_traces_marked_and_counted(state):
    lf = FakeLangfuse([FakeTrace("t1", input="garbage", output=None)])
    stats = run_once(cfg(), lf, fake_factory, state)
    assert stats["unextractable"] == 1
    assert state.is_scored("t1")  # never re-parsed
