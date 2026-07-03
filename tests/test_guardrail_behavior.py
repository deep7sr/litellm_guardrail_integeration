"""Behavioral tests for the guardrail decision loop.

The judge (score_answer) and the generation model (_regenerate) are replaced
with scripted fakes, so every path — pass, retry→pass, exhaustion→fallback,
judge failure, no-claims, self-trigger guard — is exercised deterministically.

These tests specifically pin down the three bugs visible in the old
dashboard screenshots:
  1. sub-threshold scores marked "passed"  -> test_verdict_matches_threshold*
  2. regeneration calls logged as separate scored events -> test_internal_call_*
  3. retry exhaustion not producing the fallback -> test_exhaustion_*

Requires the real dependency stack (ragas/litellm): pip install -r requirements.txt
"""

import asyncio
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ragas_faithfulness_guardrail as g  # noqa: E402
import litellm  # noqa: E402
from litellm.proxy._types import UserAPIKeyAuth  # noqa: E402


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

RAG_DATA = {
    "model": "groq-llama-3.1-8b",
    "litellm_call_id": "test-req-0001",
    "messages": [
        {"role": "system", "content": "Context: The X100 battery lasts 10 hours. It charges via USB-C."},
        {"role": "user", "content": "How long does the battery last?"},
    ],
}


class Harness:
    """Installs scripted judge scores and regeneration answers, and records
    every logged event and every regeneration call."""

    def __init__(self, monkeypatch, scores, regen_answers=()):
        self.scores = list(scores)
        self.regen_answers = list(regen_answers)
        self.events = []          # (verdict, attempt, score)
        self.scored_texts = []    # answers actually sent to the judge
        self.regen_calls = []     # messages passed to regeneration

        async def fake_score(question, contexts, answer):
            self.scored_texts.append(answer)
            return self.scores.pop(0)

        async def fake_log(req_id, attempt, score, verdict, *rest):
            self.events.append((verdict, attempt, score))

        async def fake_regen(self_gr, model, messages, req_id):
            self.regen_calls.append(messages)
            return self.regen_answers.pop(0) if self.regen_answers else None

        monkeypatch.setattr(g, "score_answer", fake_score)
        monkeypatch.setattr(g, "log_event", fake_log)
        monkeypatch.setattr(g.RagasFaithfulnessGuardrail, "_regenerate", fake_regen)
        self.guardrail = g.RagasFaithfulnessGuardrail(guardrail_name="ragas-faithfulness")

    def verdicts(self):
        return [e[0] for e in self.events]


def make_response(text):
    return litellm.ModelResponse(
        choices=[{"index": 0, "message": {"role": "assistant", "content": text}}]
    )


def run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# The decision loop
# ---------------------------------------------------------------------------

def test_verdict_matches_threshold_pass(monkeypatch):
    h = Harness(monkeypatch, scores=[0.9])
    out = run(h.guardrail._evaluate(RAG_DATA, "The battery lasts 10 hours."))
    assert out.action == "allow"
    assert out.final_text == "The battery lasts 10 hours."
    assert h.verdicts() == ["passed"]


def test_verdict_matches_threshold_below_is_never_passed(monkeypatch):
    # Old dashboard bug #1: 0.500 and 0.667 shown as "passed" at threshold 0.7.
    for bad_score in (0.5, 0.667, 0.699):
        h = Harness(monkeypatch, scores=[bad_score, 0.95], regen_answers=["corrected"])
        run(h.guardrail._evaluate(RAG_DATA, "answer"))
        assert h.events[0][0] == "retrying", f"score {bad_score} must not pass"


def test_boundary_score_exactly_at_threshold_passes(monkeypatch):
    h = Harness(monkeypatch, scores=[g.PASS_THRESHOLD])
    out = run(h.guardrail._evaluate(RAG_DATA, "answer"))
    assert out.action == "allow"
    assert h.verdicts() == ["passed"]


def test_retry_then_pass_serves_corrected_answer(monkeypatch):
    h = Harness(monkeypatch, scores=[0.3, 0.95], regen_answers=["Corrected grounded answer."])
    out = run(h.guardrail._evaluate(RAG_DATA, "Hallucinated answer."))
    assert out.action == "allow"
    assert out.final_text == "Corrected grounded answer."
    assert h.verdicts() == ["retrying", "passed"]
    assert h.events[1][1] == 1  # passed on attempt 1
    # The corrective feedback was appended to the regeneration conversation.
    assert h.regen_calls[0][-1]["content"] == g.RETRY_FEEDBACK
    assert h.regen_calls[0][-2] == {"role": "assistant", "content": "Hallucinated answer."}


def test_exhaustion_serves_fallback(monkeypatch):
    # Old dashboard bug #3 territory: all retries fail -> fallback text, verdict "exhausted".
    h = Harness(monkeypatch, scores=[0.3, 0.2, 0.1], regen_answers=["bad1", "bad2"])
    out = run(h.guardrail._evaluate(RAG_DATA, "bad0"))
    assert out.action == "replace"
    assert out.final_text == g.FALLBACK_TEXT
    assert h.verdicts() == ["retrying", "retrying", "exhausted"]
    assert h.scored_texts == ["bad0", "bad1", "bad2"]  # every attempt was re-scored
    assert len(h.regen_calls) == g.MAX_RETRIES


def test_judge_error_fails_closed(monkeypatch):
    h = Harness(monkeypatch, scores=[None])
    out = run(h.guardrail._evaluate(RAG_DATA, "answer"))
    assert out.action == "replace"
    assert out.final_text == g.FALLBACK_TEXT
    assert h.verdicts() == ["scorer_error_blocked"]


def test_judge_error_mid_retry_fails_closed(monkeypatch):
    # The old code served the regenerated-but-unverified answer here.
    h = Harness(monkeypatch, scores=[0.3, None], regen_answers=["unverified regen"])
    out = run(h.guardrail._evaluate(RAG_DATA, "bad"))
    assert out.action == "replace"
    assert out.final_text == g.FALLBACK_TEXT


def test_no_claims_nan_passes_with_own_verdict(monkeypatch):
    # An honest refusal has nothing to hallucinate; must not be logged as score 1.0.
    h = Harness(monkeypatch, scores=[float("nan")])
    out = run(h.guardrail._evaluate(RAG_DATA, "The documents don't mention this."))
    assert out.action == "allow"
    assert h.verdicts() == ["no_claims"]
    assert h.events[0][2] is None  # no fake 1.0 score


def test_regen_failure_serves_fallback(monkeypatch):
    h = Harness(monkeypatch, scores=[0.3], regen_answers=[])  # regen returns None
    out = run(h.guardrail._evaluate(RAG_DATA, "bad"))
    assert out.action == "replace"
    assert out.final_text == g.FALLBACK_TEXT
    assert h.verdicts() == ["retrying", "regen_error"]


def test_empty_conversation_is_skipped_not_crashed(monkeypatch):
    h = Harness(monkeypatch, scores=[])
    out = run(h.guardrail._evaluate({"model": "m", "messages": []}, "answer"))
    assert out.action == "allow"
    assert h.verdicts() == ["skipped_no_context"]
    assert h.scored_texts == []  # judge never called


# ---------------------------------------------------------------------------
# The proxy-facing hook
# ---------------------------------------------------------------------------

def test_hook_passes_good_answer_unchanged(monkeypatch):
    h = Harness(monkeypatch, scores=[0.9])
    resp = make_response("The battery lasts 10 hours.")
    out = run(h.guardrail.async_post_call_success_hook(dict(RAG_DATA), UserAPIKeyAuth(), resp))
    assert out.choices[0].message.content == "The battery lasts 10 hours."


def test_hook_replaces_exhausted_answer_with_fallback(monkeypatch):
    h = Harness(monkeypatch, scores=[0.3, 0.2, 0.1], regen_answers=["bad1", "bad2"])
    resp = make_response("bad0")
    out = run(h.guardrail.async_post_call_success_hook(dict(RAG_DATA), UserAPIKeyAuth(), resp))
    assert out.choices[0].message.content == g.FALLBACK_TEXT


def test_hook_swaps_in_corrected_answer_after_retry(monkeypatch):
    h = Harness(monkeypatch, scores=[0.3, 0.95], regen_answers=["Corrected."])
    resp = make_response("Hallucinated.")
    out = run(h.guardrail.async_post_call_success_hook(dict(RAG_DATA), UserAPIKeyAuth(), resp))
    assert out.choices[0].message.content == "Corrected."


def test_internal_call_is_never_scored_or_logged(monkeypatch):
    # Old dashboard bug #2: regeneration calls appeared as separate scored
    # events ("That answer was not fully grounded..." rows). Internal calls
    # must produce zero events and zero judge calls.
    h = Harness(monkeypatch, scores=[])
    data = dict(RAG_DATA)
    data["metadata"] = {g.INTERNAL_FLAG: True}
    resp = make_response("regenerated text")
    out = run(h.guardrail.async_post_call_success_hook(data, UserAPIKeyAuth(), resp))
    assert out.choices[0].message.content == "regenerated text"
    assert h.events == []
    assert h.scored_texts == []


def test_empty_or_toolcall_answer_is_skipped(monkeypatch):
    h = Harness(monkeypatch, scores=[])
    resp = make_response("")
    out = run(h.guardrail.async_post_call_success_hook(dict(RAG_DATA), UserAPIKeyAuth(), resp))
    assert out is resp
    assert h.events == []


# ---------------------------------------------------------------------------
# Streaming hook (uses lightweight fake chunks; the hook only touches
# .choices[0].delta.content and .finish_reason)
# ---------------------------------------------------------------------------

class FakeDelta:
    def __init__(self, content):
        self.content = content

class FakeChoice:
    def __init__(self, content):
        self.delta = FakeDelta(content)
        self.finish_reason = None

class FakeChunk:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


async def chunk_stream(texts):
    for t in texts:
        yield FakeChunk(t)


def test_streaming_pass_releases_original_chunks(monkeypatch):
    h = Harness(monkeypatch, scores=[0.9])

    async def collect():
        out = []
        async for c in h.guardrail.async_post_call_streaming_iterator_hook(
            UserAPIKeyAuth(), chunk_stream(["The battery ", "lasts 10 hours."]), dict(RAG_DATA)
        ):
            out.append(c)
        return out

    chunks = run(collect())
    text = "".join(c.choices[0].delta.content for c in chunks)
    assert text == "The battery lasts 10 hours."
    assert h.scored_texts == ["The battery lasts 10 hours."]  # scored as one answer


def test_streaming_failure_replaces_with_fallback(monkeypatch):
    h = Harness(monkeypatch, scores=[0.3, 0.2, 0.1], regen_answers=["bad1", "bad2"])

    async def collect():
        out = []
        async for c in h.guardrail.async_post_call_streaming_iterator_hook(
            UserAPIKeyAuth(), chunk_stream(["halluci", "nated"]), dict(RAG_DATA)
        ):
            out.append(c)
        return out

    chunks = run(collect())
    text = "".join(c.choices[0].delta.content or "" for c in chunks)
    assert text == g.FALLBACK_TEXT
    assert chunks[-1].choices[0].finish_reason == "stop"
