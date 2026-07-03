"""Rendering tests for the diagnostic dashboard (pure HTML helpers, no DB)."""

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dashboard"))

from main import claims_html, request_html, score_bar  # noqa: E402

NOW = datetime.datetime(2026, 7, 3, 12, 0, 0)


def event(**over):
    base = {
        "id": 1, "req_id": "abc123", "created_at": NOW, "attempt": 0,
        "score": 0.5, "verdict": "retrying", "target_model": "groq-llama-3.1-8b",
        "question": "How long does the battery last?",
        "answer_snippet": "The battery lasts 12 hours and comes in crimson red.",
        "context": "system: The X100 battery lasts 10 hours.",
        "claims": json.dumps([
            {"statement": "The battery lasts 12 hours.", "verdict": 0,
             "reason": "The context states 10 hours, not 12."},
            {"statement": "It comes in crimson red.", "verdict": 0,
             "reason": "No color is mentioned in the context."},
        ]),
        "error": None, "duration_ms": 1450, "threshold": 0.7,
    }
    base.update(over)
    return base


def test_claims_table_shows_verdicts_and_reasons():
    out = claims_html(event()["claims"])
    assert "0/2 claims supported" in out
    assert "NOT supported" in out
    assert "The context states 10 hours, not 12." in out
    assert "No color is mentioned in the context." in out


def test_request_card_shows_full_story():
    events = [
        event(),
        event(id=2, attempt=1, score=1.0, verdict="passed",
              answer_snippet="The battery lasts 10 hours.",
              claims=json.dumps([{"statement": "The battery lasts 10 hours.",
                                  "verdict": 1, "reason": "Directly stated."}])),
    ]
    out = request_html("abc123", events)
    assert "grounding context" in out
    assert "The X100 battery lasts 10 hours." in out           # context visible
    assert "crimson red" in out                                 # attempt-0 answer visible
    assert "regenerated answer (retry 1)" in out                # retry labelled
    assert "retrying" in out and "passed" in out                # both verdicts
    assert "2 attempt(s)" in out
    assert "2900 ms total judge time" in out


def test_request_card_shows_error_events():
    out = request_html("e1", [event(score=None, verdict="scorer_error_blocked",
                                    claims=None, error="judge scoring timed out after 60s")])
    assert "judge scoring timed out" in out
    assert "scorer_error_blocked" in out


def test_html_is_escaped():
    out = request_html("x", [event(answer_snippet="<script>alert(1)</script>", claims=None)])
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_score_bar_colors_by_threshold():
    assert "ok" in score_bar(0.9, 0.7)
    assert "bad" in score_bar(0.5, 0.7)
    assert "—" in score_bar(None, 0.7)
