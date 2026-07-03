"""Unit tests for the grounding extraction. Run with: pytest tests/ -v
(No litellm/ragas install needed — context_contract is stdlib-only.)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from context_contract import (  # noqa: E402
    extract_question,
    is_internal_call,
    prompt_grounding_context,
)


# ---- prompt_grounding_context (the grounding source) ------------------------

def test_grounding_includes_system_and_user():
    ctx = prompt_grounding_context([
        {"role": "system", "content": "The X100 battery lasts 10 hours."},
        {"role": "user", "content": "How long does the battery last?"},
    ])
    assert ctx is not None and len(ctx) == 1
    assert "battery lasts 10 hours" in ctx[0]
    assert "How long does the battery last?" in ctx[0]


def test_grounding_excludes_assistant_turns():
    # The model's own earlier output must never become ground truth.
    ctx = prompt_grounding_context([
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "the moon is made of cheese"},
        {"role": "user", "content": "restate that"},
    ])
    assert "cheese" not in ctx[0]


def test_grounding_includes_tool_outputs():
    ctx = prompt_grounding_context([
        {"role": "user", "content": "what's the weather?"},
        {"role": "tool", "content": "temperature: 21C, sunny"},
    ])
    assert "21C" in ctx[0]


def test_grounding_handles_multimodal_content():
    ctx = prompt_grounding_context([
        {"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "x"}},
        ]},
    ])
    assert ctx is not None and "what is this?" in ctx[0]


def test_grounding_empty_conversation_returns_none():
    assert prompt_grounding_context([]) is None
    assert prompt_grounding_context([{"role": "assistant", "content": "hi"}]) is None
    assert prompt_grounding_context([{"role": "user", "content": "   "}]) is None


# ---- extract_question ------------------------------------------------------

def test_question_is_last_user_message():
    data = {"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "what is the warranty?"},
    ]}
    assert extract_question(data) == "what is the warranty?"


def test_question_skips_non_string_content():
    data = {"messages": [
        {"role": "user", "content": "real question"},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]},
    ]}
    assert extract_question(data) == "real question"


def test_question_empty_when_nothing_available():
    assert extract_question({"messages": []}) == ""
    assert extract_question({}) == ""


# ---- is_internal_call ------------------------------------------------------

def test_internal_flag_detected():
    assert is_internal_call({"metadata": {"ragas_guardrail_internal": True}})
    assert is_internal_call({"litellm_params": {"metadata": {"ragas_guardrail_internal": True}}})


def test_normal_call_is_not_internal():
    assert not is_internal_call({"metadata": {}})
    assert not is_internal_call({})
    assert not is_internal_call({"metadata": {"ragas_guardrail_internal": False}})
