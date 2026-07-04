"""Extract question / retrieval context / answer from a Langfuse trace of a
LiteLLM chat completion.

Context contract (in priority order):
1. Explicit metadata contract (preferred, not attacker-controllable by end
   users): the calling app sends
       extra_body={"metadata": {"context": ["chunk 1", "chunk 2"]}}
   LiteLLM forwards request metadata into the Langfuse trace metadata.
2. Fallback convention: all system-role message contents are treated as
   retrieved context (matches the guardrail's behaviour).

A trace with no context under either rule is not scorable for faithfulness
and is skipped rather than scored against empty context.
"""

from __future__ import annotations

from typing import Any


def _messages_from_trace_input(trace_input: Any) -> list[dict]:
    """LiteLLM's Langfuse integration logs the request messages as the trace
    input. Depending on version it is either the messages list itself or a
    dict like {"messages": [...]}."""
    if isinstance(trace_input, dict):
        messages = trace_input.get("messages", [])
    elif isinstance(trace_input, list):
        messages = trace_input
    else:
        return []
    return [m for m in messages if isinstance(m, dict)]


def _content_as_text(content: Any) -> str:
    """Chat message content can be a plain string or a list of content parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(p for p in parts if p)
    return ""


def extract_context(trace_metadata: Any, messages: list[dict]) -> list[str]:
    """Return retrieval context chunks, or [] if none can be determined."""
    # 1. Explicit metadata contract.
    if isinstance(trace_metadata, dict):
        ctx = trace_metadata.get("context")
        if isinstance(ctx, str) and ctx.strip():
            return [ctx]
        if isinstance(ctx, list):
            chunks = [str(c) for c in ctx if isinstance(c, (str, int, float)) and str(c).strip()]
            if chunks:
                return chunks

    # 2. Fallback: system-role messages.
    chunks = []
    for m in messages:
        if m.get("role") == "system":
            text = _content_as_text(m.get("content"))
            if text.strip():
                chunks.append(text)
    return chunks


def extract_question(messages: list[dict]) -> str:
    """Return the most recent user-role message.

    Known limitation (documented in the project brief): in multi-turn
    conversations the originating question may be an earlier turn. We keep
    the full user-turn history joined as a fallback signal when the last
    turn is very short (e.g. "restate that").
    """
    user_turns = [
        _content_as_text(m.get("content"))
        for m in messages
        if m.get("role") == "user"
    ]
    user_turns = [t for t in user_turns if t.strip()]
    if not user_turns:
        return ""
    last = user_turns[-1]
    if len(last) < 40 and len(user_turns) > 1:
        # Short follow-up turn: include prior user turns for context.
        return "\n".join(user_turns)
    return last


def extract_answer(trace_output: Any) -> str:
    """Trace output from LiteLLM's Langfuse logger: either the assistant
    message dict, a plain string, or a full response-ish dict."""
    if isinstance(trace_output, str):
        return trace_output
    if isinstance(trace_output, dict):
        if isinstance(trace_output.get("content"), str):
            return trace_output["content"]
        choices = trace_output.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            if isinstance(msg.get("content"), str):
                return msg["content"]
    return ""
