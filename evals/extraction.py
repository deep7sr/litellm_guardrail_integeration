"""Turn a Langfuse trace (as logged by LiteLLM's langfuse callback) into an
evaluable case: question, answer, and — for RAG traffic — retrieved contexts.

Mirrors the guardrail's message-structure contract exactly:
  - context  = LAST assistant-role message carrying the evidence marker
               (assistant-role only — the same injection-safety invariant);
  - question = last user-role message;
  - <think>...</think> reasoning traces are stripped from the answer.

The trace `input`/`output` shapes vary across LiteLLM versions, so
normalization is deliberately defensive. Anything unextractable returns
None and is skipped (counted, never fatal).
"""

import re
from dataclasses import dataclass

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class EvalCase:
    question: str
    answer: str
    contexts: list[str] | None  # None => non-RAG traffic

    @property
    def is_rag(self) -> bool:
        return self.contexts is not None


def strip_reasoning(text: str) -> str:
    if not text:
        return text
    return _THINK_BLOCK_RE.sub("", text).strip()


def _get(obj, key, default=None):
    """Attribute-or-dict accessor (Langfuse SDK objects vs plain dicts)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def normalize_messages(trace_input) -> list[dict]:
    """LiteLLM logs trace input as either {'messages': [...]} or the raw
    messages list. Returns [] when neither shape matches."""
    if isinstance(trace_input, dict):
        msgs = trace_input.get("messages")
    else:
        msgs = trace_input
    if not isinstance(msgs, list):
        return []
    return [m for m in msgs if isinstance(m, dict)]


def normalize_answer(trace_output) -> str:
    """Trace output may be a string, an assistant message dict, or a nested
    {'content': ...} / {'choices': [...]} response object."""
    if isinstance(trace_output, str):
        return trace_output
    if isinstance(trace_output, dict):
        content = trace_output.get("content")
        if isinstance(content, str):
            return content
        choices = trace_output.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
    return ""


def extract_contexts(messages: list[dict], marker: str) -> list[str] | None:
    """Same rule as the guardrail: last assistant-role message carrying the
    evidence marker; user-role content is NEVER treated as context."""
    mk = marker.lower()
    for m in reversed(messages):
        if m.get("role") == "assistant" and isinstance(m.get("content"), str):
            content = m["content"]
            low = content.lower()
            if mk in low:
                body = content[low.index(mk) + len(marker):].strip()
                if body:
                    return [body]
    return None


def extract_question(messages: list[dict]) -> str:
    return next(
        (m["content"] for m in reversed(messages)
         if m.get("role") == "user" and isinstance(m.get("content"), str)),
        "",
    )


def build_case(trace, marker: str) -> EvalCase | None:
    messages = normalize_messages(_get(trace, "input"))
    answer = strip_reasoning(normalize_answer(_get(trace, "output")))
    question = extract_question(messages)
    if not question or not answer:
        return None
    return EvalCase(
        question=question,
        answer=answer,
        contexts=extract_contexts(messages, marker),
    )
