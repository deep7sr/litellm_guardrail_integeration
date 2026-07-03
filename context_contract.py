"""Grounding-source extraction for the RAGAS faithfulness guardrail.

The guardrail is transparent middleware: applications send completely
normal OpenAI-format requests, and the grounding source for every answer is
the full conversation the model was shown (minus the model's own earlier
replies). There is no integration contract and no metadata field — nothing
an application or end user has to change.

The guarantee this provides: the model invented nothing beyond what it was
told (hallucination detection). Its documented limit: a falsehood asserted
by the user is part of the conversation, so the model repeating it is
"grounded" — that is user-supplied misinformation, not model hallucination,
and cannot be detected without an out-of-band source of trusted documents.

This module is deliberately import-light (stdlib only) so the extraction
logic can be unit-tested without litellm/ragas installed.
"""

INTERNAL_FLAG = "ragas_guardrail_internal"


def _candidate_metadata_dicts(data: dict):
    """Yield every dict where LiteLLM may have placed request metadata
    (used only for the internal-call flag)."""
    md = data.get("metadata")
    if isinstance(md, dict):
        yield md
        rm = md.get("requester_metadata")
        if isinstance(rm, dict):
            yield rm
    lp = data.get("litellm_params")
    if isinstance(lp, dict):
        lpmd = lp.get("metadata")
        if isinstance(lpmd, dict):
            yield lpmd
            rm = lpmd.get("requester_metadata")
            if isinstance(rm, dict):
                yield rm


def _message_text(content) -> str:
    """Best-effort plain text from a message's content (handles the
    multimodal list-of-parts format)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            p.get("text") for p in content
            if isinstance(p, dict) and isinstance(p.get("text"), str)
        ]
        return "\n".join(p for p in parts if p)
    return ""


def prompt_grounding_context(messages: list):
    """The grounding source: the entire conversation the model was shown,
    role-labelled, as a single string in a one-element list.

    Assistant turns are excluded: if the model hallucinated earlier in the
    conversation, its own prior output must not become "ground truth" that
    lets it repeat the hallucination with a passing score. Legitimately
    grounded prior answers remain verifiable from the same non-assistant
    content they were grounded in.
    """
    lines = []
    for m in messages or []:
        if not isinstance(m, dict) or m.get("role") == "assistant":
            continue
        text = _message_text(m.get("content")).strip()
        if text:
            lines.append(f"{m.get('role', 'unknown')}: {text}")
    return ["\n\n".join(lines)] if lines else None


def extract_question(data: dict) -> str:
    """The last plain-text user message. Only used to frame RAGAS's claim
    decomposition — it is never part of the grounding evidence, so a
    conversational last turn ("restate that") costs a little precision,
    never correctness."""
    for m in reversed(data.get("messages") or []):
        if m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"].strip():
            return m["content"]
    return ""


def is_internal_call(data: dict) -> bool:
    """True for the guardrail's own regeneration calls, which must never be
    re-scored (prevents recursive self-triggering)."""
    return any(bool(md.get(INTERNAL_FLAG)) for md in _candidate_metadata_dicts(data))
