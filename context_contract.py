"""Context/question extraction contract for the RAGAS faithfulness guardrail.

Trust model
-----------
The RAG *application* (server-side code that calls the proxy) is trusted.
The *end user* typing into that application is not.

Retrieved context therefore MUST arrive via request metadata, which only the
application's server-side code can set:

    client.chat.completions.create(
        model="...",
        messages=[...],
        extra_body={
            "metadata": {
                "guardrail_context": ["chunk 1", "chunk 2", ...],
                "guardrail_question": "the user's original question",
            }
        },
    )

Context is NEVER inferred from message content. This closes the confirmed
prompt-injection vulnerability where a user could embed "established facts"
inside their own message and have the guardrail treat them as ground truth,
and it removes all dependence on message-role conventions across teams.

This module is deliberately import-light (stdlib only) so the extraction
logic can be unit-tested without litellm/ragas installed.
"""

CONTEXT_KEY = "guardrail_context"
QUESTION_KEY = "guardrail_question"
INTERNAL_FLAG = "ragas_guardrail_internal"


def _candidate_metadata_dicts(data: dict):
    """Yield every dict where LiteLLM may have placed request metadata.

    The primary location is data["metadata"] (verified working in this
    deployment), but LiteLLM versions have also nested requester metadata
    under "requester_metadata" and under litellm_params, so we check all of
    them defensively.
    """
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


def extract_context(data: dict):
    """Return the list of retrieved context strings, or None if absent.

    A present-but-malformed value (wrong type, empty list, empty strings)
    is treated as missing so the guardrail fails closed rather than scoring
    against garbage.
    """
    for md in _candidate_metadata_dicts(data):
        if CONTEXT_KEY not in md:
            continue
        ctx = md.get(CONTEXT_KEY)
        if isinstance(ctx, str) and ctx.strip():
            return [ctx]
        if isinstance(ctx, list):
            parts = [c for c in ctx if isinstance(c, str) and c.strip()]
            if parts:
                return parts
        return None
    return None


def extract_question(data: dict) -> str:
    """Prefer the explicit metadata question; fall back to the last plain-text
    user message. The fallback only affects RAGAS's claim-decomposition
    framing, never what counts as ground-truth context."""
    for md in _candidate_metadata_dicts(data):
        q = md.get(QUESTION_KEY)
        if isinstance(q, str) and q.strip():
            return q
    for m in reversed(data.get("messages") or []):
        if m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"].strip():
            return m["content"]
    return ""


def is_internal_call(data: dict) -> bool:
    """True for the guardrail's own regeneration calls, which must never be
    re-scored (prevents recursive self-triggering)."""
    return any(bool(md.get(INTERNAL_FLAG)) for md in _candidate_metadata_dicts(data))
