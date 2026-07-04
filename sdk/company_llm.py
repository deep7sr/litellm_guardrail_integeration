"""
Internal helper for calling the company LiteLLM proxy under the
faithfulness-guardrail contract.

This is the ONE canonical implementation of the contract. App teams should
use `guarded_completion` / `async_guarded_completion` instead of hand-building
the metadata, so that if the contract ever evolves, only this module changes.

Usage:

    from openai import OpenAI
    from company_llm import guarded_completion, GuardrailBlockedError

    client = OpenAI(base_url="https://llm-proxy.internal:4000/v1", api_key=TEAM_KEY)

    try:
        resp = guarded_completion(
            client,
            model="groq-llama-3.1-8b",
            messages=messages,          # prompt however your app likes
            context=retrieved_chunks,   # the chunks YOUR retrieval returned
            question=user_question,     # the user's original question
        )
        answer = resp.choices[0].message.content
    except GuardrailBlockedError as e:
        # The proxy refused to serve an ungrounded answer.
        answer = f"Sorry, I couldn't find a well-supported answer. (score={e.score})"
"""

from openai import AsyncOpenAI, BadRequestError, OpenAI


class GuardrailBlockedError(Exception):
    """Raised when the proxy blocks a response for failing the faithfulness check."""

    def __init__(self, score=None, threshold=None, raw=None):
        self.score = score
        self.threshold = threshold
        self.raw = raw
        super().__init__(
            f"Response blocked by faithfulness guardrail "
            f"(score={score}, threshold={threshold})"
        )


def _build_metadata(context, question, on_fail=None, threshold=None):
    if isinstance(context, str):
        context = [context]
    if not context or not all(isinstance(c, str) and c.strip() for c in context):
        raise ValueError(
            "context must be a non-empty string or list of non-empty strings — "
            "pass the chunks your retrieval step returned"
        )
    metadata = {
        "guardrail_context": list(context),
        "guardrail_question": question,
    }
    if on_fail is not None:
        if on_fail not in ("block", "retry"):
            raise ValueError('on_fail must be "block" or "retry"')
        metadata["guardrail_on_fail"] = on_fail
    if threshold is not None:
        metadata["guardrail_threshold"] = float(threshold)
    return metadata


def _blocked_error_from(e: BadRequestError) -> GuardrailBlockedError | None:
    """Translate the proxy's structured 400 into a typed exception, if it is one."""
    try:
        body = e.body if isinstance(e.body, dict) else {}
        detail = body.get("error", body)
        if isinstance(detail, dict) and isinstance(detail.get("message"), dict):
            detail = detail["message"]
        if isinstance(detail, dict) and detail.get("guardrail") == "ragas-faithfulness":
            return GuardrailBlockedError(
                score=detail.get("score"), threshold=detail.get("threshold"), raw=body
            )
    except Exception:
        pass
    return None


def guarded_completion(client: OpenAI, *, model, messages, context, question,
                       on_fail=None, threshold=None, **kwargs):
    """Chat completion through the proxy with the guardrail contract attached.

    Extra keyword arguments (temperature, max_tokens, ...) pass through to
    the OpenAI client unchanged. Raises GuardrailBlockedError if the proxy
    blocks the response as ungrounded.
    """
    metadata = _build_metadata(context, question, on_fail, threshold)
    try:
        return client.chat.completions.create(
            model=model, messages=messages,
            extra_body={"metadata": metadata}, **kwargs,
        )
    except BadRequestError as e:
        blocked = _blocked_error_from(e)
        if blocked is not None:
            raise blocked from e
        raise


async def async_guarded_completion(client: AsyncOpenAI, *, model, messages, context,
                                   question, on_fail=None, threshold=None, **kwargs):
    """Async variant of guarded_completion."""
    metadata = _build_metadata(context, question, on_fail, threshold)
    try:
        return await client.chat.completions.create(
            model=model, messages=messages,
            extra_body={"metadata": metadata}, **kwargs,
        )
    except BadRequestError as e:
        blocked = _blocked_error_from(e)
        if blocked is not None:
            raise blocked from e
        raise
