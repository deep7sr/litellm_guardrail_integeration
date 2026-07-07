"""
LiteLLM post-call faithfulness guardrail — reasoning variant (v2).

Builds on faithfulness_guardrail.FaithfulnessGuardrail (v1). Identical
detection, retry, and enforcement behavior, plus: when an answer fails the
faithfulness check, the judge model is asked to explain WHY — which
specific claim(s) in the answer are not supported by the context. That
explanation is:

  - returned to the caller in the block-mode 400 error (detail["reason"]),
  - appended to the fallback message served after retry exhaustion,
  - used to make retry feedback specific ("Specifically: <reason>") instead
    of generic, which helps the generation model converge faster,
  - logged to the events table (reason column) for the dashboard.

Design notes:
  - The reason comes from ONE extra judge call per failed attempt, made
    through this same proxy against JUDGE_MODEL. Recursion guard 1 in the
    base class (skip requests whose model is JUDGE_MODEL) already covers it,
    so this cannot self-trigger the guardrail.
  - Reason generation is strictly best-effort: any error or timeout returns
    None and the guardrail behaves exactly like v1. A broken reason step
    must never change enforcement outcomes.
  - We ask the judge directly instead of relying on RAGAS MetricResult
    internals, which are not a stable public API across versions.

Register in the LiteLLM config as:
    guardrail: faithfulness_guardrail_reasoning.FaithfulnessReasoningGuardrail
IMPORTANT: enable EITHER this OR the v1 guardrail, never both — both firing
on post_call would double-score every request.
"""

import asyncio
import logging

from faithfulness_guardrail import (
    FaithfulnessGuardrail,
    JUDGE_MODEL,
    JUDGE_TIMEOUT_SECONDS,
    _proxy_client,
    _strip_reasoning,
)

logger = logging.getLogger("faithfulness_guardrail")

REASON_MAX_TOKENS = 150

REASON_SYSTEM_PROMPT = (
    "You are a verification assistant. An AI answer was checked against the "
    "provided reference context and found not fully supported. In one or two "
    "short sentences, state which specific claim(s) in the answer are not "
    "supported by the context. Be concise, neutral, and factual. Do not "
    "apologize, do not suggest fixes, do not repeat the whole answer."
)


class FaithfulnessReasoningGuardrail(FaithfulnessGuardrail):
    async def _failure_reason(self, question: str, contexts: list[str],
                              answer: str, score: float) -> str | None:
        try:
            resp = await asyncio.wait_for(
                _proxy_client.chat.completions.create(
                    model=JUDGE_MODEL,
                    temperature=0,
                    max_tokens=REASON_MAX_TOKENS,
                    messages=[
                        {"role": "system", "content": REASON_SYSTEM_PROMPT},
                        {"role": "user", "content": (
                            f"Reference context:\n{chr(10).join(contexts)}\n\n"
                            f"Question: {question}\n\n"
                            f"Answer being checked:\n{answer}\n\n"
                            f"Faithfulness score: {score:.2f} (below threshold). "
                            "Which claim(s) are unsupported?"
                        )},
                    ],
                ),
                timeout=JUDGE_TIMEOUT_SECONDS,
            )
            reason = _strip_reasoning(resp.choices[0].message.content or "")
            reason = " ".join(reason.split())  # collapse whitespace/newlines
            if not reason:
                return None
            # Keep the user-facing explanation short even if the judge rambles.
            if len(reason) > 400:
                reason = reason[:397].rstrip() + "..."
            return reason
        except asyncio.TimeoutError:
            logger.warning(f"[faithfulness] reason generation timed out after {JUDGE_TIMEOUT_SECONDS}s")
            return None
        except Exception as e:
            logger.warning(f"[faithfulness] reason generation failed: {e}")
            return None
