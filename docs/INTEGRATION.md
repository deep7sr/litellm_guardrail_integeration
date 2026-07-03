# Integrating with the grounding guardrail

**There is nothing to integrate.** Point your application's OpenAI client at
this proxy's base URL and send completely normal chat completion requests —
the guardrail is invisible middleware. End users and application developers
never interact with it, configure it, or format anything for it.

## What the guardrail does with your request

For every chat completion, the model's answer is decomposed into factual
claims by a judge model, and each claim is verified against **the
conversation the model was shown** — your system prompts, user messages,
and tool outputs (the model's own earlier replies are deliberately excluded,
so a hallucination in one turn can't legitimize itself in a later turn).

1. Score = fraction of claims supported by the conversation.
2. Score ≥ threshold (default 0.7) → answer returned unchanged.
3. Score below threshold → the proxy regenerates the answer with corrective
   feedback (up to `RAGAS_MAX_RETRIES` times) and re-scores. Usually the
   corrected answer passes and that's what your user receives.
4. Still failing → a fixed fallback message is returned instead of the
   ungrounded answer. Judge errors and timeouts also return the fallback —
   an unverified answer is never served.
5. An answer with no checkable claims (e.g. an honest "the documents don't
   say") passes.

## The guarantee, stated precisely

**The model invented nothing beyond what it was told in the conversation.**

The corollary limit: if an end user asserts a false "fact" in their message
and the model repeats it, that answer is grounded in the conversation and
passes. That is user-supplied misinformation, not model hallucination —
detecting it would require an out-of-band source of trusted documents,
which this deployment has deliberately traded away for zero-integration
transparency.

## Operational notes for consuming teams

- **Latency:** expect roughly 1.5–4 s added per request for scoring (two
  judge round-trips), multiplied on retries. Set client timeouts ≥ 60 s.
- **Streaming:** supported but buffered — the stream is delivered only
  after scoring completes. Grounding can't be verified on a partial answer.
- **Fallback message:** if your users see *"I wasn't able to find a fully
  supported answer to this in the available documents,"* the guardrail
  blocked a persistently ungrounded answer. Check the dashboard (`:8080`)
  for the scored attempts.
