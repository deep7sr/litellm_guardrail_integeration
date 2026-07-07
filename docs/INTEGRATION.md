# Integrating your app with the LLM proxy (faithfulness guardrail)

*The one-pager for application teams. If you already build RAG requests the
standard OpenAI way, there is nothing extra to do.*

All LLM traffic routes through the central LiteLLM proxy. If your app is a
**RAG app** (it retrieves documents and asks the model to answer from them),
the proxy automatically verifies every answer is grounded in your retrieved
documents before your users see it.

## The contract: standard OpenAI RAG message structure

The guardrail reads context and question straight from your `messages` array
— **no metadata, no custom fields, no SDK**:

```python
context_str = "--- Retrieved Evidence ---\n" + "\n\n".join(retrieved_chunks)

client.chat.completions.create(
    model="...",
    messages=[
        {"role": "system",    "content": "You are a factual assistant. Use ONLY the retrieved context..."},
        {"role": "assistant", "content": context_str},        # your retrieved chunks
        {"role": "user",      "content": user_question},       # the question
    ],
)
```

This is OpenAI's own documented RAG format. The guardrail identifies:

| Part | Where it looks |
|---|---|
| **Context** | the last `assistant`-role message starting with `--- Retrieved Evidence ---` |
| **Question** | the last `user`-role message |

## The rules (only two)

1. **The evidence marker line matters.** `--- Retrieved Evidence ---` at the
   top of your context message is how the guardrail tells *injected context*
   apart from *the model's previous answers* in multi-turn conversations
   (both are `assistant`-role). Keep the marker exactly as shown.
2. **Only your backend writes the evidence message.** Never place user-typed
   text into it. (The guardrail also ignores the marker in `user`-role
   messages, so end users cannot forge context — but don't rely on that as
   an excuse to blur the line in your own code.)

**Not a RAG app?** Do nothing. Requests without an evidence-marked message
pass through completely unscored — logged as `unscored` on the dashboard,
never blocked, never slowed by scoring.

## What happens on failure

Behavior is central proxy configuration (`GUARDRAIL_MODE`), not per-request:

- **`retry`** (default) — the proxy regenerates the answer with corrective
  feedback up to `MAX_ATTEMPTS` times; if it still fails, your app receives
  a fixed safe fallback message instead of the ungrounded answer. With the
  reasoning guardrail active, the fallback includes a short judge-written
  explanation of what wasn't supported.
- **`block`** — the request fails with a structured 400:
  `{"error": ..., "guardrail": "faithfulness-guardrail", "score": 0.33,
  "threshold": 0.7, "reason": "..."}` (reason present with the reasoning
  variant).

## Multi-turn conversations

Follow the standard OpenAI pattern (append prior turns), and add a **fresh
evidence message before each new user question** when you re-retrieve. The
guardrail always scores against the *newest* marked evidence message.

## FAQ

**Latency cost?** One scoring pass (judge model, claim-by-claim) after
generation; failed answers add regeneration round-trips in retry mode.
Scoring is bounded by a timeout — a slow judge cannot hang your request.

**What if the judge/scoring infrastructure is down?** The proxy fails open:
your response is served unscored and logged as `error`. Your app never
hard-depends on the guardrail being up.

**Streaming?** Guarded routes are currently non-streaming. Talk to the
platform team if you need streaming on a RAG route.
