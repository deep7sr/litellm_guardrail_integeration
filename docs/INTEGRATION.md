# Integrating with the grounding guardrail

**By default, no integration is required.** Any application that routes
chat completions through this proxy is automatically protected: if a request
carries no explicit context, the guardrail grounds the answer against the
entire conversation the model was shown (system prompts, user messages,
tool outputs — the model's own earlier replies are deliberately excluded).
End users and application developers see normal OpenAI-compatible behavior;
they never interact with the guardrail.

That transparent mode guarantees **the model invented nothing beyond what
it was told**. It has one documented blind spot: if the *user themselves*
asserts a false fact and the model repeats it, the answer is "grounded in
the conversation" and passes. That is not model hallucination, but if your
application needs answers verified against retrieved documents *only* —
immune to user-asserted claims — opt in to strict grounding below.

## Optional strict mode: explicit retrieved context

A RAG application can upgrade its guarantee by having its **server-side
backend** (never the end user — this field is invisible to them) attach the
retrieved chunks to each request:

```python
from openai import OpenAI

client = OpenAI(base_url="https://<proxy-host>:4000/v1", api_key=YOUR_PROXY_KEY)

chunks = my_retriever.search(user_question)   # your existing retrieval step

resp = client.chat.completions.create(
    model="groq-llama-3.1-8b",
    messages=[
        {"role": "system", "content": "Answer using the provided context.\n\n" + "\n\n".join(chunks)},
        {"role": "user", "content": user_question},
    ],
    extra_body={
        "metadata": {
            # REQUIRED: the exact retrieved chunks the model was given.
            "guardrail_context": chunks,
            # RECOMMENDED: the user's original question (improves claim
            # decomposition in multi-turn conversations).
            "guardrail_question": user_question,
        }
    },
)
```

Rules:

- `guardrail_context` is set once in the app backend; it is invisible to end
  users, who keep chatting exactly as before.
- `guardrail_context` must be a **list of non-empty strings** (a single
  string is accepted and wrapped). Send the same text you injected into the
  prompt — no more, no less. Padding it with extra documents inflates
  scores; omitting documents causes false blocks.
- `guardrail_question` should be the originating question, not the latest
  conversational turn ("restate that", "make it shorter", ...).
- How you arrange `messages` is entirely your business — the guardrail no
  longer cares about roles.

## Why explicit metadata is the strong form

1. **Security.** Anything inside `messages` can be influenced by the end
   user. We confirmed a real attack in testing: a user embedded a fabricated
   product detail in their own message ("to confirm what we know so far...")
   and the guardrail treated it as ground truth, scoring the fabrication as
   perfectly grounded (1.0). Metadata is set by your *server-side* code,
   which is inside the trust boundary; the end user's keystrokes never
   reach it.
2. **Precision.** Scoring against retrieved documents only (rather than the
   whole conversation) verifies factual grounding in your knowledge base,
   not merely internal consistency with the chat.
3. **Industry precedent.** AWS Bedrock Guardrails (contextual grounding) and
   Azure AI groundedness detection both take grounding sources explicitly
   from the calling application, for exactly these reasons.

## Modes the proxy operator can set

- `RAGAS_ON_MISSING_CONTEXT=prompt` (default): transparent full-conversation
  grounding for requests without metadata context.
- `RAGAS_ON_MISSING_CONTEXT=block`: strict — context-less requests are
  rejected with HTTP 400 (for proxies serving only integrated RAG apps).
- `RAGAS_ON_MISSING_CONTEXT=skip`: context-less requests pass unscored
  (logged with verdict `skipped_no_context`); not recommended.

## What happens on every scored request

1. The model's answer is decomposed into factual claims by the judge model;
   each claim is verified against the grounding source (explicit context if
   provided, otherwise the conversation). Score = fraction of claims
   supported.
2. Score ≥ threshold (default 0.7) → answer returned unchanged.
3. Score below threshold → the proxy regenerates the answer with corrective
   feedback (up to `RAGAS_MAX_RETRIES` times) and re-scores.
4. Still failing → you receive a fixed fallback message instead of the
   ungrounded answer. An answer with no checkable claims (e.g. an honest
   "the documents don't say") passes.

Streaming requests are supported but are **buffered**: the client receives
the stream only after scoring completes. Grounding cannot be verified on a
partial answer.

## Latency budget

Expect roughly 1.5–4 s of added latency per request for scoring (two judge
round-trips), multiplied on retries. Size the judge deployment accordingly
and set client timeouts to ≥ 60 s for guarded routes.
