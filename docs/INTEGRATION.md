# Integration contract for RAG applications

Every RAG application routed through this LiteLLM proxy must pass its
retrieved context **explicitly in request metadata**. The guardrail never
infers context from message roles or message content.

## Why a metadata contract (read this before objecting)

1. **Security.** Anything inside `messages` can be influenced by the end
   user. We confirmed a real attack in testing: a user embedded a fabricated
   product detail in their own message ("to confirm what we know so far...")
   and a role-based extractor treated it as ground truth, scoring the
   fabrication as perfectly grounded (1.0). Metadata is set by your
   *server-side* code, which is inside the trust boundary; the end user's
   keystrokes never reach it.
2. **Correctness across teams.** Role conventions differ per app
   (context-in-system, context-in-user, multi-turn). A metadata field is the
   same for everyone and cannot silently extract zero context.
3. **Industry precedent.** AWS Bedrock Guardrails (contextual grounding) and
   Azure AI groundedness detection both require the caller to pass grounding
   sources explicitly, for exactly these reasons.

## What to send

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

- `guardrail_context` must be a **list of non-empty strings** (a single
  string is accepted and wrapped). Send the same text you injected into the
  prompt — no more, no less. Padding it with extra documents inflates
  scores; omitting documents causes false blocks.
- `guardrail_question` should be the originating question, not the latest
  conversational turn ("restate that", "make it shorter", ...).
- How you arrange `messages` is entirely your business — the guardrail no
  longer cares about roles.

## What happens if you don't send context

Default (`RAGAS_ON_MISSING_CONTEXT=block`): the request is rejected with
HTTP 400 and an error explaining this contract. This is deliberate — a
grounding guardrail that silently skips unintegrated apps protects nothing.

During migration the proxy operator may set `RAGAS_ON_MISSING_CONTEXT=skip`,
which lets context-less requests through unscored (logged with verdict
`skipped_no_context` so adoption can be tracked in the dashboard).

## What happens after you send it

1. The model's answer is decomposed into factual claims by the judge model;
   each claim is verified against `guardrail_context`. Score = fraction of
   claims supported.
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
