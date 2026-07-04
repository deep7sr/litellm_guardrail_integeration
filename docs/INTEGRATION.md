# Integrating your app with the LLM proxy (faithfulness guardrail)

*The one-pager for application teams. Total integration effort: ~2 lines of code.*

All LLM traffic in the company routes through the central LiteLLM proxy. If
your application is a **RAG app** (it retrieves documents and asks the model
to answer from them), the proxy will automatically verify that every answer
is actually grounded in your retrieved documents before your users see it.

For that to work, the proxy needs to know **which text is your retrieved
context**. It will not guess this from your prompt — you tell it explicitly.

## The fastest way: use the helper

```python
from openai import OpenAI
from company_llm import guarded_completion, GuardrailBlockedError

client = OpenAI(base_url="<proxy URL>/v1", api_key="<your team's key>")

chunks = my_vector_store.search(user_question)      # you already do this
messages = build_my_prompt(chunks, user_question)   # and this

try:
    resp = guarded_completion(
        client,
        model="groq-llama-3.1-8b",
        messages=messages,
        context=chunks,          # <- new: the chunks you retrieved
        question=user_question,  # <- new: the user's original question
    )
    answer = resp.choices[0].message.content
except GuardrailBlockedError as e:
    answer = "I couldn't find a well-supported answer in the documentation."
```

That's the whole integration. `context` and `question` are variables your
code already has in hand — the contract just asks you not to throw the
labels away when you flatten everything into a prompt.

## What you get for it

- Every response is scored 0.0–1.0 for groundedness (RAGAS Faithfulness,
  judged claim-by-claim against *your* chunks) before it reaches your user.
- Ungrounded answers never reach your user: by default the request fails
  with a clean, catchable `GuardrailBlockedError` carrying the score.
- Your app's pass/block rates appear on the central dashboard automatically.

## Rules

1. **`context` must contain ONLY text your backend retrieved** from your
   document store. Never put user-typed text in it — user text is exactly
   what the guardrail is checking *against* the context, and putting user
   claims into `context` would let users validate their own fabrications.
2. **`question` should be the user's original question**, even in multi-turn
   chats where the latest message is "rephrase that" or "make it shorter."
3. **Not a RAG app?** (plain chat, summarization of user-provided text,
   codegen) — do nothing. Your traffic passes through unscored; that's
   expected and correct.

## Options

| Parameter | Default | Meaning |
|---|---|---|
| `on_fail="block"` | `block` | Fail fast with `GuardrailBlockedError` (recommended). |
| `on_fail="retry"` | — | Proxy silently regenerates with corrective feedback (up to 3×) before giving up. Adds latency; opt in only if you prefer waiting over handling an error. |
| `threshold=0.7` | `0.7` | Minimum fraction of answer claims that must be supported by your context. Raise for high-stakes apps. |

## Raw format (if you can't use the helper)

Any OpenAI-compatible client works — attach the metadata yourself:

```python
client.chat.completions.create(
    model="...", messages=[...],
    extra_body={"metadata": {
        "guardrail_context": ["chunk 1", "chunk 2"],
        "guardrail_question": "the user's question",
    }},
)
```

A blocked response is an HTTP 400 whose error detail includes
`"guardrail": "ragas-faithfulness"`, the `score`, and the `threshold`.

## FAQ

**Does this change my prompt or my retrieval?** No. Build your prompt
exactly as before. The metadata is a *copy* of the chunks, out-of-band; the
generation model never sees it.

**What's the latency cost?** One scoring pass after generation (a claim
decomposition + verification by the judge model). In `retry` mode, failures
add regeneration round-trips — that's why `block` is the default.

**What if the scoring service is down?** The proxy fails open: your response
is served unscored and the event is logged. Your app never hard-depends on
the guardrail being up.

**Streaming?** Guarded routes are currently non-streaming. If you need
streaming on a RAG route, talk to the platform team first.
