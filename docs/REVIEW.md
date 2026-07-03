# Architecture review — findings and decisions

This is the end-to-end review requested in the project brief, covering both
the listed open problems and issues not previously identified. Each finding
states what was wrong, why it matters in production, and what the rebuilt
code in this repository does about it.

## Verdict on the overall approach

RAGAS faithfulness as a post-call guardrail is a sound, industry-aligned
design — it is structurally the same as AWS Bedrock Guardrails' contextual
grounding check and Azure AI's groundedness detection: an NLI-style judge
verifies the answer against caller-supplied grounding sources, with a
threshold and a blocking policy. No pivot is needed. Two structural things
did have to change, though: how context enters the system (finding 1) and
the streaming bypass (finding 3).

**On "100% accuracy": no LLM-judge (or any classifier) achieves it, and any
design that assumes it will fail silently.** RAGAS faithfulness is a
probabilistic scorer: claim decomposition can split claims wrongly, and
claim verification is an LLM judgment. Published benchmarks for the best
grounding detectors (HHEM, Lynx, MiniCheck, GPT-4-as-judge) sit in the
~75–90% balanced-accuracy range on hard datasets like HaluBench/AggreFact.
What production systems actually guarantee is not perfect detection but a
**fail-closed pipeline**: every answer is either verified above threshold or
never reaches the user, errors block rather than pass, and every decision is
logged for audit. That is what this rebuild delivers, plus a calibration
procedure (finding 7) to place the threshold on measured data instead of
intuition. Plan for false blocks (grounded answers occasionally replaced by
the fallback) as the price of near-zero false passes.

---

## Finding 1 — Context extraction (Open Problem #1): replaced role inference with an explicit metadata contract

**Problem.** Both extraction strategies tried so far are broken in
different ways: reading user messages as context is a confirmed
prompt-injection hole (a user's fabricated "established fact" became
ground truth and scored 1.0); reading only system messages silently
extracts *zero* context for apps that pack context into the user message,
making the guardrail score everything as ungrounded. Any role heuristic
also inherits a subtler flaw: the system prompt's *instructions* ("answer
politely...") get scored as if they were evidence.

**Decision.** Context is now only accepted from
`metadata["guardrail_context"]` (list of strings), set by the RAG
application's server-side code — the same pattern AWS Bedrock and Azure use
(explicit grounding sources). The end user cannot reach metadata, so the
injection class is closed by construction, and there is no cross-team role
convention to enforce. Yes, this is an integration requirement for every
consuming app — that was the option the brief hoped to avoid, but it is the
only option that is simultaneously correct and secure. A convention you
"avoid requiring" is still a requirement; it's just an undocumented,
unenforced one that fails silently instead of failing with a clear 400.

Missing context is **fail-closed by default** (HTTP 400 with a message
pointing to `docs/INTEGRATION.md`); `RAGAS_ON_MISSING_CONTEXT=skip` exists
for migration, and skipped requests are logged so adoption is measurable.
Malformed context (wrong type, empty strings) is treated as missing, not
scored against garbage. See `context_contract.py` and its unit tests.

## Finding 2 — Question extraction (Open Problem #2)

`metadata["guardrail_question"]` is the preferred source; the last
plain-text user message is only a fallback. Note the question only shapes
RAGAS's claim decomposition — with the metadata contract it can no longer
affect what counts as evidence, so the residual risk of a weak fallback is
accuracy, not security.

## Finding 3 — Streaming responses bypassed the guardrail entirely (not in the brief; highest-severity new finding)

The old hook returned early for anything that isn't a `litellm.ModelResponse`
— which includes every streaming response. Any client sending
`stream: true` received completely unguarded output. The rebuild implements
`async_post_call_streaming_iterator_hook`: the stream is buffered, assembled,
scored through the same evaluation path, and either released or replaced
with fallback chunks. Buffering removes streaming's latency benefit; that is
inherent to pre-delivery grounding verification (Bedrock has the same
tradeoff) and must be documented to consuming teams.

## Finding 4 — Self-triggering (Open Problem #4): made structural instead of name-based

The old guards were fragile: a string match on the model name
`"judge-model"` (breaks on rename) and a metadata flag on regeneration
calls that looped back through the proxy's public HTTP endpoint using the
master key (a privilege-escalation risk if the flag check ever regressed,
and a hard dependency on `localhost:4000`). The rebuild removes both hazards
structurally:

- **Judge calls** go directly from the guardrail to the judge endpoint
  (`JUDGE_BASE_URL`), never through the proxy. The judge is no longer in
  `model_list` at all — recursion on judge traffic is impossible.
- **Regeneration calls** use the in-process LiteLLM router
  (`llm_router.acompletion`), which bypasses the proxy request layer and
  its guardrail hooks — no HTTP loopback, no master key in the guardrail.
- The `ragas_guardrail_internal` metadata flag is retained as
  defense-in-depth.

## Finding 5 — Fail-open error paths (not in the brief)

Three paths in the old code served unverified or known-bad answers:
scorer exception → original answer served unmodified; scorer exception
*mid-retry* → the just-regenerated, never-scored answer served; and
`NaN → 1.0` conflated "no checkable claims" with "perfectly grounded."
Now: scorer errors follow `RAGAS_ON_SCORER_ERROR` (default: serve the
fallback, fail-closed) at every point in the loop, scoring has an explicit
timeout (a hung judge can no longer hang user requests indefinitely) and a
concurrency semaphore, and NaN is logged as its own verdict (`no_claims`)
and passed deliberately — an honest refusal has nothing to hallucinate —
rather than disguised as a perfect score.

## Finding 6 — Judge model (Open Problem #3): self-hosted recommendation

**Recommendation: Qwen2.5-32B-Instruct served by vLLM** (compose file ships
with `Qwen/Qwen2.5-7B-Instruct` as the default so it runs on a single 24 GB
GPU; set `JUDGE_MODEL=Qwen/Qwen2.5-32B-Instruct` on ~2×A100/H100 or an
AWQ/GPTQ quant on one 80 GB card for production accuracy). Reasons:

- The Qwen2.5 instruct family is consistently strong at instruction
  following and JSON emission, which is exactly what RAGAS/`instructor`
  stresses. Your observed failure mode (one model returning malformed JSON,
  a sibling size working) is typical of weaker JSON-followers.
- More importantly, **vLLM's guided decoding (xgrammar/outlines) makes
  malformed JSON structurally impossible** — when `instructor` requests a
  JSON schema via `response_format`, vLLM constrains token sampling to that
  schema. The reliability problem stops being a model-selection gamble.
- vLLM exposes an OpenAI-compatible endpoint, so the existing
  `llm_factory(model, client=AsyncOpenAI(...))` wiring is unchanged — only
  `JUDGE_BASE_URL` differs between dev (Groq) and prod (vLLM).

Alternatives considered: Llama-3.3-70B-Instruct (comparable quality, more
VRAM, Meta license); Patronus **Lynx-70B** and Bespoke **MiniCheck-7B** —
purpose-built hallucination judges that outperform general models per
judged token, but they are single-call classifiers, not
instructor-compatible general LLMs, so they can't drive RAGAS's
decompose-then-verify pipeline. They are the right choice **if** you ever
drop RAGAS for a cheaper single-call architecture (see "If you pivot"
below).

## Finding 7 — Threshold (Open Problem #5): keep a single threshold, but calibrate it

A claim-count- or length-dependent threshold sounds appealing but has no
evidential basis and adds an unauditable moving part. Keep one static
threshold per route, but **derive it**: collect ~200+ real (context, answer)
pairs, hand-label each answer grounded/ungrounded, score them with this
exact judge+prompt stack, and pick the threshold from the resulting
precision/recall curve at your required false-pass rate. Re-run whenever the
judge model or RAGAS version changes — the score distribution is a property
of the whole stack, not of RAGAS in the abstract. `0.7` remains the default
until you have that data; it is configurable per deployment via
`RAGAS_PASS_THRESHOLD`. Note the granularity caveat: an answer with 2 claims
can only score 0, 0.5, or 1.0, so short answers are effectively judged
all-or-nothing regardless of threshold — one more reason a fancy dynamic
threshold buys little.

## Finding 8 — Smaller issues fixed in the rebuild

- **Secrets:** master key and DB password were hardcoded in compose; now
  required via `.env` (`:?` guards make compose refuse to start without
  them). The master key is no longer readable by the guardrail at all.
- **Traceability:** events now use LiteLLM's own `litellm_call_id` (when
  present) instead of an unrelated random UUID.
- **Multi-choice responses:** only `choices[0]` is scored/replaced; if any
  app sends `n>1`, additional choices would ship unverified. Left as a
  documented limitation (reject `n>1` at the proxy if this matters to you).
- **Tool/function-call responses** (no text content) are skipped explicitly
  rather than crashing or scoring empty strings.
- **PII:** the events table stores question/answer snippets in plaintext.
  Add a retention policy and restrict DB access before real user traffic.
- **Image name** `litellm-hhem:latest` renamed to
  `litellm-ragas-guardrail`, built from a checked-in Dockerfile instead of
  a hand-built local image.
- **Retry-exhaustion path** now has unit-testable structure; still worth one
  forced end-to-end test in staging (point `RAGAS_PASS_THRESHOLD=1.01` at a
  test route to force exhaustion deliberately — cheaper than hunting for an
  adversarial prompt the model falls for).

## If you ever need to pivot (asked in brief §6)

RAGAS is not wrong, but know its cost profile: 2 judge round-trips per
attempt, LLM-priced. If judge cost/latency becomes the constraint, the
industry alternative is a **single-call fine-tuned grounding classifier**
(Bespoke MiniCheck-7B, Patronus Lynx, or Vectara HHEM-2.1 — the same family
leadership set aside, but as a *pre-filter*, not the decider): run the cheap
classifier on everything and invoke RAGAS only on borderline scores. That
hybrid is how several eval vendors (Vectara, Galileo, Patronus) structure
their own products. The guardrail's evaluation core is isolated in
`_evaluate()` precisely so a pre-filter stage can be added without touching
the contract, retry, or streaming machinery.
