 > **STATUS NOTE (superseded mechanism, principle unchanged).** After team
 > review, the *delivery mechanism* changed: instead of a custom metadata
 > field, apps now pass context via OpenAI's standard RAG message structure —
 > an `assistant`-role message prefixed with `--- Retrieved Evidence ---`
 > (see `docs/INTEGRATION.md`). The *principle* this document argues for is
 > unchanged and still enforced: context is an explicit, app-controlled
 > input, never inferred from (or trusted out of) user-typed text. The
 > evidence below — our confirmed vulnerability and the industry survey —
 > remains the justification for that principle.

# Why the guardrail requires an explicit context contract

*Justification document for the `guardrail_context` / `guardrail_question`
integration requirement. Prepared for team review.*

## Executive summary

Our faithfulness guardrail verifies that every RAG answer is grounded in the
retrieved documents. To verify an answer **against** the context, the
guardrail must know **what the context was**. There are only two ways to get
it:

1. **Guess** — re-parse the request's `messages` array and infer which parts
   are retrieved context (by message role, by labels like "Context:", etc.).
2. **Be told** — require the application to pass the retrieved chunks in an
   explicit, dedicated field, out-of-band from the prompt.

We require option 2. This document shows that option 1 is not merely
inferior but **broken in two independent ways we have already demonstrated
in our own testing**, that every major vendor in this space (AWS, Microsoft,
NVIDIA) imposes the identical requirement, and that the cost to application
teams is two lines of code.

---

## 1. The problem: a prompt is one undifferentiated stream of text

An LLM request is a `messages` array — role-tagged strings. Within a
message, "retrieved context" and "user question" are separated by nothing
but formatting conventions the app author chose: a `Context:` label, an XML
tag, a paragraph break. The proxy (LiteLLM) forwards this array unchanged
and enforces no structure on it. **Any meaning assigned to "this part is
context" is a convention, not a fact of the request.**

The generation model copes with this via reading comprehension — that's how
all RAG works. A *verification* system cannot: it needs ground truth about
what the trusted source material was, and a convention that any caller (or
any end user, via their typed message) can imitate is not ground truth.

## 2. Evidence from our own testing: two confirmed failure modes

These are not hypotheticals. Both occurred in our proof-of-concept.

### Failure A — the guardrail validated a fabrication (security failure)

An earlier version inferred context from message roles, including `user`
messages. A test prompt embedded a **fabricated product detail inside the
user's own message**, phrased as "to confirm what we know so far…". The
guardrail extracted the user's fabrication as trusted context, then scored
the model's fabricated answer **1.0 — perfectly grounded** — because the
answer matched the user's own false claim.

This inverts the guardrail's purpose: a system meant to block fabrications
can be steered into *certifying* them. It is a textbook instance of prompt
injection ([OWASP LLM01, the #1 risk in the OWASP Top 10 for LLM
applications](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)),
whose core mitigation guidance is to **treat all user input and retrieved
content as untrusted and clearly separate trusted from untrusted data** —
which is precisely what role-based inference fails to do, and what the
explicit contract does structurally.

### Failure B — silent non-function for common prompt styles (reliability failure)

After restricting inference to `system`-role messages only (closing Failure
A), a second problem appeared: many real RAG apps pack context and question
into a **single user message** (`"Context: {docs}\n\nQuestion: {q}"`). For
these apps, role-based extraction finds **zero context**, so nearly every
answer scores as ungrounded and the guardrail becomes non-functional for
that application — silently. Multi-turn conversations similarly defeat
"latest user message = question" inference.

### Why we can't just write a smarter heuristic

Every fix moves the problem. Parse `Context:` labels? Users can type
"Context:" in a chat box. Whitelist per-app prompt formats? Now the platform
team maintains a fragile parser per team, which breaks on every prompt
tweak. The root cause is invariant: **the messages array mixes
attacker-influencable text with app-supplied text, and no parser can
distinguish them after the fact.** Only the application knows which strings
came from its document store, because it is the party that retrieved them.
Therefore the application must say so — there is no correct design in which
it doesn't.

## 3. Industry evidence: every major vendor requires exactly this

This is not our invention. Every production groundedness/hallucination
guardrail we surveyed refuses to infer context and requires the caller to
supply it explicitly:

| Vendor / product | Their required field | Source |
|---|---|---|
| **AWS Bedrock Guardrails** — contextual grounding check | Requires three explicit inputs: **grounding source**, **query**, and content to guard. "Any new information introduced in the response will be considered un-grounded." | [AWS documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-contextual-grounding-check.html) |
| **Microsoft Azure AI Content Safety** — groundedness detection | `groundingSources` is a **required** request parameter: an explicit array of source documents to validate against. | [Microsoft Learn](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/quickstart-groundedness), [API reference](https://learn.microsoft.com/en-us/rest/api/contentsafety/text-groundedness-detection-operations/detect-groundedness-options?view=rest-contentsafety-2024-02-15-preview) |
| **NVIDIA NeMo Guardrails** — fact-checking rail | Evidence must be supplied via the `$relevant_chunks` context variable — "extracted automatically if the built-in knowledge base is used, **or provided directly alongside the query**." | [NVIDIA documentation](https://docs.nvidia.com/nemo/guardrails/latest/configure-rails/guardrail-catalog/fact-checking.html) |
| **RAGAS** (the scoring library we use) | The Faithfulness metric's input schema requires `retrieved_contexts` as an **explicit field** — RAGAS itself has no mode that infers context from a prompt. | [RAGAS docs](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/) |
| **Coforge TRUST AI** (our reference architecture) | The orchestrator passes query, context, and response to the guardrail service as **separate fields in a REST call** — context is never fished out of the conversation. | Reference architecture diagrams (internal) |
| **OWASP GenAI Security Project** — LLM01 | Core prompt-injection mitigation: "clearly separate system instructions, user input, and external content"; "segregate external content so untrusted data cannot influence" trusted paths. | [OWASP LLM01](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) |

The pattern is unanimous. If inferring context from prompts were viable,
at least one of AWS, Microsoft, or NVIDIA would offer it. None do.

## 4. What happens if we ship without the contract

Concretely, per stakeholder:

- **End users** can inject "established facts" into their messages and have
  the system certify hallucinations as verified (Failure A). The guardrail
  becomes a false badge of trust — **worse than having no guardrail**,
  because downstream consumers will believe the "verified" stamp.
- **Application teams** whose prompt style doesn't match our heuristic get a
  guardrail that blocks nearly everything or scores everything ungrounded
  (Failure B) — they will disable it, and we lose coverage exactly where we
  claimed to provide it company-wide.
- **The platform team** inherits an unwinnable maintenance burden: a parser
  that must anticipate every team's prompt format forever, where every miss
  is either a security hole or an outage.
- **The company**, in any audit or incident review, must defend a guardrail
  whose trust boundary is "we guess" — against an industry baseline (AWS,
  Microsoft, NVIDIA, OWASP) that says explicit segregation is the standard.

## 5. What the contract costs

Two lines per application, using variables the app already holds:

```python
extra_body={"metadata": {
    "guardrail_context": chunks,     # already in a variable — retrieval returned it
    "guardrail_question": question,  # already in a variable — the user sent it
}}
```

Mitigations that make adoption cheap and measurable:

- **Internal helper library** (`company_llm.guarded_completion`) — teams call
  one function; the contract has a single canonical implementation.
- **No prompt changes** — apps keep building prompts exactly as before; the
  metadata is an out-of-band copy the generation model never sees.
- **Graceful non-adoption** — requests without the field pass through
  unscored (never broken), logged with verdict `unscored`, so adoption is a
  dashboard metric, not an honor system.
- **Graded rollout** — observe (log unscored) → warn (contact teams via
  dashboard data) → enforce per API key once a team confirms integration.
- **Non-RAG apps do nothing** — chat/summarization/codegen traffic is
  correctly and permanently unscored.

## 6. Anticipated objections, answered

**"Can't the proxy just parse the prompt? The context is right there."**
It's there for a *reader*, not for a *parser with a security obligation*.
Anything a parser keys on (roles, labels, position) is plain text an end
user can imitate in their own message — we demonstrated exactly this
(Failure A). AWS, Microsoft, and NVIDIA all had this choice and all
require explicit sources instead.

**"This burdens every team in the company."**
Two lines, via a helper function, using variables their code already has.
Compare with the alternative burden: every team's prompt format must
conform to (and never drift from) a central parsing heuristic — that's a
larger, ongoing, and *silent-failure-prone* coupling.

**"Teams will forget or skip it."**
Then their traffic shows as `unscored` on the dashboard, attributable to
their API key. Nothing breaks; visibility is automatic; enforcement can be
flipped per key when policy requires. Forgetting is detectable — a wrong
heuristic guess is not.

**"Couldn't an LLM classify which parts of the prompt are context?"**
Then the guardrail's trust boundary depends on another LLM reading
adversarial text — the same class of vulnerability, one level down, plus
added cost and latency on every request. Verification infrastructure should
not rest on the mechanism it exists to check.

**"Does anyone else actually do it this way?"**
Yes — everyone. See the table in §3: AWS (`grounding source`, required),
Azure (`groundingSources`, required), NVIDIA (`$relevant_chunks`), RAGAS
(`retrieved_contexts`, required), and the Coforge reference architecture
(explicit fields in the REST call). We found **no** counterexample of a
production guardrail that infers context from raw prompts.

**"What about apps that can't be modified?"**
They pass through unscored — same behavior as today, now visible on a
dashboard. The contract adds a capability; it removes nothing.

## 7. Recommendation

Adopt the explicit context contract as the integration standard for all RAG
applications using the central proxy, ship the helper library alongside it,
and roll out in observe → warn → enforce stages using the `unscored` metric
as the adoption tracker.
