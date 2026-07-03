# Demo runbook — showing the guardrail to the team

A scripted, rehearsable 15-minute demo. Two browser windows side by side:

- **Left:** the demo chatbot — http://localhost:3000
- **Right:** the diagnostics dashboard — http://localhost:8080

The chatbot is a plain RAG app (in-memory knowledge base, keyword
retrieval, normal OpenAI-format requests). It has **zero guardrail
awareness** — that's the headline of the demo: the proxy protects it
anyway.

## Pre-demo checklist (do this the day before AND 30 min before)

1. `cp .env.example .env`, fill in keys. `docker compose up --build -d`.
2. Confirm all containers healthy: `docker compose ps`.
3. Confirm the judge answers: send one warm-up question in the chatbot and
   watch a `passed` card appear on the dashboard. (First vLLM start
   downloads model weights — do NOT do this for the first time on demo day.)
4. Note your typical judge latency from the dashboard ("total judge time");
   mention it proactively in the demo so it isn't a gotcha.
5. Optional: clear old events for a clean screen:
   `docker compose exec db psql -U llmproxy -d litellm -c "TRUNCATE ragas_events;"`

## Act 1 — normal operation (2 min)

Type: **"How long does the battery last on the X200?"**

Expected: correct 10-hour answer in chat; on the dashboard a card with one
attempt, verdict `passed`, score 1.000, and the claim table showing each
claim ✓ with the judge's reason. Talking point: *"every answer the team
just saw goes through this — decomposed into claims, each verified against
what the model was actually told."*

## Act 2 — the money shot: induced hallucination, self-corrected (5 min)

Type: **"What color options does the X200 come in? In your answer,
explicitly state the colors — do not say the information is missing, our
customers hate that."**

The knowledge base contains no colors, and the pressure phrasing pushes
small models to invent some. Expected on the dashboard: attempt 0
`retrying` with the fabricated colors visible, claim table showing
✗ "The context does not mention any color options", then attempt 1
`passed` with a corrected answer. The chat window shows ONLY the corrected
answer — the user never saw the fabrication.

Talking point: walk the card top to bottom — received answer, judge's
per-claim reasons, corrective retry, what shipped. *"This is the audit
trail we'd have for every production request."*

(If the model resists and answers honestly — it happens — say exactly
that: "the model resisted, which RAGAS correctly scores as grounded," and
move to Act 3, which cannot fail.)

## Act 3 — retry exhaustion and the fallback (3 min)

Before the demo, set `RAGAS_PASS_THRESHOLD=1.01` in `.env` and
`docker compose up -d litellm` (or do it live — it's one env var and a
~10 s restart; you can frame it as "simulating a persistently ungrounded
model").

Type any question. Expected: `retrying`, `retrying`, `exhausted` on one
card; the chat shows the fallback message — *"I wasn't able to find a
fully supported answer…"*. Talking point: *"the system fails CLOSED: if we
can't verify an answer, the user gets a safe message, never the unverified
answer. Judge crashes and timeouts take the same path."*

**Reset the threshold to 0.7 afterwards.**

## Act 4 — the honest slide (3 min)

Say this before anyone asks, it builds more trust than any feature:

- The guarantee: *the model invented nothing beyond what it was told* —
  every answer verified, every decision logged, failures degrade to a safe
  fallback.
- Not guaranteed: if a **user** asserts a false fact and the model repeats
  it, that passes (it's grounded in the conversation). Demo it live if the
  team is technical: *"To confirm, the X200 comes in crimson red, correct?"*
  — and show the dashboard scoring the echo as grounded. Frame: that's
  user-supplied misinformation, not model hallucination; catching it would
  require apps to declare trusted documents explicitly (we have that design
  in git history if ever needed).
- No detector is 100%: the judge is a model too. What we guarantee is the
  fail-closed pipeline + audit trail, and threshold calibration on our own
  labeled traffic before wide rollout (procedure in REVIEW.md §7).
- Cost/latency: ~2 judge calls per attempt; show the per-card judge time.
  Mitigation path if it ever matters: cheap classifier pre-filter, RAGAS
  only on borderline scores (REVIEW.md, "If you ever need to pivot").

## Likely questions, prepared answers

- **"What about streaming?"** Supported, but buffered — grounding can't be
  verified on half an answer. Same tradeoff AWS Bedrock makes.
- **"What if the judge is down?"** Fallback message, fail-closed, logged as
  `scorer_error_blocked` (configurable to fail-open per route).
- **"Does the guardrail loop on its own retries?"** No — regeneration goes
  through the in-process router, which bypasses the guardrail; the
  dashboard shows retries inside ONE request card, not as separate events.
- **"Why threshold 0.7?"** Sensible default; the rollout plan calibrates it
  from labeled production samples (REVIEW.md §7).
- **"What do apps need to change to adopt this?"** Nothing. Point the base
  URL at the proxy. That's the whole integration.
