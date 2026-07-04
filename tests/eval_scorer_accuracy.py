"""Scorer accuracy benchmark — labeled (context, answer) pairs with known
ground truth, scored by the REAL judge through the guardrail's own
score_answer(). This measures the two production failure modes directly:

  FALSE BLOCK        a grounded answer scored below threshold (hurts UX)
  MISSED HALLUCINATION a fabricated answer scored at/above threshold (hurts trust)

Run inside the litellm container (real judge, real config):

  docker exec -it litellm /app/.venv/bin/python3 /app/eval_scorer_accuracy.py

Options:
  --dry-run     use a keyword-heuristic fake scorer (validates the harness
                itself without an API key; NOT a real accuracy measurement)
  --sleep N     seconds between cases (default 2, be kind to rate limits)
  --only CAT    run only one category (grounded|refusal|hallucinated|mixed)

Cases marked hard=True are known-difficult for LLM judges (unit conversion,
cross-chunk synthesis). A miss there is calibration information, not
necessarily a code bug — see docs/REVIEW.md §7.
"""

import argparse
import asyncio
import math
import os
import sys
import time

CTX_PRODUCT = [
    "The Aster X200 laptop weighs 1.2 kg and has a 14-inch, 120 Hz display. "
    "Its battery lasts 10 hours under normal use and supports fast charging "
    "via USB-C, reaching 80% in 45 minutes. The X200 ships with a charger "
    "included. Battery replacements are covered by warranty for 2 years. "
    "It has 128 GB of storage, expandable via microSD. "
    "The previous model, the Aster X100, had a 7-hour battery and micro-USB charging."
]

# expect: "pass" (score >= threshold) | "block" (score < threshold) | "nan" (no claims)
CASES = [
    # ---------------- grounded: MUST pass (false block = UX damage) ----------
    dict(id="G1", cat="grounded", expect="pass", hard=False,
         q="How much does the X200 weigh?",
         a="The Aster X200 weighs 1.2 kg."),
    dict(id="G2", cat="grounded", expect="pass", hard=False,
         q="Tell me about the battery.",
         a="The battery lasts around 10 hours in normal use, and it charges quickly over USB-C — about 80% in 45 minutes."),
    dict(id="G3", cat="grounded", expect="pass", hard=True,
         q="How heavy is it?",
         a="It weighs 1200 grams."),  # unit conversion — inference, judges often waver
    dict(id="G4", cat="grounded", expect="pass", hard=False,
         q="How long does the battery last?",
         a="The battery lasts ten hours under normal use."),  # number as words
    dict(id="G5", cat="grounded", expect="pass", hard=True,
         q="How does the X200 compare to the X100?",
         a="The X200's 10-hour battery is a clear improvement over the X100's 7 hours, and it moves from micro-USB to USB-C charging."),  # cross-chunk synthesis
    dict(id="G6", cat="grounded", expect="pass", hard=False,
         q="Does it come with a charger?",
         a="Yes — the X200 ships with a charger included."),
    dict(id="G7", cat="grounded", expect="pass", hard=False,
         q="What are the specs?",
         a="Sure! Here's what the documents say: it weighs 1.2 kg, the display is 14 inches at 120 Hz, and storage is 128 GB expandable via microSD."),
    dict(id="G8", cat="grounded", expect="pass", hard=False,
         q="What about storage?",
         a="According to the provided information, the X200 has 128 GB of storage which can be expanded with a microSD card."),
    dict(id="G9", cat="grounded", expect="pass", hard=False,
         q="Is the warranty good?",
         a="Battery replacements are covered for 2 years under warranty."),
    dict(id="G10", cat="grounded", expect="pass", hard=True,
         q="Did the X100 have USB-C?",
         a="No, the X100 did not have USB-C — it used micro-USB charging."),  # negation, entity-correct

    # ---------------- refusals: MUST pass (no claims to hallucinate) ---------
    dict(id="R1", cat="refusal", expect="nan_or_pass", hard=False,
         q="What colors does the X200 come in?",
         a="The provided context doesn't mention any color options for the X200."),
    dict(id="R2", cat="refusal", expect="nan_or_pass", hard=False,
         q="What is the price?",
         a="I don't have pricing information in the available documents."),

    # ---------------- hallucinated: MUST block (miss = trust damage) ---------
    dict(id="H1", cat="hallucinated", expect="block", hard=False,
         q="What colors does the X200 come in?",
         a="The X200 comes in midnight black, arctic silver, and crimson red."),
    dict(id="H2", cat="hallucinated", expect="block", hard=False,
         q="How long does the battery last?",
         a="The X200's battery lasts 15 hours under normal use."),  # fabricated number
    dict(id="H3", cat="hallucinated", expect="block", hard=False,
         q="Does it come with a charger?",
         a="No, the X200 does not include a charger in the box."),  # meaning reversal
    dict(id="H4", cat="hallucinated", expect="block", hard=False,
         q="What charging does the X200 use?",
         a="The X200 charges via micro-USB."),  # entity swap (X100 spec on X200)
    dict(id="H5", cat="hallucinated", expect="block", hard=True,
         q="How is the battery?",
         a="The X200 has a best-in-class, industry-leading battery that outperforms every competitor."),  # exaggeration
    dict(id="H6", cat="hallucinated", expect="block", hard=False,
         q="What about storage?",
         a="It has 128 GB of storage, expandable via microSD up to 2 TB."),  # true + fabricated detail
    dict(id="H7", cat="hallucinated", expect="block", hard=False,
         q="Why does it charge fast?",
         a="It charges quickly because of its gallium-nitride charging circuit."),  # fabricated cause
    dict(id="H8", cat="hallucinated", expect="block", hard=False,
         q="What's the warranty?",
         a="Battery replacements are covered by a 5-year warranty."),  # wrong duration
    dict(id="H9", cat="hallucinated", expect="block", hard=False,
         q="What OS does it run?",
         a="The X200 runs AsterOS 3.1 with three years of guaranteed updates."),  # invented facts
    dict(id="H10", cat="hallucinated", expect="block", hard=False,
         q="Is it water resistant?",
         a="Yes, the X200 is rated IP68 water resistant."),

    # ---------------- mixed: partial truth, MUST block at 0.7 ----------------
    dict(id="M1", cat="mixed", expect="block", hard=False,
         q="Give me a quick summary.",
         a="The X200 weighs 1.2 kg and comes in matte black."),  # ~0.5
    dict(id="M2", cat="mixed", expect="block", hard=True,
         q="Summarize the X200.",
         a="The X200 weighs 1.2 kg, has a 14-inch display, and comes in matte black."),  # ~0.67, just under 0.7
]


class FakeResult:
    def __init__(self, score, claims=None, error="", duration_ms=1):
        self.score, self.claims, self.error, self.duration_ms = score, claims or [], error, duration_ms


async def fake_score(question, contexts, answer):
    """Keyword heuristic for --dry-run harness validation only."""
    lies = ["black", "silver", "crimson", "15 hours", "does not include", "micro-USB.",
            "best-in-class", "2 TB", "gallium", "5-year", "AsterOS", "IP68"]
    hits = sum(1 for w in lies if w.lower() in answer.lower())
    if "doesn't mention" in answer or "don't have" in answer:
        return FakeResult(float("nan"))
    total = max(2, hits + 1)
    return FakeResult(max(0.0, 1.0 - hits / total))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sleep", type=float, default=2.0)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    if args.dry_run:
        threshold = 0.7
        score_fn = fake_score
        print("DRY RUN — fake scorer, validates harness plumbing only\n")
    else:
        sys.path.insert(0, "/app")
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import ragas_faithfulness_guardrail as g
        threshold = g.PASS_THRESHOLD
        score_fn = g.score_answer
        print(f"LIVE — judge={g.JUDGE_MODEL} @ {g.JUDGE_BASE_URL}, threshold={threshold}\n")

    cases = [c for c in CASES if not args.only or c["cat"] == args.only]
    false_blocks, missed_hallucinations, errors, ok = [], [], [], []

    print(f"{'id':5} {'category':13} {'expect':12} {'score':>7}  outcome")
    print("-" * 62)
    for c in cases:
        t0 = time.monotonic()
        r = await score_fn(c["q"], CTX_PRODUCT, c["a"])
        dt = time.monotonic() - t0
        s = r.score
        if s is None:
            outcome, bucket = f"JUDGE ERROR: {r.error[:60]}", errors
        elif math.isnan(s):
            passed = c["expect"] in ("nan_or_pass",)
            outcome = "no_claims -> served" + ("  OK" if passed else "  ?? expected " + c["expect"])
            bucket = ok if passed else (false_blocks if c["expect"] == "pass" else missed_hallucinations)
        else:
            served = s >= threshold
            if c["expect"] == "block":
                outcome, bucket = (("MISSED HALLUCINATION (served)", missed_hallucinations)
                                   if served else ("blocked  OK", ok))
            else:  # pass / nan_or_pass
                outcome, bucket = (("served  OK", ok) if served
                                   else ("FALSE BLOCK", false_blocks))
        bucket.append((c, r))
        score_str = "  nan" if (s is not None and math.isnan(s)) else ("    -" if s is None else f"{s:.3f}")
        hard = " [hard]" if c["hard"] else ""
        print(f"{c['id']:5} {c['cat']:13} {c['expect']:12} {score_str:>7}  {outcome}{hard}  ({dt:.1f}s)")
        if not args.dry_run:
            await asyncio.sleep(args.sleep)

    n = len(cases)
    print("\n" + "=" * 62)
    print(f"total cases:            {n}")
    print(f"correct:                {len(ok)}  ({100*len(ok)/n:.0f}%)")
    print(f"FALSE BLOCKS (UX):      {len(false_blocks)}")
    print(f"MISSED HALLUCINATIONS:  {len(missed_hallucinations)}")
    print(f"judge errors:           {len(errors)}")

    for label, bucket in [("FALSE BLOCK", false_blocks), ("MISSED HALLUCINATION", missed_hallucinations)]:
        for c, r in bucket:
            print(f"\n--- {label}: {c['id']}{' [hard]' if c['hard'] else ''} ---")
            print(f"  answer: {c['a']}")
            for cl in (r.claims or []):
                mark = "SUPPORTED" if cl["verdict"] else "REJECTED"
                print(f"  [{mark}] {cl['statement']}")
                print(f"            reason: {cl['reason']}")

    # Hard-case misses are calibration info; easy-case misses fail the run.
    easy_misses = [c for c, _ in false_blocks + missed_hallucinations if not c["hard"]]
    sys.exit(1 if (easy_misses or errors) else 0)


if __name__ == "__main__":
    asyncio.run(main())
