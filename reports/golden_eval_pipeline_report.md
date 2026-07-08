# Golden Dataset Evaluation — Pipeline (Structural) Mode

Generated: 2026-07-08T14:20:04.312374+00:00

**What this validates:** the guardrail's real extraction and enforcement code, run against every row, with the judge's score replaced by an oracle equal to the row's label. This proves the code is structurally correct — it does **not** measure the real judge model's scoring accuracy (see the Live Mode section).

## Extraction correctness (config-independent)

- Context extraction correct: **64/64**
- Question extraction correct: **64/64**
- Multi-turn rows: 4, context correctly disambiguated: **4/4**
- No extraction failures.

## Enforcement correctness by configuration

| Config | Total | Correct | Accuracy |
|---|---|---|---|
| v1_retry | 64 | 64 | 100.0% |
| v1_block | 64 | 64 | 100.0% |
| v2_retry | 64 | 64 | 100.0% |
| v2_block | 64 | 64 | 100.0% |

### v1_retry

**By domain:**

| Domain | Correct/Total |
|---|---|
| bank | 10/10 |
| drone | 11/11 |
| helpdesk | 11/11 |
| hr | 10/10 |
| insurance | 11/11 |
| saas | 11/11 |

**By answer pattern:**

| Pattern | Correct/Total |
|---|---|
| grounded_multi_fact | 6/6 |
| grounded_multiturn | 2/2 |
| grounded_negation | 6/6 |
| grounded_paraphrase | 6/6 |
| grounded_refusal | 6/6 |
| grounded_verbatim | 6/6 |
| hallucinated_contradiction | 6/6 |
| hallucinated_full | 6/6 |
| hallucinated_multiturn | 2/2 |
| hallucinated_partial | 6/6 |
| hallucinated_specifics | 6/6 |
| hallucinated_wrong_entity | 6/6 |

**By label:**

| Label | Correct/Total |
|---|---|
| grounded | 32/32 |
| hallucinated | 32/32 |

### v1_block

**By domain:**

| Domain | Correct/Total |
|---|---|
| bank | 10/10 |
| drone | 11/11 |
| helpdesk | 11/11 |
| hr | 10/10 |
| insurance | 11/11 |
| saas | 11/11 |

**By answer pattern:**

| Pattern | Correct/Total |
|---|---|
| grounded_multi_fact | 6/6 |
| grounded_multiturn | 2/2 |
| grounded_negation | 6/6 |
| grounded_paraphrase | 6/6 |
| grounded_refusal | 6/6 |
| grounded_verbatim | 6/6 |
| hallucinated_contradiction | 6/6 |
| hallucinated_full | 6/6 |
| hallucinated_multiturn | 2/2 |
| hallucinated_partial | 6/6 |
| hallucinated_specifics | 6/6 |
| hallucinated_wrong_entity | 6/6 |

**By label:**

| Label | Correct/Total |
|---|---|
| grounded | 32/32 |
| hallucinated | 32/32 |

### v2_retry

**By domain:**

| Domain | Correct/Total |
|---|---|
| bank | 10/10 |
| drone | 11/11 |
| helpdesk | 11/11 |
| hr | 10/10 |
| insurance | 11/11 |
| saas | 11/11 |

**By answer pattern:**

| Pattern | Correct/Total |
|---|---|
| grounded_multi_fact | 6/6 |
| grounded_multiturn | 2/2 |
| grounded_negation | 6/6 |
| grounded_paraphrase | 6/6 |
| grounded_refusal | 6/6 |
| grounded_verbatim | 6/6 |
| hallucinated_contradiction | 6/6 |
| hallucinated_full | 6/6 |
| hallucinated_multiturn | 2/2 |
| hallucinated_partial | 6/6 |
| hallucinated_specifics | 6/6 |
| hallucinated_wrong_entity | 6/6 |

**By label:**

| Label | Correct/Total |
|---|---|
| grounded | 32/32 |
| hallucinated | 32/32 |

### v2_block

**By domain:**

| Domain | Correct/Total |
|---|---|
| bank | 10/10 |
| drone | 11/11 |
| helpdesk | 11/11 |
| hr | 10/10 |
| insurance | 11/11 |
| saas | 11/11 |

**By answer pattern:**

| Pattern | Correct/Total |
|---|---|
| grounded_multi_fact | 6/6 |
| grounded_multiturn | 2/2 |
| grounded_negation | 6/6 |
| grounded_paraphrase | 6/6 |
| grounded_refusal | 6/6 |
| grounded_verbatim | 6/6 |
| hallucinated_contradiction | 6/6 |
| hallucinated_full | 6/6 |
| hallucinated_multiturn | 2/2 |
| hallucinated_partial | 6/6 |
| hallucinated_specifics | 6/6 |
| hallucinated_wrong_entity | 6/6 |

**By label:**

| Label | Correct/Total |
|---|---|
| grounded | 32/32 |
| hallucinated | 32/32 |

## What this does NOT prove

- Whether the real RAGAS judge model actually scores these answers correctly — the oracle here is a stand-in equal to the label. Known risk patterns to watch specifically in live mode: `grounded_negation` and `grounded_refusal` (RAGAS has historically under-scored correct negative/refusal answers in this project's own testing).
- The 0.7 default threshold — that can only be validated/calibrated in live mode with real scores (see `run_golden_eval.py --mode live`).

## Next step

Run `python scripts/run_golden_eval.py --mode live --base-url <proxy> --api-key <key> --judge-model <judge>` on the deployment VM against the real judge model, then review `reports/golden_eval_live_report.md` for real accuracy, the threshold sweep, and the per-pattern score distribution — especially the negation/refusal rows.
