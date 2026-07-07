"""
Golden-dataset evaluation runner for the faithfulness guardrails.

Two modes:

  pipeline  Runs every row through the REAL guardrail hook (real message
            extraction, real multi-turn marker disambiguation, real
            enforcement/retry/block logic, real recursion-guard-safe
            plumbing) with the judge's score replaced by a deterministic
            ORACLE equal to the row's label (grounded->1.0, hallucinated->
            0.0). This needs no network access and verifies the CODE is
            structurally correct: does a grounded answer always pass
            unchanged, and is a hallucinated answer NEVER served as fact
            (blocked or replaced by the fallback), for every row, every
            domain, every answer pattern, single- and multi-turn.
            It does NOT tell you whether the real judge model scores
            accurately — that's what live mode is for.

  live      Runs every row through the REAL RAGAS Faithfulness scorer
            against a running judge model (same class the guardrail uses),
            sweeps the threshold, and reports precision/recall/F1 and a
            per-pattern score distribution. Requires network access to a
            LiteLLM proxy + judge model (run this on the deployment VM).

Usage:
    # Structural correctness (no network needed):
    python scripts/run_golden_eval.py --mode pipeline

    # Real judge accuracy (run on the VM, proxy already up):
    python scripts/run_golden_eval.py --mode live \\
        --base-url http://localhost:4000/v1 --api-key sk-1234 \\
        --judge-model judge-model
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "guardrail"))
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

# The guardrail logs one INFO/WARNING line per scoring attempt; across 64
# rows x 4 configs that's noisy. Quiet it for this batch run — the eval's
# own per-row results are what matter here, not the guardrail's live logs.
logging.getLogger("faithfulness_guardrail").setLevel(logging.ERROR)

import litellm  # noqa: E402
import faithfulness_guardrail as fg  # noqa: E402
import faithfulness_guardrail_reasoning as fgr  # noqa: E402

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "golden_faithfulness_dataset.json")
SYSTEM_PROMPT = "You are a factual assistant. Use ONLY the retrieved context. If the context doesn't contain the answer, say so explicitly."


def load_dataset(path=DATASET_PATH):
    with open(path) as f:
        return json.load(f)


def build_messages(row):
    """Builds the exact OpenAI RAG message structure the guardrail parses,
    including prior turns for multi-turn rows (an earlier evidence message +
    prior answer, both before the fresh evidence + question)."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in row.get("history", []):
        msgs.append({"role": "assistant", "content": fg.EVIDENCE_MARKER + "\n" + "\n\n".join(turn["evidence"])})
        msgs.append({"role": "user", "content": turn["question"]})
        msgs.append({"role": "assistant", "content": turn["answer"]})  # prior model answer, unmarked
    msgs.append({"role": "assistant", "content": fg.EVIDENCE_MARKER + "\n" + "\n\n".join(row["context"])})
    msgs.append({"role": "user", "content": row["question"]})
    return msgs


def make_response(content):
    return litellm.ModelResponse(choices=[{"index": 0, "message": {"role": "assistant", "content": content}}])


# ---------------------------------------------------------------------------
# Pipeline mode
# ---------------------------------------------------------------------------

class StaticRegen:
    """Regeneration stub for pipeline mode. A real generation model given
    corrective feedback might fix its answer; this stub can't (it's a fixed
    test row, not a live model) — it always returns the SAME original
    answer. That is the harder case, not an easier one: it proves that when
    self-correction genuinely fails, the guardrail still never lets the
    hallucinated content through after MAX_RETRIES — it falls back instead."""

    def __init__(self, fixed_answer):
        self.fixed_answer = fixed_answer
        self.calls = 0
        outer = self

        class _Completions:
            async def create(self, **kwargs):
                outer.calls += 1
                return make_response(outer.fixed_answer)

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def oracle_score(label):
    async def _score(question, contexts, answer):
        return 1.0 if label == "grounded" else 0.0
    return _score


async def run_pipeline_row(row, guardrail_cls, on_fail):
    messages = build_messages(row)

    # Structural check, independent of enforcement: did extraction find the
    # right context/question at all (the thing most likely to silently
    # break, especially in multi-turn)? The guardrail returns the evidence
    # message's full text as ONE joined string (matching exactly what was
    # placed in the assistant message), not re-split into the original list
    # of chunks — so the correct comparison is against the joined text, not
    # the row's raw chunk list.
    extracted_ctx = fg._extract_contexts(messages)
    expected_joined = "\n\n".join(row["context"])
    extraction_ok = extracted_ctx == [expected_joined]
    question_ok = fg._raw_user_message(messages) == row["question"]

    fg.DEFAULT_ON_FAIL = on_fail
    fg._score = oracle_score(row["label"])
    fg._proxy_client = StaticRegen(row["answer"])
    fgr._proxy_client = fg._proxy_client  # reason calls (v2) share the stub too

    g = guardrail_cls()
    resp = make_response(row["answer"])
    blocked_detail = None
    try:
        out = await g.async_post_call_success_hook(
            {"model": "agent-model-under-test", "messages": messages}, None, resp)
        final_content = out.choices[0].message.content
        outcome = (
            "passed_unchanged" if final_content == row["answer"]
            else "exhausted_fallback" if final_content == fg.FALLBACK_TEXT or (
                final_content and final_content.startswith(fg.FALLBACK_TEXT))
            else "unexpected:" + repr(final_content)[:80]
        )
    except Exception as e:
        outcome = "blocked"
        blocked_detail = getattr(e, "detail", str(e))

    if row["label"] == "grounded":
        enforcement_correct = outcome == "passed_unchanged"
    else:
        enforcement_correct = outcome in ("blocked", "exhausted_fallback")

    return {
        "id": row["id"], "domain": row["domain"], "pattern": row["pattern"], "label": row["label"],
        "extraction_ok": extraction_ok, "question_ok": question_ok,
        "outcome": outcome, "enforcement_correct": enforcement_correct,
        "blocked_reason": (blocked_detail or {}).get("reason") if isinstance(blocked_detail, dict) else None,
        "extracted_context": extracted_ctx,
    }


async def run_pipeline(dataset):
    configs = [
        (fg.FaithfulnessGuardrail, "retry", "v1_retry"),
        (fg.FaithfulnessGuardrail, "block", "v1_block"),
        (fgr.FaithfulnessReasoningGuardrail, "retry", "v2_retry"),
        (fgr.FaithfulnessReasoningGuardrail, "block", "v2_block"),
    ]
    all_results = {}
    for cls, on_fail, label in configs:
        rows_out = []
        for row in dataset["rows"]:
            rows_out.append(await run_pipeline_row(row, cls, on_fail))
        all_results[label] = rows_out
    return all_results


def summarize_pipeline(all_results):
    summary = {"configs": {}, "extraction": {}}
    # Extraction correctness is config-independent — computed once from any config.
    any_cfg = next(iter(all_results.values()))
    total = len(any_cfg)
    ctx_ok = sum(r["extraction_ok"] for r in any_cfg)
    q_ok = sum(r["question_ok"] for r in any_cfg)
    mt_rows = [r for r in any_cfg if "multiturn" in r["pattern"]]
    mt_ok = sum(r["extraction_ok"] for r in mt_rows)
    summary["extraction"] = {
        "total_rows": total,
        "context_extraction_correct": ctx_ok,
        "question_extraction_correct": q_ok,
        "multiturn_rows": len(mt_rows),
        "multiturn_context_correct": mt_ok,
        "failures": [r["id"] for r in any_cfg if not r["extraction_ok"] or not r["question_ok"]],
    }

    for cfg_name, rows in all_results.items():
        n = len(rows)
        n_correct = sum(r["enforcement_correct"] for r in rows)
        by_domain = defaultdict(lambda: {"total": 0, "correct": 0})
        by_pattern = defaultdict(lambda: {"total": 0, "correct": 0})
        by_label = defaultdict(lambda: {"total": 0, "correct": 0})
        for r in rows:
            for bucket, key in ((by_domain, r["domain"]), (by_pattern, r["pattern"]), (by_label, r["label"])):
                bucket[key]["total"] += 1
                bucket[key]["correct"] += int(r["enforcement_correct"])
        failures = [r for r in rows if not r["enforcement_correct"]]
        summary["configs"][cfg_name] = {
            "total": n, "correct": n_correct,
            "accuracy": round(n_correct / n, 4) if n else None,
            "by_domain": dict(by_domain), "by_pattern": dict(by_pattern), "by_label": dict(by_label),
            "failures": [{"id": r["id"], "outcome": r["outcome"]} for r in failures],
        }
    return summary


# ---------------------------------------------------------------------------
# Live mode (real judge — run on the VM)
# ---------------------------------------------------------------------------

async def run_live(dataset, base_url, api_key, judge_model):
    import numpy as np
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    judge = llm_factory(judge_model, client=client)
    scorer = Faithfulness(llm=judge)

    scored = []
    for row in dataset["rows"]:
        question = row["question"]
        try:
            result = await scorer.ascore(user_input=question, response=row["answer"], retrieved_contexts=row["context"])
            score = 1.0 if np.isnan(result.value) else float(result.value)
        except Exception as e:
            print(f"[{row['id']}] scoring error: {e}", file=sys.stderr)
            score = None
        scored.append({**row, "score": score})
        print(f"[{row['id']}] label={row['label']:12s} pattern={row['pattern']:24s} score={score}")

    valid = [r for r in scored if r["score"] is not None]
    sweep = []
    for t100 in range(30, 96, 5):
        t = t100 / 100
        tp = fp = fn = tn = 0
        for r in valid:
            pred_hallucinated = r["score"] < t
            actual_hallucinated = r["label"] == "hallucinated"
            if pred_hallucinated and actual_hallucinated:
                tp += 1
            elif pred_hallucinated and not actual_hallucinated:
                fp += 1
            elif not pred_hallucinated and actual_hallucinated:
                fn += 1
            else:
                tn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        sweep.append({"threshold": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                      "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)})
    best = max(sweep, key=lambda s: s["f1"]) if sweep else None
    return {"rows": scored, "threshold_sweep": sweep, "recommended_threshold": best}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def render_pipeline_report(summary, dataset_meta, out_path):
    lines = []
    lines.append("# Golden Dataset Evaluation — Pipeline (Structural) Mode\n")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
    lines.append(
        "**What this validates:** the guardrail's real extraction and enforcement code, "
        "run against every row, with the judge's score replaced by an oracle equal to the "
        "row's label. This proves the code is structurally correct — it does **not** measure "
        "the real judge model's scoring accuracy (see the Live Mode section).\n"
    )
    ex = summary["extraction"]
    lines.append("## Extraction correctness (config-independent)\n")
    lines.append(f"- Context extraction correct: **{ex['context_extraction_correct']}/{ex['total_rows']}**")
    lines.append(f"- Question extraction correct: **{ex['question_extraction_correct']}/{ex['total_rows']}**")
    lines.append(f"- Multi-turn rows: {ex['multiturn_rows']}, context correctly disambiguated: **{ex['multiturn_context_correct']}/{ex['multiturn_rows']}**")
    if ex["failures"]:
        lines.append(f"- ⚠️ Extraction failures: {ex['failures']}")
    else:
        lines.append("- No extraction failures.")
    lines.append("")

    lines.append("## Enforcement correctness by configuration\n")
    lines.append("| Config | Total | Correct | Accuracy |")
    lines.append("|---|---|---|---|")
    for cfg_name, cfg in summary["configs"].items():
        lines.append(f"| {cfg_name} | {cfg['total']} | {cfg['correct']} | {cfg['accuracy']*100:.1f}% |")
    lines.append("")

    for cfg_name, cfg in summary["configs"].items():
        lines.append(f"### {cfg_name}\n")
        lines.append("**By domain:**\n")
        lines.append("| Domain | Correct/Total |")
        lines.append("|---|---|")
        for d, v in sorted(cfg["by_domain"].items()):
            lines.append(f"| {d} | {v['correct']}/{v['total']} |")
        lines.append("\n**By answer pattern:**\n")
        lines.append("| Pattern | Correct/Total |")
        lines.append("|---|---|")
        for p, v in sorted(cfg["by_pattern"].items()):
            lines.append(f"| {p} | {v['correct']}/{v['total']} |")
        lines.append("\n**By label:**\n")
        lines.append("| Label | Correct/Total |")
        lines.append("|---|---|")
        for lbl, v in sorted(cfg["by_label"].items()):
            lines.append(f"| {lbl} | {v['correct']}/{v['total']} |")
        if cfg["failures"]:
            lines.append(f"\n**Failures:** {cfg['failures']}")
        lines.append("")

    lines.append("## What this does NOT prove\n")
    lines.append(
        "- Whether the real RAGAS judge model actually scores these answers correctly — "
        "the oracle here is a stand-in equal to the label. Known risk patterns to watch "
        "specifically in live mode: `grounded_negation` and `grounded_refusal` (RAGAS has "
        "historically under-scored correct negative/refusal answers in this project's own "
        "testing).\n"
        "- The 0.7 default threshold — that can only be validated/calibrated in live mode "
        "with real scores (see `run_golden_eval.py --mode live`).\n"
    )
    lines.append("## Next step\n")
    lines.append(
        "Run `python scripts/run_golden_eval.py --mode live --base-url <proxy> --api-key <key> "
        "--judge-model <judge>` on the deployment VM against the real judge model, then review "
        "`reports/golden_eval_live_report.md` for real accuracy, the threshold sweep, and the "
        "per-pattern score distribution — especially the negation/refusal rows.\n"
    )
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def render_live_report(live_result, out_path):
    lines = []
    lines.append("# Golden Dataset Evaluation — Live Mode (Real Judge)\n")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
    lines.append("## Per-row scores\n")
    lines.append("| ID | Domain | Pattern | Label | Score |")
    lines.append("|---|---|---|---|---|")
    for r in live_result["rows"]:
        lines.append(f"| {r['id']} | {r['domain']} | {r['pattern']} | {r['label']} | {r['score']} |")
    lines.append("\n## Threshold sweep\n")
    lines.append("| Threshold | Precision | Recall | F1 | TP | FP | FN | TN |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in live_result["threshold_sweep"]:
        lines.append(f"| {s['threshold']} | {s['precision']} | {s['recall']} | {s['f1']} | {s['tp']} | {s['fp']} | {s['fn']} | {s['tn']} |")
    if live_result["recommended_threshold"]:
        b = live_result["recommended_threshold"]
        lines.append(f"\n**Recommended threshold ≈ {b['threshold']}** (best F1 = {b['f1']}, precision = {b['precision']}, recall = {b['recall']})\n")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pipeline", "live"], default="pipeline")
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--base-url", default="http://localhost:4000/v1")
    parser.add_argument("--api-key", default=os.environ.get("LITELLM_MASTER_KEY", "sk-1234"))
    parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "judge-model"))
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "reports"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dataset = load_dataset(args.dataset)

    if args.mode == "pipeline":
        all_results = asyncio.run(run_pipeline(dataset))
        summary = summarize_pipeline(all_results)
        write_json({"raw": all_results, "summary": summary}, os.path.join(args.out_dir, "golden_eval_pipeline_results.json"))
        render_pipeline_report(summary, dataset, os.path.join(args.out_dir, "golden_eval_pipeline_report.md"))
        print(json.dumps(summary["extraction"], indent=2))
        for cfg_name, cfg in summary["configs"].items():
            print(f"{cfg_name}: {cfg['correct']}/{cfg['total']} ({cfg['accuracy']*100:.1f}%)")
        print(f"\nFull report: {os.path.join(args.out_dir, 'golden_eval_pipeline_report.md')}")
    else:
        result = asyncio.run(run_live(dataset, args.base_url, args.api_key, args.judge_model))
        write_json(result, os.path.join(args.out_dir, "golden_eval_live_results.json"))
        render_live_report(result, os.path.join(args.out_dir, "golden_eval_live_report.md"))
        print(f"\nFull report: {os.path.join(args.out_dir, 'golden_eval_live_report.md')}")


if __name__ == "__main__":
    main()
