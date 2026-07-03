"""Diagnostic dashboard for the RAGAS faithfulness guardrail.

Shows the complete picture per request: every attempt's answer as received
from the generation model, the grounding context the judge compared it
against, the claim-by-claim verdicts WITH the judge's stated reasons, the
score vs threshold, timings, errors, and the final outcome.

Testing/demo tool — export the ragas_events table to your real
observability stack for production monitoring.
"""

import html
import json
import os
from collections import OrderedDict

import asyncpg
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()
_pool = None

FINAL_VERDICTS = {
    "passed", "exhausted", "no_claims", "skipped_no_context",
    "scorer_error_blocked", "scorer_error_allowed", "regen_error",
}

VERDICT_HELP = {
    "passed": "score ≥ threshold — answer served to the user",
    "retrying": "score below threshold — regenerated with corrective feedback",
    "exhausted": "all retries below threshold — fallback message served",
    "no_claims": "answer contained no checkable claims (e.g. honest refusal) — served",
    "skipped_no_context": "empty conversation, nothing to verify against — served",
    "scorer_error_blocked": "judge failed — fallback served (fail-closed)",
    "scorer_error_allowed": "judge failed — unverified answer served (fail-open configured)",
    "regen_error": "regeneration call failed — fallback served",
}


async def pool():
    global _pool
    if _pool is None:
        dsn = os.environ["DATABASE_URL"].split("?")[0]
        _pool = await asyncpg.create_pool(dsn, min_size=1, max_size=3)
    return _pool


def esc(v) -> str:
    return html.escape(str(v)) if v is not None else ""


def score_bar(score, threshold):
    if score is None:
        return "<span class='muted'>—</span>"
    pct = max(0, min(100, round(score * 100)))
    cls = "ok" if threshold is not None and score >= threshold else "bad"
    return (
        f"<span class='scoreval {cls}'>{score:.3f}</span>"
        f"<span class='bar'><span class='fill {cls}' style='width:{pct}%'></span></span>"
    )


def claims_html(claims_raw):
    if not claims_raw:
        return ""
    claims = json.loads(claims_raw) if isinstance(claims_raw, str) else claims_raw
    if not claims:
        return ""
    supported = sum(1 for c in claims if c.get("verdict"))
    rows = "".join(
        "<tr>"
        f"<td class='cv {'ok' if c.get('verdict') else 'bad'}'>{'✓ supported' if c.get('verdict') else '✗ NOT supported'}</td>"
        f"<td>{esc(c.get('statement'))}</td>"
        f"<td class='muted'>{esc(c.get('reason'))}</td>"
        "</tr>"
        for c in claims
    )
    return f"""
    <details open>
      <summary>Judge verdicts — {supported}/{len(claims)} claims supported</summary>
      <table class='claims'><tr><th>verdict</th><th>claim extracted from the answer</th><th>judge's reason</th></tr>{rows}</table>
    </details>"""


def attempt_html(e):
    verdict = e["verdict"]
    help_text = VERDICT_HELP.get(verdict, "")
    dur = f"{e['duration_ms']} ms judge time" if e["duration_ms"] is not None else ""
    error = f"<div class='error'>error: {esc(e['error'])}</div>" if e["error"] else ""
    answer_label = "answer received from generation model" if e["attempt"] == 0 \
        else f"regenerated answer (retry {e['attempt']})"
    return f"""
    <div class='attempt v-{esc(verdict)}'>
      <div class='attempt-head'>
        <span class='badge v-{esc(verdict)}' title='{esc(help_text)}'>{esc(verdict)}</span>
        <span>attempt {e['attempt']}</span>
        <span class='score'>{score_bar(e['score'], e['threshold'])}</span>
        <span class='muted'>{e['created_at']:%H:%M:%S} · {dur}</span>
      </div>
      <div class='muted small'>{esc(help_text)}</div>
      {error}
      <details {'open' if verdict in ('retrying', 'exhausted') else ''}>
        <summary>{answer_label}</summary>
        <pre>{esc(e['answer_snippet'])}</pre>
      </details>
      {claims_html(e['claims'])}
    </div>"""


def request_html(req_id, events):
    events = sorted(events, key=lambda e: (e["attempt"], e["id"]))
    last = events[-1]
    final = last["verdict"]
    attempts = "".join(attempt_html(e) for e in events)
    context = next((e["context"] for e in events if e["context"]), None)
    context_block = f"""
      <details>
        <summary>grounding context (what the judge compared against)</summary>
        <pre>{esc(context)}</pre>
      </details>""" if context else ""
    total_ms = sum(e["duration_ms"] or 0 for e in events)
    return f"""
    <div class='request v-{esc(final)}'>
      <div class='req-head'>
        <span class='badge v-{esc(final)}'>{esc(final)}</span>
        <b>{esc(last['question'])[:160] or '(no question)'}</b>
        <span class='muted'>req {esc(req_id)} · {esc(last['target_model'])} ·
          {len(events)} attempt(s) · {total_ms} ms total judge time</span>
      </div>
      {context_block}
      {attempts}
    </div>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    p = await pool()
    async with p.acquire() as conn:
        try:
            events = await conn.fetch(
                """SELECT id, req_id, created_at, attempt, score, verdict, target_model,
                          question, answer_snippet, context, claims, error, duration_ms, threshold
                   FROM ragas_events
                   WHERE req_id IN (SELECT req_id FROM ragas_events
                                    GROUP BY req_id ORDER BY MAX(id) DESC LIMIT 30)
                   ORDER BY id DESC"""
            )
            stats = await conn.fetch(
                "SELECT verdict, COUNT(*) AS n FROM ragas_events GROUP BY verdict ORDER BY n DESC"
            )
        except asyncpg.UndefinedTableError:
            events, stats = [], []

    # Group by request, newest request first.
    grouped = OrderedDict()
    for e in events:
        grouped.setdefault(e["req_id"], []).append(e)

    requests_block = "".join(request_html(rid, evs) for rid, evs in grouped.items()) \
        or "<p class='muted'>No events yet — send a request through the proxy.</p>"

    stat_cells = "".join(
        f"<div class='stat'><div class='n'>{r['n']}</div><div class='muted'>{esc(r['verdict'])}</div></div>"
        for r in stats
    )
    return render_page(stat_cells, requests_block)


def render_page(stat_cells: str, requests_block: str) -> str:
    return f"""<!doctype html><html><head>
<meta http-equiv="refresh" content="5"><title>RAGAS Guardrail — Diagnostics</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 1.5rem; background:#0d1117; color:#e6edf3; }}
 .muted {{ color:#8b949e; }} .small {{ font-size:.8rem; }}
 .stats {{ display:flex; gap:.8rem; margin-bottom:1.2rem; flex-wrap:wrap; }}
 .stat {{ border:1px solid #30363d; border-radius:8px; padding:.6rem 1rem; text-align:center; }}
 .stat .n {{ font-size:1.4rem; font-weight:700; }}
 .request {{ border:1px solid #30363d; border-left:4px solid #30363d; border-radius:8px;
             padding:.8rem 1rem; margin-bottom:1rem; }}
 .request.v-passed, .request.v-no_claims {{ border-left-color:#2ea043; }}
 .request.v-exhausted, .request.v-scorer_error_blocked, .request.v-regen_error {{ border-left-color:#f85149; }}
 .req-head {{ display:flex; gap:.8rem; align-items:baseline; flex-wrap:wrap; }}
 .attempt {{ border-top:1px solid #21262d; margin-top:.7rem; padding-top:.7rem; }}
 .attempt-head {{ display:flex; gap:1rem; align-items:center; flex-wrap:wrap; }}
 .badge {{ padding:.15rem .55rem; border-radius:999px; font-size:.78rem; font-weight:600; }}
 .v-passed.badge, .v-no_claims.badge {{ background:#12341a; color:#3fb950; }}
 .v-retrying.badge {{ background:#3a2d10; color:#d29922; }}
 .v-exhausted.badge, .v-scorer_error_blocked.badge, .v-regen_error.badge {{ background:#3d1418; color:#f85149; }}
 .v-skipped_no_context.badge, .v-scorer_error_allowed.badge {{ background:#21262d; color:#8b949e; }}
 .bar {{ display:inline-block; width:110px; height:8px; background:#21262d; border-radius:4px;
         margin-left:.5rem; vertical-align:middle; }}
 .fill {{ display:block; height:100%; border-radius:4px; }}
 .fill.ok {{ background:#2ea043; }} .fill.bad {{ background:#f85149; }}
 .scoreval.ok {{ color:#3fb950; font-weight:700; }} .scoreval.bad {{ color:#f85149; font-weight:700; }}
 pre {{ background:#161b22; border:1px solid #21262d; border-radius:6px; padding:.6rem .8rem;
        white-space:pre-wrap; word-break:break-word; font-size:.82rem; max-height:280px; overflow:auto; }}
 details {{ margin:.4rem 0; }} summary {{ cursor:pointer; color:#58a6ff; font-size:.85rem; }}
 table.claims {{ border-collapse:collapse; width:100%; font-size:.82rem; margin-top:.4rem; }}
 table.claims th, table.claims td {{ border:1px solid #21262d; padding:.35rem .5rem;
        text-align:left; vertical-align:top; }}
 td.cv.ok {{ color:#3fb950; white-space:nowrap; }} td.cv.bad {{ color:#f85149; white-space:nowrap; }}
 .error {{ color:#f85149; font-size:.85rem; margin:.3rem 0; }}
</style></head><body>
<h1>RAGAS Faithfulness Guardrail — Diagnostics</h1>
<p class='muted'>Each card is one user request. Attempts show the answer received from the
generation model, the judge's claim-by-claim verdicts with reasons, and what the guardrail
decided. Threshold: pass requires score ≥ configured threshold (shown per attempt).</p>
<div class="stats">{stat_cells}</div>
{requests_block}
</body></html>"""
