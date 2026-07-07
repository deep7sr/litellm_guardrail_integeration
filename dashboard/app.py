"""Live dashboard over the faithfulness_events table — shows the raw user
message, scoring question, retrieved context, generated response, verdict,
score, and (when the reasoning guardrail is active) the judge's explanation
for failures, for every guardrail check."""

import html
import os

import asyncpg
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="Faithfulness Guardrail Dashboard")
_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        dsn = os.environ["DATABASE_URL"].split("?")[0]
        _pool = await asyncpg.create_pool(dsn, min_size=1, max_size=3)
    return _pool


PAGE = """<!doctype html>
<html><head><title>Faithfulness Guardrail Dashboard</title>
<meta http-equiv="refresh" content="5">
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #fafafa; }}
 h1 {{ margin-bottom: 0.2rem; }}
 .stats {{ margin-bottom: 1.5rem; }}
 .stats span {{ margin-right: 2rem; font-size: 18px; }}
 table {{ border-collapse: collapse; width: 100%; background: white; }}
 th, td {{ border: 1px solid #ddd; padding: 8px 10px; font-size: 13px; text-align: left; vertical-align: top; }}
 th {{ background: #f0f0f0; position: sticky; top: 0; }}
 td.wrap {{ max-width: 240px; white-space: pre-wrap; word-break: break-word; }}
 .passed {{ color: #197a2f; font-weight: 600; }}
 .blocked, .exhausted {{ color: #b3261e; font-weight: 600; }}
 .retrying {{ color: #a8660d; font-weight: 600; }}
 .unscored, .error {{ color: #666; font-weight: 600; }}
</style></head><body>
<h1>Faithfulness Guardrail — Live Events</h1>
<div class="stats">{stats}</div>
<table>
<tr>
  <th>Time</th><th>Req</th><th>Attempt</th><th>Score</th><th>Verdict</th><th>Model</th>
  <th>User Message (raw)</th><th>Scoring Question</th><th>Context (retrieved)</th>
  <th>Generated Response</th><th>Failure Reason (judge)</th>
</tr>
{rows}
</table>
</body></html>"""


def esc(s):
    return html.escape(s or "")


@app.get("/", response_class=HTMLResponse)
async def index():
    pool = await get_pool()
    async with pool.acquire() as conn:
        counts = await conn.fetch(
            "SELECT verdict, count(*) AS n FROM faithfulness_events GROUP BY verdict ORDER BY n DESC"
        )
        events = await conn.fetch(
            "SELECT created_at, req_id, attempt, score, verdict, target_model, "
            "user_message, question, context_snippet, answer_snippet, reason "
            "FROM faithfulness_events ORDER BY id DESC LIMIT 50"
        )
    stats = " ".join(
        f'<span class="{r["verdict"]}"><b>{r["verdict"]}</b>: {r["n"]}</span>' for r in counts
    ) or "No events yet."
    rows = "\n".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td>"
        '<td class="{}">{}</td><td>{}</td>'
        '<td class="wrap">{}</td><td class="wrap">{}</td><td class="wrap">{}</td>'
        '<td class="wrap">{}</td><td class="wrap">{}</td></tr>'.format(
            r["created_at"].strftime("%H:%M:%S"), esc(r["req_id"]), r["attempt"],
            "-" if r["score"] is None else f'{r["score"]:.3f}',
            r["verdict"], r["verdict"], esc(r["target_model"]),
            esc(r["user_message"])[:300],
            esc(r["question"])[:300],
            esc(r["context_snippet"])[:400],
            esc(r["answer_snippet"])[:400],
            esc(r["reason"])[:400] or "-",
        )
        for r in events
    )
    return PAGE.format(stats=stats, rows=rows)
