"""Minimal live dashboard over the ragas_events table (testing/demo use)."""

import os

import asyncpg
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="RAGAS Guardrail Dashboard")
_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        dsn = os.environ["DATABASE_URL"].split("?")[0]
        _pool = await asyncpg.create_pool(dsn, min_size=1, max_size=3)
    return _pool


PAGE = """<!doctype html>
<html><head><title>RAGAS Guardrail Dashboard</title>
<meta http-equiv="refresh" content="5">
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
 table {{ border-collapse: collapse; width: 100%; }}
 th, td {{ border: 1px solid #ccc; padding: 6px 10px; font-size: 14px; text-align: left; }}
 th {{ background: #f0f0f0; }}
 .passed {{ color: #197a2f; }} .blocked, .exhausted {{ color: #b3261e; }}
 .retrying {{ color: #a8660d; }} .unscored, .error {{ color: #666; }}
 .stats span {{ margin-right: 2rem; font-size: 18px; }}
</style></head><body>
<h1>RAGAS Faithfulness Guardrail</h1>
<div class="stats">{stats}</div>
<h2>Recent events</h2>
<table>
<tr><th>Time</th><th>Req</th><th>Attempt</th><th>Score</th><th>Verdict</th><th>Model</th><th>Question</th></tr>
{rows}
</table>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    pool = await get_pool()
    async with pool.acquire() as conn:
        counts = await conn.fetch(
            "SELECT verdict, count(*) AS n FROM ragas_events GROUP BY verdict ORDER BY n DESC"
        )
        events = await conn.fetch(
            "SELECT created_at, req_id, attempt, score, verdict, target_model, question "
            "FROM ragas_events ORDER BY id DESC LIMIT 50"
        )
    stats = " ".join(
        f'<span class="{r["verdict"]}"><b>{r["verdict"]}</b>: {r["n"]}</span>' for r in counts
    ) or "No events yet."
    rows = "\n".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td>"
        '<td class="{}">{}</td><td>{}</td><td>{}</td></tr>'.format(
            r["created_at"].strftime("%H:%M:%S"), r["req_id"], r["attempt"],
            "-" if r["score"] is None else f'{r["score"]:.3f}',
            r["verdict"], r["verdict"], r["target_model"] or "-",
            (r["question"] or "")[:80],
        )
        for r in events
    )
    return PAGE.format(stats=stats, rows=rows)
