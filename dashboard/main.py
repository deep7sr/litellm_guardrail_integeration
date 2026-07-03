"""Minimal live dashboard over the ragas_events table (testing/demo tool,
not a production monitoring stack — export the table to your real
observability system for that)."""

import html
import os

import asyncpg
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()
_pool = None


async def pool():
    global _pool
    if _pool is None:
        dsn = os.environ["DATABASE_URL"].split("?")[0]
        _pool = await asyncpg.create_pool(dsn, min_size=1, max_size=3)
    return _pool


@app.get("/", response_class=HTMLResponse)
async def index():
    p = await pool()
    async with p.acquire() as conn:
        try:
            stats = await conn.fetch(
                "SELECT verdict, COUNT(*) AS n FROM ragas_events GROUP BY verdict ORDER BY n DESC"
            )
            events = await conn.fetch(
                """SELECT req_id, created_at, attempt, score, verdict, target_model,
                          question, answer_snippet
                   FROM ragas_events ORDER BY id DESC LIMIT 50"""
            )
        except asyncpg.UndefinedTableError:
            stats, events = [], []

    stat_cells = "".join(
        f"<div class='stat'><div class='n'>{r['n']}</div><div>{html.escape(r['verdict'])}</div></div>"
        for r in stats
    ) or "<p>No events yet.</p>"

    def row(r):
        score = "" if r["score"] is None else f"{r['score']:.3f}"
        return (
            "<tr>"
            f"<td>{html.escape(r['req_id'])}</td>"
            f"<td>{r['created_at']:%H:%M:%S}</td>"
            f"<td>{r['attempt']}</td>"
            f"<td>{score}</td>"
            f"<td class='v-{html.escape(r['verdict'])}'>{html.escape(r['verdict'])}</td>"
            f"<td>{html.escape(r['target_model'] or '')}</td>"
            f"<td>{html.escape((r['question'] or '')[:80])}</td>"
            f"<td>{html.escape((r['answer_snippet'] or '')[:120])}</td>"
            "</tr>"
        )

    rows = "".join(row(r) for r in events)

    return f"""<!doctype html><html><head>
<meta http-equiv="refresh" content="5"><title>RAGAS Guardrail</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
 .stats {{ display: flex; gap: 1rem; margin-bottom: 1.5rem; }}
 .stat {{ border: 1px solid #ccc; border-radius: 8px; padding: .8rem 1.2rem; text-align: center; }}
 .stat .n {{ font-size: 1.6rem; font-weight: 700; }}
 table {{ border-collapse: collapse; width: 100%; font-size: .85rem; }}
 th, td {{ border: 1px solid #ddd; padding: .35rem .5rem; text-align: left; }}
 .v-passed {{ color: #0a7a2f; font-weight: 600; }}
 .v-exhausted, .v-rejected_no_context, .v-scorer_error_blocked, .v-regen_error {{ color: #b00020; font-weight: 600; }}
 .v-retrying {{ color: #b57f00; }}
</style></head><body>
<h1>RAGAS Faithfulness Guardrail — live events</h1>
<div class="stats">{stat_cells}</div>
<table><tr><th>req</th><th>time</th><th>attempt</th><th>score</th><th>verdict</th>
<th>model</th><th>question</th><th>answer</th></tr>{rows}</table>
</body></html>"""
