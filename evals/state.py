"""Idempotency state: which traces have already been scored.

SQLite on a volume — survives restarts, needs no extra service, and keeps
re-runs from double-spending judge calls on the same traces (sampling is
deterministic per trace id, so without this every cycle would re-pick the
same traces inside the lookback window)."""

import sqlite3
import time
from pathlib import Path


class ScoreState:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS scored_traces (
                   trace_id TEXT PRIMARY KEY,
                   scored_at REAL NOT NULL,
                   metrics TEXT NOT NULL
               )"""
        )
        self._conn.commit()

    def is_scored(self, trace_id: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM scored_traces WHERE trace_id = ?", (trace_id,)
        )
        return cur.fetchone() is not None

    def mark_scored(self, trace_id: str, metric_names: list[str]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO scored_traces (trace_id, scored_at, metrics) VALUES (?, ?, ?)",
            (trace_id, time.time(), ",".join(metric_names)),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
