"""Minimal Langfuse public-API client (REST, basic auth).

We deliberately use the stable public REST API instead of the Langfuse SDK so
the worker has no SDK-version coupling with the LiteLLM proxy's own Langfuse
integration.

Endpoints used:
  GET  /api/public/traces          (paginated list)
  GET  /api/public/traces/{id}     (detail, includes scores)
  POST /api/public/scores          (attach a score to a trace)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger("eval_worker.langfuse")


class LangfuseClient:
    def __init__(self, host: str, public_key: str, secret_key: str, timeout: float = 30.0):
        self.base_url = host.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            auth=(public_key, secret_key),
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    def list_traces(
        self,
        from_timestamp: datetime | None = None,
        page: int = 1,
        limit: int = 50,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"page": page, "limit": limit, "orderBy": "timestamp.asc"}
        if from_timestamp is not None:
            params["fromTimestamp"] = (
                from_timestamp.astimezone(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )
        resp = self._client.get("/api/public/traces", params=params)
        resp.raise_for_status()
        return resp.json()

    def get_trace(self, trace_id: str) -> dict[str, Any]:
        resp = self._client.get(f"/api/public/traces/{trace_id}")
        resp.raise_for_status()
        return resp.json()

    def create_score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "traceId": trace_id,
            "name": name,
            "value": value,
            "dataType": "NUMERIC",
        }
        if comment:
            payload["comment"] = comment[:2000]
        resp = self._client.post("/api/public/scores", json=payload)
        resp.raise_for_status()
