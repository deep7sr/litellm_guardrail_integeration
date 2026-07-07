"""Stdlib-only OpenAI-compatible fake upstream for end-to-end testing the
Langfuse + eval pipeline WITHOUT any real model provider (no API keys, no
egress).

Behavior:
  - /v1/chat/completions with response_format json_schema (DeepEval judge
    calls, forwarded by the proxy): returns JSON synthesized from the
    requested schema — verdict-ish string fields get "yes" so metrics
    compute deterministic passing scores.
  - plain requests (the "application" model): returns a fixed grounded
    answer used by the E2E RAG request.

This is test scaffolding only — never deploy it as a model backend.
"""

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GROUNDED_ANSWER = "The drone's maximum payload capacity is 2.5 kg."


def _resolve(schema: dict, root: dict) -> dict:
    while isinstance(schema, dict) and "$ref" in schema:
        node = root
        for part in schema["$ref"].lstrip("#/").split("/"):
            node = node[part]
        schema = node
    return schema if isinstance(schema, dict) else {}


def fill(schema: dict, root: dict, prop_name: str = ""):
    """Produce a minimal instance conforming to a JSON schema."""
    schema = _resolve(schema, root)
    if "enum" in schema:
        return "yes" if "yes" in schema["enum"] else schema["enum"][0]
    for combo in ("anyOf", "oneOf", "allOf"):
        if combo in schema and schema[combo]:
            return fill(schema[combo][0], root, prop_name)
    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "string")
    if t == "object" or "properties" in schema:
        return {k: fill(v, root, k) for k, v in (schema.get("properties") or {}).items()}
    if t == "array":
        return [fill(schema.get("items", {"type": "string"}), root, prop_name)]
    if t == "string":
        return "yes" if "verdict" in prop_name.lower() else "E2E stub statement."
    if t == "number":
        return 0.9
    if t == "integer":
        return 1
    if t == "boolean":
        return True
    return "E2E stub."


def _schema_from_response_format(rf) -> dict | None:
    if not isinstance(rf, dict):
        return None
    if rf.get("type") == "json_schema":
        js = rf.get("json_schema") or {}
        return js.get("schema") or js
    if rf.get("type") == "json_object":
        return {"type": "object", "properties": {}}
    # Some clients send the schema dict directly.
    if "properties" in rf or "$defs" in rf:
        return rf
    return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter logs
        print(f"[fake-upstream] {self.address_string()} {fmt % args}", flush=True)

    def _send_json(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/v1/models") or self.path == "/health":
            self._send_json(200, {"object": "list",
                                  "data": [{"id": "fake", "object": "model"}]})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if "/chat/completions" not in self.path:
            self._send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "bad json"})
            return

        schema = _schema_from_response_format(req.get("response_format"))
        if schema is not None:
            content = json.dumps(fill(schema, schema))
        else:
            content = GROUNDED_ANSWER

        self._send_json(200, {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.get("model", "fake"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        })


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", 8000), Handler)
    print("[fake-upstream] listening on :8000", flush=True)
    server.serve_forever()
