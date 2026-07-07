"""End-to-end check for the Langfuse + async eval pipeline.

Run after the E2E stack is up (see docker-compose.e2e.yml header):

    python3 e2e/run_e2e_check.py

Steps:
  1. send a RAG-style chat completion through the LiteLLM proxy
     (evidence-marker message structure, same contract as production);
  2. wait for the trace to land in Langfuse (proxy success_callback);
  3. wait for the eval runner to score it and push
     deepeval_faithfulness + deepeval_answer_relevancy back to Langfuse.

Exits 0 on success, 1 with diagnostics on timeout. Stdlib only.
"""

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

PROXY = os.environ.get("E2E_PROXY_URL", "http://localhost:4000")
LANGFUSE = os.environ.get("E2E_LANGFUSE_URL", "http://localhost:3000")
MASTER_KEY = os.environ.get("LITELLM_MASTER_KEY", "sk-1234")
LF_PUBLIC = os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-lf-dev-only")
LF_SECRET = os.environ.get("LANGFUSE_SECRET_KEY", "sk-lf-dev-only")
TIMEOUT_S = int(os.environ.get("E2E_TIMEOUT_SECONDS", "300"))

MARKER = "--- Retrieved Evidence ---"
QUESTION = "What is the drone's maximum payload capacity?"
CONTEXT = "The X-500 drone has a maximum payload capacity of 2.5 kg."
EXPECTED_SCORES = {"deepeval_faithfulness", "deepeval_answer_relevancy"}


def _req(url, payload=None, auth_basic=None, auth_bearer=None):
    headers = {"Content-Type": "application/json"}
    if auth_basic:
        headers["Authorization"] = "Basic " + base64.b64encode(
            f"{auth_basic[0]}:{auth_basic[1]}".encode()).decode()
    if auth_bearer:
        headers["Authorization"] = f"Bearer {auth_bearer}"
    data = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(r, timeout=30) as resp:
        return json.loads(resp.read())


def step1_send_rag_request() -> str:
    body = {
        "model": "groq-llama-3.1-8b",
        "messages": [
            {"role": "system", "content": "You are a factual assistant. Use ONLY the retrieved context."},
            {"role": "assistant", "content": f"{MARKER}\n{CONTEXT}"},
            {"role": "user", "content": QUESTION},
        ],
    }
    resp = _req(f"{PROXY}/v1/chat/completions", payload=body, auth_bearer=MASTER_KEY)
    answer = resp["choices"][0]["message"]["content"]
    print(f"[1/3] proxy answered: {answer!r}")
    return answer


def _lf_get(path):
    return _req(f"{LANGFUSE}{path}", auth_basic=(LF_PUBLIC, LF_SECRET))


def step2_wait_for_trace(deadline) -> str:
    while time.time() < deadline:
        try:
            traces = _lf_get("/api/public/traces?limit=20").get("data", [])
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"      langfuse not ready yet ({e}); retrying...")
            traces = []
        for t in traces:
            body = json.dumps(t.get("input") or "")
            # The question text also appears inside judge prompts — require
            # the app request's evidence marker and exclude tagged judge
            # traffic so only the real application trace matches.
            if (QUESTION in body and MARKER in body
                    and "eval-internal" not in (t.get("tags") or [])):
                print(f"[2/3] trace in Langfuse: {t['id']}")
                return t["id"]
        time.sleep(5)
    raise TimeoutError("trace never appeared in Langfuse")


def step3_wait_for_scores(trace_id, deadline) -> dict:
    while time.time() < deadline:
        try:
            scores = _lf_get(f"/api/public/scores?traceId={trace_id}").get("data", [])
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"      score fetch failed ({e}); retrying...")
            scores = []
        found = {s["name"]: s for s in scores
                 if s.get("name") in EXPECTED_SCORES
                 and s.get("traceId") == trace_id}  # server-side filter is advisory
        if EXPECTED_SCORES <= set(found):
            print("[3/3] eval scores on the trace:")
            for name, s in sorted(found.items()):
                print(f"      {name} = {s['value']}  comment={str(s.get('comment'))[:80]!r}")
            return found
        time.sleep(5)
    raise TimeoutError(f"scores never appeared on trace {trace_id}")


def main() -> int:
    deadline = time.time() + TIMEOUT_S
    try:
        step1_send_rag_request()
        trace_id = step2_wait_for_trace(deadline)
        step3_wait_for_scores(trace_id, deadline)
    except Exception as e:
        print(f"E2E FAILED: {e}", file=sys.stderr)
        return 1
    print("E2E PASSED: request -> proxy -> Langfuse trace -> deepeval scores")
    return 0


if __name__ == "__main__":
    sys.exit(main())
