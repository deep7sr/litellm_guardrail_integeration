"""Demo RAG chatbot for showcasing the faithfulness guardrail.

A deliberately tiny, dependency-light RAG app: an in-memory product
knowledge base, naive keyword retrieval (deterministic — no vector DB to
set up or go wrong in a live demo), chunks pasted into the system prompt,
and completely normal OpenAI-format requests to the LiteLLM proxy.

The point of the demo: this app knows NOTHING about the guardrail — no
special fields, no integration — yet every answer it shows has been
verified (or corrected, or blocked) by the proxy. Run it next to the
diagnostics dashboard (:8080) to show the scoring live.
"""

import html
import os
import uuid

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

PROXY_BASE_URL = os.environ.get("PROXY_BASE_URL", "http://litellm:4000/v1")
PROXY_API_KEY = os.environ.get("PROXY_API_KEY", "sk-1234")
MODEL = os.environ.get("DEMO_MODEL", "groq-llama-3.1-8b")

# ---------------------------------------------------------------------------
# Tiny knowledge base — the ONLY facts the bot is allowed to know.
# Keep this on a slide during the demo so the team can verify answers
# against it by eye.
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = [
    "The Aster X200 battery lasts 10 hours under normal use.",
    "The Aster X200 supports fast charging via USB-C, reaching 80% in 45 minutes.",
    "Aster X200 battery replacements are covered by warranty for 2 years.",
    "The Aster X200 weighs 340 grams.",
    "The Aster X200 has 128 GB of storage, expandable via microSD.",
    "The Aster X100 (previous model) had a 7-hour battery and micro-USB charging.",
    "Aster devices are assembled in Vietnam.",
    "The Aster X200 display is 6.4 inches, 120 Hz.",
]


def retrieve(question: str, top_k: int = 3):
    """Naive keyword-overlap retrieval — deterministic and explainable,
    which is what you want in a live demo."""
    q_words = {w.lower().strip("?.,!") for w in question.split() if len(w) > 2}
    scored = []
    for chunk in KNOWLEDGE_BASE:
        c_words = {w.lower().strip("?.,!") for w in chunk.split()}
        overlap = len(q_words & c_words)
        if overlap:
            scored.append((overlap, chunk))
    scored.sort(key=lambda t: -t[0])
    return [c for _, c in scored[:top_k]] or KNOWLEDGE_BASE[:top_k]


# ---------------------------------------------------------------------------
# Chat state (in-memory, per browser session — demo only)
# ---------------------------------------------------------------------------

SESSIONS: dict = {}

app = FastAPI()


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.post("/chat")
async def chat(req: ChatRequest):
    history = SESSIONS.setdefault(req.session_id, [])
    chunks = retrieve(req.message)

    system_prompt = (
        "You are the Aster support assistant. Answer the customer's question "
        "using the context below.\n\nContext:\n" + "\n".join(f"- {c}" for c in chunks)
    )
    messages = (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": req.message}]
    )

    async with httpx.AsyncClient(timeout=180) as client:
        try:
            r = await client.post(
                f"{PROXY_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {PROXY_API_KEY}"},
                json={"model": MODEL, "messages": messages},
            )
            r.raise_for_status()
            answer = r.json()["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001 — surface any proxy error in the UI
            return JSONResponse({"error": f"proxy error: {e}"}, status_code=502)

    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": answer})
    return {"answer": answer, "retrieved": chunks}


@app.post("/reset")
async def reset(req: ChatRequest):
    SESSIONS.pop(req.session_id, None)
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
async def index():
    kb = "".join(f"<li>{html.escape(c)}</li>" for c in KNOWLEDGE_BASE)
    return f"""<!doctype html><html><head><title>Aster Support — RAG demo</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 0; background:#0d1117; color:#e6edf3;
        display:flex; height:100vh; }}
 #left {{ flex:1; display:flex; flex-direction:column; max-width:760px; margin:0 auto; padding:1rem; }}
 #kb {{ width:340px; border-left:1px solid #30363d; padding:1rem; font-size:.85rem; overflow:auto; }}
 #kb h3 {{ margin-top:0; }} #kb li {{ margin-bottom:.5rem; color:#8b949e; }}
 #log {{ flex:1; overflow-y:auto; padding:.5rem 0; }}
 .msg {{ margin:.5rem 0; padding:.6rem .9rem; border-radius:10px; max-width:85%;
        white-space:pre-wrap; word-break:break-word; }}
 .user {{ background:#1f6feb33; margin-left:auto; }}
 .bot {{ background:#161b22; border:1px solid #30363d; }}
 .retrieved {{ font-size:.75rem; color:#8b949e; margin:.2rem 0 .8rem; }}
 .err {{ color:#f85149; }}
 form {{ display:flex; gap:.5rem; padding:.8rem 0; }}
 input {{ flex:1; padding:.6rem .8rem; border-radius:8px; border:1px solid #30363d;
         background:#161b22; color:#e6edf3; font-size:1rem; }}
 button {{ padding:.6rem 1rem; border-radius:8px; border:none; background:#238636;
          color:white; font-weight:600; cursor:pointer; }}
 .pending {{ color:#8b949e; font-style:italic; }}
 h2 {{ margin:.3rem 0; }} .sub {{ color:#8b949e; font-size:.85rem; margin-bottom:.5rem; }}
</style></head><body>
<div id="left">
  <h2>Aster Support Assistant</h2>
  <div class="sub">Every answer below passed the faithfulness guardrail on the LiteLLM proxy.
    Watch the scoring live on the <b>diagnostics dashboard (:8080)</b>.</div>
  <div id="log"></div>
  <form onsubmit="send(); return false;">
    <input id="box" placeholder="Ask about the Aster X200…" autofocus autocomplete="off">
    <button>Send</button>
  </form>
</div>
<div id="kb"><h3>Knowledge base (all the bot knows)</h3><ul>{kb}</ul></div>
<script>
const sid = crypto.randomUUID();
const log = document.getElementById("log");
function add(cls, text) {{
  const d = document.createElement("div");
  d.className = "msg " + cls; d.textContent = text;
  log.appendChild(d); log.scrollTop = log.scrollHeight; return d;
}}
async function send() {{
  const box = document.getElementById("box");
  const text = box.value.trim(); if (!text) return;
  box.value = ""; add("user", text);
  const pend = add("bot pending", "… generating + guardrail verification …");
  try {{
    const r = await fetch("/chat", {{ method:"POST",
      headers:{{"Content-Type":"application/json"}},
      body: JSON.stringify({{session_id: sid, message: text}}) }});
    const j = await r.json();
    pend.remove();
    if (j.error) {{ add("bot err", j.error); return; }}
    add("bot", j.answer);
    const meta = document.createElement("div");
    meta.className = "retrieved";
    meta.textContent = "retrieved: " + j.retrieved.join(" | ");
    log.appendChild(meta); log.scrollTop = log.scrollHeight;
  }} catch (e) {{ pend.remove(); add("bot err", "request failed: " + e); }}
}}
</script></body></html>"""
