"""Deterministic OpenAI-compatible mock server for end-to-end pipeline tests.

Two personalities, selected by requested model name:

* ``mock-generator``: a fake RAG answer model. Grounded questions get an
  answer supported by the provided context; questions containing the word
  "color" get an answer that includes a fabricated claim (the marker word
  ``XyloBrite``), which the mock judge will flag as unsupported.

* ``mock-judge``: a fake DeepEval judge. It recognizes DeepEval's
  faithfulness / answer-relevancy prompt stages by their schema keywords and
  returns valid JSON. A claim is judged unsupported iff it contains the
  fabrication marker — so faithfulness scores are exactly computable in the
  tests (fraction of non-fabricated sentences).

Everything is deterministic: same input -> same output, no randomness.
"""

from __future__ import annotations

import ast
import json
import re
import time
import uuid

from fastapi import FastAPI, Request

app = FastAPI()

FABRICATION_MARKER = "xylobrite"

GROUNDED_ANSWER = (
    "The AeroBook X200 battery lasts 12 hours under normal use. "
    "It fully charges in 90 minutes with the included charger."
)
FABRICATED_ANSWER = (
    "The AeroBook X200 battery lasts 12 hours under normal use. "
    "It is available in an exclusive XyloBrite purple color."
)


def _last_user_content(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
    return ""


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _generator_reply(messages: list[dict]) -> str:
    question = _last_user_content(messages).lower()
    if "color" in question:
        return FABRICATED_ANSWER
    return GROUNDED_ANSWER


def _parse_list(blob: str) -> list:
    """DeepEval interpolates lists as Python reprs (single quotes), not JSON."""
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(blob)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            continue
    return _sentences(blob)


def _extract_section(prompt: str, header: str) -> str:
    """Return text between `header` and the trailing 'JSON:' (or end)."""
    idx = prompt.rfind(header)
    if idx == -1:
        return ""
    section = prompt[idx + len(header):]
    end = section.rfind("JSON:")
    if end != -1:
        section = section[:end]
    return section.strip()


def _judge_reply(messages: list[dict]) -> str:
    prompt = "\n".join(
        m.get("content", "") for m in messages if isinstance(m.get("content"), str)
    )

    # Answer-relevancy statements stage.
    if '"statements"' in prompt:
        text = _extract_section(prompt, "Text:")
        return json.dumps({"statements": _sentences(text)})

    # Verdict stages (faithfulness has a Claims: section, relevancy has Statements:).
    if "'verdicts'" in prompt or '"verdicts"' in prompt:
        claims_blob = _extract_section(prompt, "Claims:")
        if claims_blob:
            claims = _parse_list(claims_blob)
            verdicts = []
            for claim in claims:
                if FABRICATION_MARKER in str(claim).lower():
                    verdicts.append(
                        {"verdict": "no", "reason": "This claim is not in the retrieval context."}
                    )
                else:
                    verdicts.append({"verdict": "yes", "reason": None})
            return json.dumps({"verdicts": verdicts})
        statements_blob = _extract_section(prompt, "Statements:")
        statements = _parse_list(statements_blob) if statements_blob else []
        return json.dumps(
            {"verdicts": [{"verdict": "yes", "reason": None} for _ in statements]}
        )

    # Truths stage (faithfulness, from retrieval context).
    if '"truths"' in prompt:
        return json.dumps({"truths": ["The AeroBook X200 battery lasts 12 hours."]})

    # Claims stage (faithfulness, from AI output).
    if '"claims"' in prompt:
        text = _extract_section(prompt, "AI Output:")
        return json.dumps({"claims": _sentences(text)})

    # Reason stage (both metrics).
    if '"reason"' in prompt:
        return json.dumps({"reason": "Deterministic mock judge reason."})

    return json.dumps({"error": "mock judge did not recognize prompt stage"})


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "")
    messages = body.get("messages", [])

    if "judge" in model:
        content = _judge_reply(messages)
    else:
        content = _generator_reply(messages)

    return {
        "id": f"chatcmpl-mock-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=9001, log_level="warning")
