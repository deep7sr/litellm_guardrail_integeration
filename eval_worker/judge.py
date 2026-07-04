"""DeepEval judge model that routes all judge calls through the LiteLLM proxy.

This keeps a single egress point for LLM traffic: when the company later
swaps the judge for a self-hosted model, only the proxy's model_list changes
— the eval worker is untouched.
"""

from __future__ import annotations

import json
import os
import re

from deepeval.models.base_model import DeepEvalBaseLLM
from openai import AsyncOpenAI, OpenAI


def _extract_json_block(text: str) -> str:
    """Judge models sometimes wrap JSON in markdown fences or prose.
    Return the outermost JSON object/array found in the text."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # Fast path: already valid JSON.
    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, ValueError):
        pass
    start_candidates = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not start_candidates:
        raise ValueError(f"no JSON found in judge output: {text[:200]!r}")
    start = min(start_candidates)
    end = max(text.rfind("}"), text.rfind("]"))
    if end <= start:
        raise ValueError(f"unterminated JSON in judge output: {text[:200]!r}")
    return text[start : end + 1]


class ProxyJudge(DeepEvalBaseLLM):
    """OpenAI-compatible judge served by the LiteLLM proxy."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        self.model_name = model or os.environ.get("JUDGE_MODEL", "judge-model")
        self.base_url = base_url or os.environ.get(
            "LITELLM_BASE_URL", "http://localhost:4000/v1"
        )
        self.api_key = api_key or os.environ.get("LITELLM_API_KEY", "sk-1234")
        self.timeout = timeout
        self._client = OpenAI(
            base_url=self.base_url, api_key=self.api_key, timeout=timeout
        )
        self._aclient = AsyncOpenAI(
            base_url=self.base_url, api_key=self.api_key, timeout=timeout
        )

    def load_model(self):
        return None

    def _messages(self, prompt: str, schema) -> list[dict]:
        if schema is not None:
            prompt = (
                prompt
                + "\n\nIMPORTANT: Respond ONLY with valid JSON matching this JSON schema"
                + " (no prose, no markdown fences):\n"
                + json.dumps(schema.model_json_schema())
            )
        return [{"role": "user", "content": prompt}]

    def _parse(self, text: str, schema):
        if schema is None:
            return text
        return schema.model_validate_json(_extract_json_block(text))

    def generate(self, prompt: str, schema=None):
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=self._messages(prompt, schema),
            temperature=0,
        )
        return self._parse(resp.choices[0].message.content or "", schema)

    async def a_generate(self, prompt: str, schema=None):
        resp = await self._aclient.chat.completions.create(
            model=self.model_name,
            messages=self._messages(prompt, schema),
            temperature=0,
        )
        return self._parse(resp.choices[0].message.content or "", schema)

    def get_model_name(self):
        return f"litellm-proxy/{self.model_name}"
