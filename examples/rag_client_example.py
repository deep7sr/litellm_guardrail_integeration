"""
Example RAG application calling the LiteLLM proxy under the message-structure
contract (OpenAI RAG pattern).

The guardrail is transparent to developers: no metadata, no special fields.
The app simply formats its request the standard OpenAI way —

  - system message:    instructions
  - assistant message: the retrieved context, prefixed with the evidence
                       marker (this is OpenAI's own documented format)
  - user message:      the user's question

The guardrail identifies context as the last assistant-role message carrying
the "--- Retrieved Evidence ---" marker. The marker matters in multi-turn
conversations: prior model answers are ALSO assistant-role messages, and the
marker is what tells them apart.
"""

from openai import OpenAI

client = OpenAI(base_url="http://localhost:4000/v1", api_key="sk-1234")

# Pretend these came from your vector store.
retrieved_chunks = [
    "The X100 drone has a flight time of 28 minutes on a full charge.",
    "The X100 is available in matte black only and weighs 249 grams.",
]
question = "How long does the battery last on the X100?"

# OpenAI's documented context formatting — the marker is part of the format.
context_str = "--- Retrieved Evidence ---\n" + "\n\n".join(retrieved_chunks)

response = client.chat.completions.create(
    model="groq-llama-3.1-8b",
    messages=[
        {"role": "system", "content": "You are a factual assistant. Answer using ONLY the retrieved context. If the context does not contain the answer, say so explicitly."},
        {"role": "assistant", "content": context_str},
        {"role": "user", "content": question},
    ],
)

print(response.choices[0].message.content)

# Notes:
# - If the answer fails the faithfulness check, the proxy (in the default
#   "retry" mode) silently regenerates it up to MAX_ATTEMPTS times; if it
#   still fails, a safe fallback message is served instead — with the
#   reasoning guardrail active, the fallback includes the judge's short
#   explanation of what wasn't supported.
# - In "block" mode the request instead fails with a structured 400 whose
#   detail includes the score, threshold, and (reasoning variant) reason.
# - Requests with no evidence-marked assistant message pass through
#   unscored — non-RAG apps need to do nothing at all.
