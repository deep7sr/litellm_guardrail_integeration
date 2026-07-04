"""
Example RAG application calling the LiteLLM proxy under the guardrail contract.

The app is responsible for retrieval; it passes the retrieved chunks and the
originating question explicitly via metadata so the faithfulness guardrail
never has to guess which part of the prompt is context.
"""

from openai import OpenAI

client = OpenAI(base_url="http://localhost:4000/v1", api_key="sk-1234")

# Pretend these came from your vector store.
retrieved_chunks = [
    "The X100 drone has a flight time of 28 minutes on a full charge.",
    "The X100 is available in matte black only and weighs 249 grams.",
]
question = "How long does the battery last on the X100?"

response = client.chat.completions.create(
    model="groq-llama-3.1-8b",
    messages=[
        {"role": "system", "content": "Answer using only the provided context.\n\n"
                                      "Context:\n" + "\n".join(retrieved_chunks)},
        {"role": "user", "content": question},
    ],
    extra_body={
        "metadata": {
            # Required for the response to be faithfulness-scored:
            "guardrail_context": retrieved_chunks,
            # Recommended — the true originating question, independent of
            # however many conversational turns follow:
            "guardrail_question": question,
            # Optional per-request overrides:
            # "guardrail_on_fail": "retry",   # default is "block"
            # "guardrail_threshold": 0.8,
        }
    },
)

print(response.choices[0].message.content)
