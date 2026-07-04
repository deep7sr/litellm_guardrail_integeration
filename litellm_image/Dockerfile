# LiteLLM proxy image with the extra dependencies needed by the RAGAS
# faithfulness guardrail and the Langfuse callback. Replaces the ad-hoc
# `litellm-hhem:latest` image from earlier experimentation.
FROM ghcr.io/berriai/litellm:main-stable

RUN pip install --no-cache-dir \
    "ragas>=0.2" \
    "langfuse>=2,<3" \
    asyncpg \
    numpy
