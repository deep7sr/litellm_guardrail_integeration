FROM ghcr.io/berriai/litellm:main-stable

USER root
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY context_contract.py ragas_faithfulness_guardrail.py /app/
