FROM python:3.11-slim

WORKDIR /app

# ca-bundle.crt is OPTIONAL (the * glob copies nothing when absent): drop a
# CA bundle next to this Dockerfile when building behind TLS-intercepting
# proxies (office VM corporate TLS, sandboxes). Never commit it.
COPY requirements.txt ca-bundle.crt* /tmp/build/
RUN if [ -f /tmp/build/ca-bundle.crt ]; then export PIP_CERT=/tmp/build/ca-bundle.crt; fi \
    && pip install --no-cache-dir -r /tmp/build/requirements.txt

COPY . /app/evals/

# DeepEval is used strictly as a local library: no telemetry, no
# Confident AI login, results never leave this container.
ENV DEEPEVAL_TELEMETRY_OPT_OUT=YES \
    PYTHONUNBUFFERED=1

CMD ["python", "-m", "evals.run_pipeline"]
