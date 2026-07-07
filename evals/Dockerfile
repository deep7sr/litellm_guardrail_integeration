FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/evals-requirements.txt
RUN pip install --no-cache-dir -r /app/evals-requirements.txt

COPY . /app/evals/

# DeepEval is used strictly as a local library: no telemetry, no
# Confident AI login, results never leave this container.
ENV DEEPEVAL_TELEMETRY_OPT_OUT=YES \
    PYTHONUNBUFFERED=1

CMD ["python", "-m", "evals.run_pipeline"]
