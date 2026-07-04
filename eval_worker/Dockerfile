FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py .

ENV DEEPEVAL_TELEMETRY_OPT_OUT=YES \
    ERROR_REPORTING=NO \
    PYTHONUNBUFFERED=1

CMD ["python", "worker.py"]
