FROM python:3.11-slim
COPY fake_upstream.py /app/fake_upstream.py
EXPOSE 8000
CMD ["python", "/app/fake_upstream.py"]
