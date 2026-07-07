# E2E stand-in for the production litellm-hhem image (which is built on the
# VM). pip-installed LiteLLM proxy + the Langfuse v2 SDK the callback needs.
FROM python:3.11-slim

# Optional CA bundle for TLS-intercepting proxies (see evals/Dockerfile).
COPY fake_upstream.py ca-bundle.crt* /tmp/build/
RUN if [ -f /tmp/build/ca-bundle.crt ]; then export PIP_CERT=/tmp/build/ca-bundle.crt; fi \
    && pip install --no-cache-dir "litellm[proxy]" "langfuse>=2.54,<3"

EXPOSE 4000
ENTRYPOINT ["litellm"]
