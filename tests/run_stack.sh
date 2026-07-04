#!/usr/bin/env bash
# Bring up the local verification stack (mock LLM, Langfuse stub, LiteLLM proxy).
# Usage: PROXY_VENV=/home/user/venvs/proxy ./run_stack.sh /path/to/rundir
set -euo pipefail

RUN_DIR="${1:?usage: run_stack.sh <run_dir>}"
PROXY_VENV="${PROXY_VENV:?set PROXY_VENV}"
TESTS_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$RUN_DIR"

export LANGFUSE_HOST="http://127.0.0.1:9002"
export LANGFUSE_PUBLIC_KEY="pk-lf-test"
export LANGFUSE_SECRET_KEY="sk-lf-test"
# Fast flushing so tests don't wait on the SDK's batching interval.
export LANGFUSE_FLUSH_INTERVAL="1"
export LANGFUSE_FLUSH_AT="1"

"$PROXY_VENV/bin/python" "$TESTS_DIR/mock_llm_server.py" \
  > "$RUN_DIR/mock_llm.log" 2>&1 &
echo $! > "$RUN_DIR/mock_llm.pid"

"$PROXY_VENV/bin/python" "$TESTS_DIR/langfuse_stub.py" \
  > "$RUN_DIR/langfuse_stub.log" 2>&1 &
echo $! > "$RUN_DIR/langfuse_stub.pid"

sleep 2

"$PROXY_VENV/bin/litellm" --config "$TESTS_DIR/litellm_config.test.yaml" \
  --port 4010 --host 127.0.0.1 \
  > "$RUN_DIR/litellm.log" 2>&1 &
echo $! > "$RUN_DIR/litellm.pid"

echo "stack starting in $RUN_DIR (litellm :4010, mock llm :9001, langfuse stub :9002)"
