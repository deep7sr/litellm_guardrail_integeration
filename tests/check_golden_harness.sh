#!/usr/bin/env bash
# Self-check for the CI eval gate, run against the deterministic mock stack.
#
# The mock generation model fabricates (marker: XyloBrite) on the two
# color-related golden cases and answers faithfully on the rest. A correct
# harness must therefore fail EXACTLY those two cases. If it fails more,
# scoring is broken; if it fails fewer, the gate is not catching
# hallucinations.
set -uo pipefail

WORKER_PYTHON="${WORKER_PYTHON:-/home/user/venvs/worker/bin/python}"
cd "$(dirname "$0")/.."

OUT="$("$WORKER_PYTHON" -m pytest -q evals/test_golden_dataset.py 2>&1)"
STATUS=$?

echo "$OUT" | tail -5

if [ $STATUS -eq 0 ]; then
  echo "HARNESS SELF-CHECK FAILED: expected the fabrication cases to fail, but everything passed."
  exit 1
fi

for case_id in missing-fact-color injected-fact-in-question; do
  if ! echo "$OUT" | grep -q "FAILED.*${case_id}"; then
    echo "HARNESS SELF-CHECK FAILED: expected case '${case_id}' to be caught."
    exit 1
  fi
done

FAILED_COUNT=$(echo "$OUT" | grep -c '^FAILED')
if [ "$FAILED_COUNT" -ne 2 ]; then
  echo "HARNESS SELF-CHECK FAILED: expected exactly 2 failing cases, got ${FAILED_COUNT}."
  exit 1
fi

echo "HARNESS SELF-CHECK PASSED: gate catches exactly the fabricated cases."
