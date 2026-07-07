import os
import sys

# Must be set before ragas (GitPython) is imported anywhere.
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
# Ensure event logging is disabled in tests (no DB).
os.environ.pop("DATABASE_URL", None)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "guardrail"))
