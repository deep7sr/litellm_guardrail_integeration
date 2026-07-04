FROM ghcr.io/berriai/litellm:main-stable

USER root
WORKDIR /app

# ragas 0.3.x unconditionally imports ragas.experiment at package load time,
# which imports GitPython, which probes for a `git` binary on PATH and
# raises ImportError if it's missing — even though the guardrail never uses
# git/experiment-tracking functionality. Rather than install a git binary
# just to satisfy an unused feature, silence GitPython's probe.
ENV GIT_PYTHON_REFRESH=quiet

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY context_contract.py ragas_faithfulness_guardrail.py /app/
