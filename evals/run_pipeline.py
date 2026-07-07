"""Entrypoint for the eval runner.

  python -m evals.run_pipeline          # loop forever (container mode)
  python -m evals.run_pipeline --once   # single cycle (cron / manual / CI)
"""

import argparse
import logging
import sys
import time

from .config import EvalConfig
from .judge import default_metric_factory
from .pipeline import run_once
from .state import ScoreState

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("eval_runner")


def _build_langfuse(cfg: EvalConfig):
    from langfuse import Langfuse

    return Langfuse(
        public_key=cfg.langfuse_public_key,
        secret_key=cfg.langfuse_secret_key,
        host=cfg.langfuse_host,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LiteLLM proxy eval pipeline")
    parser.add_argument("--once", action="store_true",
                        help="run one cycle and exit (cron mode)")
    args = parser.parse_args(argv)

    cfg = EvalConfig.from_env()
    if not cfg.langfuse_public_key or not cfg.langfuse_secret_key:
        logger.error("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set")
        return 2
    if not cfg.proxy_api_key:
        logger.error("EVAL_PROXY_API_KEY not set (use a budgeted virtual key)")
        return 2

    langfuse = _build_langfuse(cfg)
    metric_factory = default_metric_factory(cfg)
    state = ScoreState(cfg.state_path)

    while True:
        try:
            run_once(cfg, langfuse, metric_factory, state)
            langfuse.flush()
        except Exception:
            # A dead Langfuse or proxy must not kill the monitoring loop —
            # log, back off one interval, try again.
            logger.exception("eval cycle failed")
        if args.once:
            return 0
        time.sleep(cfg.interval_seconds)


if __name__ == "__main__":
    sys.exit(main())
