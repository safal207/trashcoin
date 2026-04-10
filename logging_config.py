"""Centralised logging configuration.

Call `setup_logging()` once at application startup. All modules then use
`logging.getLogger(__name__)` to get a pre-configured logger.

Log level is controlled by the LOG_LEVEL environment variable (default INFO).
"""

import logging
import os
import sys


def setup_logging() -> None:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid adding duplicate handlers when modules are reloaded in tests
    if not root.handlers:
        root.addHandler(handler)
    else:
        root.handlers = [handler]
