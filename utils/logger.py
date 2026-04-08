"""
Structured logging setup.
Call get_logger(__name__) in every module.
"""
from __future__ import annotations

import logging
import sys
from functools import lru_cache

from config.settings import settings


@lru_cache(maxsize=None)
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, settings.log_level, logging.INFO))
    logger.propagate = False
    return logger
