import logging
import sys
from functools import lru_cache

from pythonjsonlogger.json import JsonFormatter


class _ServiceJsonFormatter(JsonFormatter):
    """Adds a fixed 'service' field to every log record."""

    def add_fields(
        self,
        log_record: dict,
        record: logging.LogRecord,
        message_dict: dict,
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record.setdefault("service", record.name)
        # Rename levelname → level so consumers get a consistent field name
        log_record["level"] = log_record.pop("levelname", record.levelname)


@lru_cache(maxsize=None)
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            _ServiceJsonFormatter(
                fmt="%(asctime)s %(level)s %(name)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
                rename_fields={"asctime": "timestamp"},
            )
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
