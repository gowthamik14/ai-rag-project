import logging
import sys

import uvicorn
from pythonjsonlogger.json import JsonFormatter

from api.app import create_app
from config.settings import settings

app = create_app()


def _configure_root_logging() -> None:
    formatter = JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        rename_fields={"asctime": "timestamp", "levelname": "level"},
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.WARNING)

    logging.getLogger("rag.graph").setLevel(logging.DEBUG)


if __name__ == "__main__":
    _configure_root_logging()
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
