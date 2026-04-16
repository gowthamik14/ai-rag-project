import logging
import uvicorn
from api.app import create_app
from config.settings import settings

logging.basicConfig(level=logging.WARNING)
logging.getLogger("rag.graph").setLevel(logging.DEBUG)

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
