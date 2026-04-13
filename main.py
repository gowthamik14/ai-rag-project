import uvicorn
from api.app import create_app
from config.settings import settings

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
