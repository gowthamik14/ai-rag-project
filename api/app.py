from fastapi import FastAPI
from config.settings import settings
from api.routes import authorised_knowledge_router, chat_router, health_router


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version=settings.app_version)
    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(authorised_knowledge_router)
    return app
