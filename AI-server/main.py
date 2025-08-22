from fastapi import FastAPI
from app.core.config import settings
from app.core.logging import setup_logging
from app.routers import health, analyze, stream

def create_app() -> FastAPI:
    setup_logging(settings.LOG_LEVEL)
    app = FastAPI(title="AI Server", version="1.0.0")
    app.include_router(health.router)
    app.include_router(analyze.router)
    return app

app = create_app()
app.include_router(stream.router)