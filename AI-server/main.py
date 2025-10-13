import os, asyncio
if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env") # fastapi가 .env를 읽을 수 있도록 추가

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging import setup_logging
from app.routers import health, analyze, stream, config_seats
from app.models.inference import _engine, InferenceEngine, SeatWire

def create_app() -> FastAPI:
    setup_logging(settings.LOG_LEVEL)
    app = FastAPI(title="AI Server", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:3002",
            "http://127.0.0.1:3002",
        ],
        allow_credentials=True,
        allow_methods=["*"],     # PUT, GET, OPTIONS 모두 허용
        allow_headers=["*"],     # Content-Type 등 헤더 허용
    )

    # ✅ 라우터 등록은 여기서 다
    app.include_router(health.router)
    app.include_router(analyze.router)
    app.include_router(stream.router)
    app.include_router(config_seats.router)

    return app

app = create_app()
