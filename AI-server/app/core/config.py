# .env에서 불러온 환경변수들을 관리하는 설정 모듈.
# Event-server 주소, JWT 토큰, 카메라 ID 같은 설정값을 여기서 관리.
import os
from pydantic import BaseModel

class Settings(BaseModel):
    EVENT_SERVER_URL: str = os.getenv("EVENT_SERVER_URL", "http://localhost:3000")
    EVENT_SERVER_INGEST_PATH: str = os.getenv("EVENT_SERVER_INGEST_PATH", "/api/v1/detections")
    EVENT_SERVER_TOKEN: str = os.getenv("EVENT_SERVER_TOKEN", "dev-token")
    CAMERA_ID: str = os.getenv("CAMERA_ID", "cam-dev")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "*")
    REQUEST_TIMEOUT_SEC: float = float(os.getenv("REQUEST_TIMEOUT_SEC", "5"))

settings = Settings()
