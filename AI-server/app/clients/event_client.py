# 외부 서버와 통신하는 코드를 모아둔 폴더 > Event-server(/api/v1/detections)에 분석 결과를 전송하는 HTTP 클라이언트
import httpx, logging
from app.core.config import settings
from app.schemas.detection import DetectionBatch

log = logging.getLogger("event_client")

async def post_detections(batch: DetectionBatch) -> int:
    url = settings.EVENT_SERVER_URL.rstrip("/") + settings.EVENT_SERVER_INGEST_PATH
    headers = {"Authorization": f"Bearer {settings.EVENT_SERVER_TOKEN}"}
    timeout = httpx.Timeout(settings.REQUEST_TIMEOUT_SEC)

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=batch.model_dump(), headers=headers)
        log.info("POST %s -> %s", url, resp.status_code)
        return resp.status_code
