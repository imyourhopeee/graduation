# CCTV 프레임이나 DetectionBatch를 입력받아 분석하고 Event-server로 전송
from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np, cv2, logging
from app.models.inference import run_inference_on_image
from app.clients.event_client import post_detections

router = APIRouter(prefix="/analyze", tags=["analyze"])
log = logging.getLogger("analyze")

@router.post("/frame") # 이미지 파일 업로드 → 추론 → Event-server로 푸시
async def analyze_frame(file: UploadFile = File(...)):
    """
    프론트/수집기가 업로드한 프레임을 분석하고,
    결과를 Event-server로 즉시 전송한다.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="image required")

    data = await file.read()
    np_img = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="invalid image")

    batch = run_inference_on_image(img)
    status = await post_detections(batch)
    return {"forwarded_status": status, "count": len(batch.detections)}

@router.post("/push") # 이미 DetectionBatch JSON을 입력받아 Event-server로 전달
async def push_batch(batch: dict):
    """
    (옵션) 외부 추론 파이프라인이 이미 DetectionBatch 형태로 넘겨줄 때.
    그대로 Event-server로 전달.
    """
    # 최소 검증
    if "detections" not in batch or "camera_id" not in batch:
        raise HTTPException(status_code=400, detail="invalid payload")

    # 바로 전송
    from app.schemas.detection import DetectionBatch
    model = DetectionBatch.model_validate(batch)
    status = await post_detections(model)
    return {"forwarded_status": status, "count": len(model.detections)}
