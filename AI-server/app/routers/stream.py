from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import cv2, time
import numpy as np
from app.services.inference import run_inference_on_image

router = APIRouter(prefix="/stream", tags=["stream"])

def mjpeg_generator():
    cap = cv2.VideoCapture(0)  # RTSP면 "rtsp://..." 로 변경
    while True:
        ok, frame = cap.read()
        if not ok: time.sleep(0.03); continue
        # TODO: 여기에 실제 추론 + 블러 처리 코드 삽입
        batch = run_inference_on_image(frame)
        # 오버레이 예시(선택): bbox를 그린 뒤 인코딩
        for det in batch.detections:
            x,y,w,h = map(int, det.bbox)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        ok, jpg = cv2.imencode(".jpg", frame)
        if not ok: continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
            pg.tobytes() + b"\r\n")
        time.sleep(0.03)  # ~30fps

@router.get("/mjpeg")
def stream_mjpeg():
    return StreamingResponse(mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame")

