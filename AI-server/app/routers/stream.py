# app/routers/stream.py
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
import os, time
import cv2
import numpy as np

from app.models.blur import run_inference_on_image  # 블러 모듈

router = APIRouter(prefix="/stream", tags=["stream"])

# 전역 싱글톤(모델 1회 로드)
_BLUR: BlurEngine | None = None

def get_blur_engine(conf: float | None = None) -> BlurEngine:
    global _BLUR
    if _BLUR is None:
        model_path = os.getenv("BLUR_MODEL", "runs/detect/train11/weights/best.pt")
        default_conf = float(os.getenv("BLUR_CONF", "0.5"))
        _BLUR = BlurEngine(model_path=model_path, conf=default_conf)
    if conf is not None:
        _BLUR.conf = float(conf)  # 요청별 conf 오버라이드 허용(선택)
    return _BLUR

def _open_cap(src: str):
    # Windows 웹캠 안정화
    if src.strip() == "0":
        return cv2.VideoCapture(0, cv2.CAP_DSHOW)
    return cv2.VideoCapture(src)

def mjpeg_generator(source: str, do_blur: bool, scale: float, quality: int, conf: float | None):
    cap = _open_cap(source)
    if not cap.isOpened():
        blk = np.zeros((240, 320, 3), np.uint8)
        cv2.putText(blk, "camera open fail", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        ok, jpg = cv2.imencode(".jpg", blk)
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
        return

    eng = get_blur_engine(conf if do_blur else None)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.03)
                continue
            
            res = run_inference_on_image(frame, camera_id="cam2",
                                         do_blur=do_blur, do_intrusion=do_intrusion)
            frame = res.frame  # 블러 반영된 프레임

            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
            # if do_blur:
            #     frame, _ = eng.process(frame)  # 블러 적용

            # if scale != 1.0:
            #     h, w = frame.shape[:2]
            #     frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

            # ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
            if not ok:
                continue

            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            time.sleep(0.03)  # ~30fps
    finally:
        cap.release()

@router.get("/raw")
def stream_raw(
    cam: str = "0",
    scale: float = Query(1.0, ge=0.25, le=2.0),
    quality: int = Query(80, ge=10, le=95),
):
    return StreamingResponse(
        mjpeg_generator(cam, do_blur=False, scale=scale, quality=quality, conf=None),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

@router.get("/blur")
def stream_blur(
    cam: str = "0",
    conf: float | None = Query(None, description="override model confidence (0~1)"),
    scale: float = Query(1.0, ge=0.25, le=2.0),
    quality: int = Query(80, ge=10, le=95),
):
    return StreamingResponse(
        mjpeg_generator(cam, do_blur=True, scale=scale, quality=quality, conf=conf),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
