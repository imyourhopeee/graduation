# app/routers/stream.py
from __future__ import annotations
import os
import platform
import time
import json
from pathlib import Path
import jwt
import cv2
import numpy as np
import requests
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from app.models.inference import run_inference_on_image, _engine  # 오케스트레이터 및 엔진

router = APIRouter(prefix="/stream", tags=["stream"])

# ========= 이벤트 전송 =========
EVENT_URL = os.getenv("EVENT_URL", "http://localhost:3002/events")
AI_JWT_SECRET = os.getenv("AI_JWT_SECRET", "changeme")
SESSION = requests.Session()

_AI_JWT = None
_AI_JWT_EXP = 0
_SENT_STARTED: set[str] = set()
_SENT_IDENTITY: set[str] = set()

def _get_ai_token(camera_id: str = "cam2") -> str:
    """AI 역할용 JWT를 캐싱해서 사용."""
    global _AI_JWT, _AI_JWT_EXP
    now = int(time.time())
    # 토큰이 없거나 만료 임박(30초 이내)이면 재발급
    if not _AI_JWT or now > (_AI_JWT_EXP - 30):
        payload = {
            "role": "ai",
            "camera_id": camera_id,  # 원하면 추가 메타
            "iat": now,
            "exp": now + 60 * 30,    # 10분 유효
        }
    tok = jwt.encode(payload, AI_JWT_SECRET, algorithm="HS256")
    if isinstance(tok, bytes):  # PyJWT v1 대비
        tok = tok.decode("utf-8")
    return tok

# def _post_event(payload: dict) -> None:
#     try:
#         SESSION.post(EVENT_URL, json=payload, timeout=1.0)
#     except Exception:
#         # 이벤트 서버 이슈가 있어도 스트리밍은 계속
#         pass
def _post_event(payload: dict, camera_id: str = "cam2") -> None:
    try:
        tok = _get_ai_token(camera_id)
        headers = {"Authorization": f"Bearer {tok}"}
        SESSION.post(EVENT_URL, json=payload, headers=headers, timeout=2.0)
    except Exception:
        pass

def _safe_int_pair(t):
    # ('12','34') 같은 문자열 좌표도 안전히 변환
    return (int(float(t[0])), int(float(t[1])))


def draw_seats(frame: np.ndarray, show_debug: bool = True) -> np.ndarray:
    h, w = frame.shape[:2]
    seats = _engine().get_seats() or []

    if show_debug:
        cv2.putText(frame, f"{w}x{h}  seats:{len(seats)}",
                    (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    if not seats:
        return frame

    for s in seats:
        # 안전 추출
        if isinstance(s, dict):
            p1 = s.get("p1", [0, 0]); p2 = s.get("p2", [0, 0])
            d_near = float(s.get("d_near", 0)); d_far = float(s.get("d_far", 0))
            inward = 1 if int(s.get("inward_sign", 1)) >= 0 else -1
            seat_id = int(s.get("seat_id", 0))
        else:
            p1 = list(getattr(s, "p1")); p2 = list(getattr(s, "p2"))
            d_near = float(getattr(s, "d_near")); d_far = float(getattr(s, "d_far"))
            inward = 1 if int(getattr(s, "inward_sign")) >= 0 else -1
            seat_id = int(getattr(s, "seat_id", 0))

        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])

        # 기준선 수직 단위벡터 (보라와 동일)
        ux, uy = (x2 - x1), (y2 - y1)
        L = (ux*ux + uy*uy) ** 0.5
        if L < 1e-6:
            continue
        nx = inward * (-uy / L)
        ny = inward * ( ux / L)

        # 보라와 동일: y에 따라 d_far↔d_near 선형 보간
        y_near = max(y1, y2)
        y_far  = min(y1, y2)
        if abs(y_near - y_far) < 1e-6:
            d1 = d2 = d_near
        else:
            def depth_at(y):
                t = (y - y_far) / (y_near - y_far)
                t = 0.0 if t < 0 else (1.0 if t > 1 else t)
                return d_far * (1.0 - t) + d_near * t
            d1, d2 = depth_at(y1), depth_at(y2)

        # 기준선(p1,p2) + 각 끝점만 평행이동한 선(p1+dn, p2+dn) → 보라와 동일한 사다리꼴
        a2 = (int(x1 + nx * d1), int(y1 + ny * d1))
        b2 = (int(x2 + nx * d2), int(y2 + ny * d2))
        poly = np.array([(int(x1), int(y1)),
                         (int(x2), int(y2)),
                         b2, a2], dtype=np.int32)

        # 노란 사다리꼴
        cv2.polylines(frame, [poly], True, (0, 255, 255), 2, cv2.LINE_AA)

        # 라벨
        midx = int((x1 + x2) * 0.5 + nx * (d1 + 12))
        midy = int((y1 + y2) * 0.5 + ny * (d1 + 12))
        cv2.putText(frame, f"Seat {seat_id}", (midx, midy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2, cv2.LINE_AA)

    return frame


# ========= 카메라 =========
def _open_cap(src: str, width: int = 1280, height: int = 960):
    if src.strip() == "0":
        if platform.system().lower().startswith("win"):
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(src)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass


    # 🔧 기본 해상도 강제 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        for _ in range(3):
            time.sleep(0.5)
            cap.release()
            if platform.system().lower().startswith("win") and src.strip() == "0":
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(src)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            if cap.isOpened():
                break
    return cap


# ========= 스트리밍 =========
def mjpeg_generator(
    source: str,
    do_blur: bool,
    do_intrusion: bool,
    scale: float,
    quality: int,
    conf: float | None,
    roi_debug: bool,
    width: int = 1280,
    height: int = 960,
):
    cap = _open_cap(source, width=width, height=height)
    if not cap.isOpened():
        while True:
            blk = np.zeros((240, 320, 3), np.uint8)
            cv2.putText(blk, "camera open fail (retrying...)", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            ok, jpg = cv2.imencode(".jpg", blk)
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            time.sleep(1.0)
            cap.release()
            cap = _open_cap(source, width=width, height=height)
            if cap.isOpened():
                break

    try:
        while True:
            try:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("[stream] read() failed, reopening camera")
                    cap.release()
                    time.sleep(0.5)
                    cap = _open_cap(source, width=width, height=height)
                    continue

                # 1) 추론 (블러/침입)
                res = run_inference_on_image(
                    frame,
                    camera_id="cam2",
                    do_blur=do_blur,
                    do_intrusion=do_intrusion,
                )
                frame = res.frame

                # 2) 저장된 좌석 ROI를 '항상' 덧그린다 (여기가 핵심)
                if roi_debug:
                    frame = draw_seats(frame, show_debug=True)

                # 3) 침입 이벤트 전송
                cid = (res.meta or {}).get("correlation_id")
                if res.intrusion_started and cid and cid not in _SENT_STARTED:
                    _post_event({
                        "type": "intrusion_started",
                        "correlation_id": cid,
                        "seat_id": res.seat_id,
                        "camera_id": (res.meta or {}).get("camera_id"),
                        "identity": res.identity,
                        "identity_conf": res.identity_conf,
                        "phone_capture": res.phone_capture,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    })
                    _SENT_STARTED.add(cid)

                if res.intrusion_active and cid and (res.identity is not None) and (cid not in _SENT_IDENTITY):
                    _post_event({
                        "type": "intrusion_identity",
                        "correlation_id": cid,
                        "seat_id": res.seat_id,
                        "camera_id": (res.meta or {}).get("camera_id"),
                        "identity": res.identity,
                        "identity_conf": res.identity_conf,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    })
                    _SENT_IDENTITY.add(cid)

                if (not res.intrusion_active) and cid:
                    _SENT_STARTED.discard(cid)
                    _SENT_IDENTITY.discard(cid)

            # except Exception as e:
            #     # 추가함
            #     print("[stream] inner error:", e)
            #     cap.release()
            #     time.sleep(0.5)
            #     cap = _open_cap(source, width=width, height=height)

            #     # 추론 실패: 빨간 배너만 띄우고 계속 진행
            #     overlay = frame.copy()
            #     h, w = overlay.shape[:2]
            #     cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 255), -1)
            #     cv2.putText(overlay, f"inference error: {str(e)[:60]}",
            #                 (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            #     frame = overlay
            except Exception as e:
                print("[stream] inner error:", e)
                cap.release()
                time.sleep(0.5)
                cap = _open_cap(source, width=width, height=height)

                # 추론 실패 시 표시 프레임 (frame이 None일 수도 있음)
                if 'frame' in locals() and frame is not None:
                    overlay = frame.copy()
                else:
                    overlay = np.zeros((height, width, 3), dtype=np.uint8)

                h, w = overlay.shape[:2]
                cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 255), -1)
                cv2.putText(overlay, f"inference error: {str(e)[:60]}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                frame = overlay

            # 4) 리사이즈 & JPEG 인코딩
            if scale != 1.0:
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_LINEAR)

            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
            if not ok:
                continue

            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            del jpg #추가함
            time.sleep(0.03)  # ~30fps
    finally:
        cap.release()


# ========= 라우트 =========
@router.get("/raw")
def stream_raw(
    cam: str = "0",
    scale: float = Query(1.0, ge=0.25, le=2.0),
    quality: int = Query(80, ge=10, le=95),
    roi: int = Query(1, description="좌석 시각화(1=on, 0=off)"),
    w: int = Query(1280),
    h: int = Query(960),
):
    return StreamingResponse(
    mjpeg_generator(
        source=cam,
        do_blur=False,
        do_intrusion=False,
        scale=scale,
        quality=quality,
        conf=None,
        roi_debug=bool(roi),
        width=w,
        height=h,
    ),
    media_type="multipart/x-mixed-replace; boundary=frame",
)

@router.get("/blur")
def stream_blur(
    cam: str = "0",
    conf: float | None = Query(None, description="override model confidence (0~1)"),
    scale: float = Query(1.0, ge=0.25, le=2.0),
    quality: int = Query(80, ge=10, le=95),
    roi: int = Query(1, description="좌석 시각화(1=on, 0=off)"),
    w: int = Query(1280, description="출력 가로 해상도"),
    h: int = Query(960, description="출력 세로 해상도"),
):
    return StreamingResponse(
            mjpeg_generator(
                source=cam,
                do_blur=True,
                do_intrusion=True,
                scale=scale,
                quality=quality,
                conf=conf,
                roi_debug=bool(roi),
                width=w,
                height=h,
            ),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )
