# app/routers/stream.py
from __future__ import annotations
import os
import asyncio
import platform
import json
import types
from pathlib import Path
import jwt
import numpy as np
import requests
import cv2, time, threading
from typing import Dict, Tuple
from starlette.concurrency import iterate_in_threadpool
from starlette.responses import StreamingResponse


from fastapi import APIRouter, Query
from app.models.inference import run_inference_on_image, _engine  # 오케스트레이터 및 엔진

_caps: Dict[Tuple[str,int,int], cv2.VideoCapture] = {}
_caps_lock = threading.Lock()
router = APIRouter(prefix="/stream", tags=["stream"])

# ========= 이벤트 전송 =========
EVENT_URL = os.getenv("EVENT_URL", "http://localhost:3002/events")
AI_JWT_SECRET = os.getenv("AI_JWT_SECRET", "changeme")
SESSION = requests.Session()

_AI_JWT = None
_AI_JWT_EXP = 0
_AI_JWT_KEY = None  # (secret, camera_id)을 기억해서 키가 바뀌면 재발급
_SENT_STARTED: set[str] = set()
_SENT_IDENTITY: set[str] = set()

def _get_ai_token(camera_id: str = "cam2") -> str:
    """AI 역할용 JWT를 캐싱해서 사용."""
    global _AI_JWT, _AI_JWT_EXP, _AI_JWT_KEY
    secret = os.getenv("AI_JWT_SECRET", "changeme")
    now = int(time.time())
    key = (secret, camera_id)

    need_new = (
        _AI_JWT is None
        or (_AI_JWT_EXP - 30) <= now
        or _AI_JWT_KEY != key
    )

    if need_new:
        payload = {
            "sub": "ai",
            "role": "ai",           # verifyAI가 소문자 'ai' 요구 → 확실히 소문자로
            "camera_id": camera_id,
            "iat": now,
            "exp": now + 60 * 5,    # 캐시/검증 문제 줄이려 5분으로 단축 (원하면 30분으로)
        }
        tok = jwt.encode(payload, secret, algorithm="HS256")
        if isinstance(tok, bytes):
            tok = tok.decode("utf-8")
        _AI_JWT = tok
        _AI_JWT_EXP = payload["exp"]
        _AI_JWT_KEY = key

        print(f"[AI_TOKEN] issued role=ai cam={camera_id} exp={_AI_JWT_EXP} secret_fpr={hash(secret)%100000:05d}")

    return _AI_JWT

    # # 토큰이 없거나 만료 임박(30초 이내)이면 재발급
    # if (not _AI_JWT) or (now > (_AI_JWT_EXP - 30)):
    #     payload = {
    #         "sub": "ai",
    #         "role": "ai",
    #         "camera_id": camera_id,
    #         "iat": now,
    #         "exp": now + 60 * 30,  # 30분 유효 (주석은 10분이었는데 실제값과 맞춤)
    #     }
    #     tok = jwt.encode(payload, AI_JWT_SECRET, algorithm="HS256")
    #     if isinstance(tok, bytes):  # PyJWT v1 대비
    #         tok = tok.decode("utf-8")
    #     _AI_JWT = tok
    #     _AI_JWT_EXP = payload["exp"]

    # return _AI_JWT

def _post_event(payload: dict, camera_id: str = "cam0") -> None:
    base = os.getenv("EVENT_SERVER_URL", "http://localhost:3002")
    url = f"{base.rstrip('/')}/events"

    # 토큰 생성
    now = int(time.time())
    token = jwt.encode(
        {"sub": "ai", "role": "ai", "camera_id": camera_id, "iat": now, "exp": now + 300},
        AI_JWT_SECRET,
        algorithm="HS256",
    )
    if isinstance(token, bytes):
        token = token.decode("utf-8")

    body = dict(payload)
    body.setdefault("camera_id", camera_id)
    body.setdefault("at", now)

    # 헤더 구성
    headers = {
        "Authorization": f"Bearer {token}",
        "X-AI-Token": token,
        "Content-Type": "application/json",
    }

    # 🔍 디버그용 로그 추가 — 실제 어떤 토큰/URL로 보내는지 확인
    print(f"[POST_EVENT] → {url}")
    print(f"[POST_EVENT] headers.Authorization = Bearer {token[:40]}...")  # 앞부분만
    print(f"[POST_EVENT] payload = {body}")

    try:
        r = SESSION.post(url, json=body, headers=headers, timeout=5)
        if r.status_code == 401:
            print("[POST_EVENT] ⚠️ 401 Unauthorized — retrying without X-AI-Token header...")
            alt_headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            r = SESSION.post(url, json=body, headers=alt_headers, timeout=5)

        if 200 <= r.status_code < 300:
            print(f"[AI→EVENT] ✅ {r.status_code} {body.get('type')}")
        else:
            print(f"[AI→EVENT] ❌ {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"[AI→EVENT] EXC {e.__class__.__name__}: {e}")

# def _post_event(payload: dict, camera_id: str = "cam0") -> None:
#     # .env 키를 너희 이벤트 서버 명세에 맞춤 (EVENT_SERVER_URL 사용)
#     base = os.getenv("EVENT_SERVER_URL", "http://localhost:3002")
#     url = f"{base.rstrip('/')}/events"

#     # 토큰은 HS256, role=ai, camera_id 포함
#     now = int(time.time())
#     token = jwt.encode(
#         {"role": "ai", "camera_id": camera_id, "iat": now, "exp": now + 300},
#         AI_JWT_SECRET,
#         algorithm="HS256",
#     )
#     if isinstance(token, bytes):
#         token = token.decode("utf-8")

#     body = dict(payload)
#     body.setdefault("camera_id", camera_id)
#     body.setdefault("at", now)

#     # 여러 서버 구현을 모두 만족시키도록 인증을 넉넉하게 넣음
#     headers = {
#         "Authorization": f"Bearer {token}",
#         "X-AI-Token": token,                 # 미들웨어가 이 헤더를 보는 경우 대비
#         "Content-Type": "application/json",
#     }

    try:
        r = SESSION.post(url, json=body, headers=headers, timeout=5)
        if r.status_code == 401:
            # 혹시 Authorization만 허용/불허가 섞인 경우를 대비한 2차 시도
            alt_headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            r = SESSION.post(url, json=body, headers=alt_headers, timeout=5)

        if 200 <= r.status_code < 300:
            print(f"[AI→EVENT] {r.status_code} {body.get('type')}")
        else:
            print(f"[AI→EVENT] {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"[AI→EVENT] EXC {e.__class__.__name__}: {e}")


# def _post_event(payload: dict, camera_id: str = "cam2") -> None:
#     global _AI_JWT, _AI_JWT_EXP
#     try:
#         tok = _get_ai_token(camera_id)
#         headers = {"Authorization": f"Bearer {tok}"}
#         body = dict(payload)
#         body.setdefault("camera_id", camera_id)
#         if "at" not in body and "timestamp" not in body:
#             body["at"] = int(time.time())

#         # 1차 요청
#         r = SESSION.post(EVENT_URL, json=body, headers=headers, timeout=2.0)

#         # 토큰 만료 시 1회만 재시도
#         if r.status_code == 401:
#             _AI_JWT = None
#             _AI_JWT_EXP = 0
#             tok = _get_ai_token(camera_id)
#             headers["Authorization"] = f"Bearer {tok}"
#             r = SESSION.post(EVENT_URL, json=body, headers=headers, timeout=2.0)

#         # 상태별 로그
#         if 200 <= r.status_code < 300:
#             print(f"[AI→EVENT] ✅ {r.status_code} {payload.get('type')}")
#         elif 400 <= r.status_code < 500:
#             print(f"[AI→EVENT] ⚠️ Client {r.status_code}")
#         else:
#             print(f"[AI→EVENT] ⚠️ Server {r.status_code}")

#     except Exception as e:
#         print(f"[AI→EVENT] ❌ Exception: {e.__class__.__name__} {e}")



def _safe_int_pair(t):
    # ('12','34') 같은 문자열 좌표도 안전히 변환
    return (int(float(t[0])), int(float(t[1])))


def draw_seats(frame: np.ndarray, show_debug: bool = True) -> np.ndarray:
    h, w = frame.shape[:2]
    seats = _engine().get_seats() or []

    # 0) 좌표 스케일 결정 (정규화/기준해상도 자동 추정)
    xs, ys = [], []
    for s in seats:
        p1 = s.get("p1") if isinstance(s, dict) else list(getattr(s, "p1"))
        p2 = s.get("p2") if isinstance(s, dict) else list(getattr(s, "p2"))
        xs += [float(p1[0]), float(p2[0])]
        ys += [float(p1[1]), float(p2[1])]

    xmax, ymax = (max(xs or [0.0]), max(ys or [0.0]))
    normalized = (xmax <= 1.01 and ymax <= 1.01)
    if normalized:
        scale_x, scale_y = float(w), float(h)
    else:
        # 픽셀 좌표: 좌석 정의 당시의 기준 해상도 추정 → 프레임보다 큰 값이면 그걸 기준으로 스케일
        base_w = max(float(xmax), float(w)) or 1.0
        base_h = max(float(ymax), float(h)) or 1.0
        scale_x = w / base_w if xmax > w * 1.02 else 1.0
        scale_y = h / base_h if ymax > h * 1.02 else 1.0

    if show_debug:
        cv2.putText(frame, f"{w}x{h}  seats:{len(seats)}",
                    (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    if not seats:
        return frame

    for s in seats:
        # 안전 추출 (기존 그대로)
        if isinstance(s, dict):
            p1 = s.get("p1", [0, 0]); p2 = s.get("p2", [0, 0])
            d_near = float(s.get("d_near", 0)); d_far = float(s.get("d_far", 0))
            inward = 1 if int(s.get("inward_sign", 1)) >= 0 else -1
            seat_id = int(s.get("seat_id", 0))
            ref_w = int(s.get("ref_w", w)); ref_h = int(s.get("ref_h", h))  # 참조 해상도
        else:
            p1 = list(getattr(s, "p1")); p2 = list(getattr(s, "p2"))
            d_near = float(getattr(s, "d_near")); d_far = float(getattr(s, "d_far"))
            inward = 1 if int(getattr(s, "inward_sign")) >= 0 else -1
            seat_id = int(getattr(s, "seat_id", 0))
            # SeatWire 객체에서 참조 해상도 가져오기 (없으면 현재 프레임 해상도를 가정)
            ref_w = int(getattr(s, "ref_w", w))
            ref_h = int(getattr(s, "ref_h", h))

        # 1) 여기만 변경: 좌표 스케일 적용
        x1, y1 = float(p1[0]) * scale_x, float(p1[1]) * scale_y
        x2, y2 = float(p2[0]) * scale_x, float(p2[1]) * scale_y

        # 2) 이하 기존 로직 동일
        ux, uy = (x2 - x1), (y2 - y1)
        L = (ux*ux + uy*uy) ** 0.5
        if L < 1e-6:
            continue
        nx = inward * (-uy / L)
        ny = inward * ( ux / L)

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

        a2 = (int(x1 + nx * d1), int(y1 + ny * d1))
        b2 = (int(x2 + nx * d2), int(y2 + ny * d2))
        poly = np.array([(int(x1), int(y1)), (int(x2), int(y2)), b2, a2], dtype=np.int32)

        cv2.polylines(frame, [poly], True, (0, 255, 255), 2, cv2.LINE_AA)

        midx = int((x1 + x2) * 0.5 + nx * (d1 + 12))
        midy = int((y1 + y2) * 0.5 + ny * (d1 + 12))
        cv2.putText(frame, f"Seat {seat_id}", (midx, midy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2, cv2.LINE_AA)

    return frame


# ========= 카메라 =========
def _open_cap(src: str, width: int = 1280, height: int = 960) -> cv2.VideoCapture:
    """Windows에서 백엔드 백업 + 재시도 + 워밍업까지 포함해서 안정적으로 연다."""
    # 키: (src,width,height)별로 한 번만 오픈
    key = (src, width, height)
    with _caps_lock:
        if key in _caps and _caps[key].isOpened():
            return _caps[key]

        # 새로 시도
        def _try_open(backend=None):
            if src.strip().isdigit():
                idx = int(src)
                cap = cv2.VideoCapture(idx, backend) if backend is not None else cv2.VideoCapture(idx)
            else:
                cap = cv2.VideoCapture(src)
            if not cap or not cap.isOpened():
                return None
            # 해상도/코덱/FPS 설정
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            # 일부 장치에서 MJPG로 바꿔야 해상도/프레임이 안정
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FPS, 30)
            # MSMF에서 버퍼 줄이면 지연 줄어듦(지원 안 하면 무시)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # 워밍업: 초기 프레임 버리기
            for _ in range(8):
                cap.read()
                time.sleep(0.01)
            return cap

        # 백엔드 우선순위: DSHOW → MSMF → 기본
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
        last_err = None
        for be in backends:
            cap = _try_open(be)
            if cap and cap.isOpened():
                _caps[key] = cap
                return cap
            last_err = be

        raise RuntimeError(f"Failed to open camera src={src} backend_tried={last_err}")

def _close_cap_if_unused(src: str, width: int, height: int):
    key = (src, width, height)
    with _caps_lock:
        cap = _caps.pop(key, None)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


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

    # 연속 실패 카운터
    fail_cnt = 0
    FAIL_REOPEN = 10  # 연속 10프레임 실패하면 재오픈

    try:
        while True:
            try:
                ok, frame = cap.read()
                if not ok or frame is None:
                    fail_cnt += 1
                    if fail_cnt >= FAIL_REOPEN:
                        print("[stream] read() consecutive fail -> reopen camera")
                        try:
                            cap.release()
                        except Exception:
                            pass
                        time.sleep(0.2)
                        cap = _open_cap(source, width=width, height=height)
                        fail_cnt = 0
                    else:
                        time.sleep(0.02)
                    continue

                # 정상 읽기
                fail_cnt = 0

                # 1) 추론 (블러/침입)
                cam_id = f"cam{source}" if str(source).strip().isdigit() else str(source)
                try:
                    res = run_inference_on_image(
                        frame,
                        camera_id=cam_id,
                        do_blur=do_blur,
                        do_intrusion=do_intrusion,
                    )
                except Exception as e:
                    print("[stream] run_inference_on_image() failed:", e)
                    res = types.SimpleNamespace()
                    res.frame = frame                      # ← 여기서 인스턴스에 대입
                    res.intrusion_started = False
                    res.intrusion_active = False
                    res.seat_id = None
                    res.meta = {"camera_id": cam_id}
                    res.identity = None
                    res.identity_conf = None
                    res.phone_capture = None

                # 2) ROI 오버레이
                if roi_debug:
                    frame = draw_seats(frame, show_debug=True)

                # 3) 이벤트 전송 (기존 로직 유지)
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
                    _SENT_IDENTITY.add(cid) #중복방지

                if (not res.intrusion_active) and cid:
                    _SENT_STARTED.discard(cid)
                    _SENT_IDENTITY.discard(cid)

            except Exception as e:
                # OpenCV/추론 중 예외 발생 시: 로그 + 재오픈 + 안전 프레임
                print("[stream] inner error:", repr(e))
                try:
                    cap.release()
                except Exception:
                    pass
                time.sleep(0.5)
                cap = _open_cap(source, width=width, height=height)

                if 'frame' in locals() and frame is not None:
                    overlay = frame.copy()
                else:
                    overlay = np.zeros((height, width, 3), dtype=np.uint8)

                h_, w_ = overlay.shape[:2]
                cv2.rectangle(overlay, (0, 0), (w_, 40), (0, 0, 255), -1)
                cv2.putText(overlay, f"inference error: {str(e)[:60]}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                frame = overlay

            # 4) 리사이즈 & JPEG 인코딩
            if scale != 1.0:
                h0, w0 = frame.shape[:2]
                frame = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)

            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
            if not ok:
                continue

            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + jpg.tobytes() + b"\r\n")
            del jpg
            time.sleep(0.03)  # ~30fps
    finally:
        try:
            cap.release()
        except Exception:
            pass



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
    async def _stream():
        try:
            async for chunk in iterate_in_threadpool(
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
                )
            ):
                yield chunk
        except (GeneratorExit, asyncio.CancelledError):
            # 탭 닫힘 등 정상 종료
            pass
        except Exception as e:
            print("[stream/raw] outer error:", repr(e))

    return StreamingResponse(_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

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
    async def _stream():
        try:
            async for chunk in iterate_in_threadpool(
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
                )
            ):
                yield chunk
        except (GeneratorExit, asyncio.CancelledError):
            # 탭 닫힘 등 정상 종료
            pass
        except Exception as e:
            print("[stream/raw] outer error:", repr(e))

    return StreamingResponse(_stream(), media_type="multipart/x-mixed-replace; boundary=frame")