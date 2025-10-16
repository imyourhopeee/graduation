# app/models/inference.py
from __future__ import annotations
import os, time, uuid, threading, json, cv2, httpx
import numpy as np
import jwt
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

from app.models.intrusion_tripwire import TripwireApp, SeatWire
from app.models.blur import BlurEngine
from deep_sort.face_rec import recognize_identity

OWNER_ID = "yoojin"  # 좌석 주인 라벨

# ---- (1) PhoneBackDetector 로딩 안전화 ----
class _DummyPhoneDetector:
    def scan(self, timeout_sec: float) -> bool:
        return False

def _load_phone_detector():
    try:
        from app.models.photo import PhoneBackDetector as _Real
        print("[PHONE] ✅ using real PhoneBackDetector")
        return _Real()
    except Exception as e:
        print("[PHONE] ⚠️ fallback to dummy:", repr(e))
        return _DummyPhoneDetector()

# ---- (2) 상수 ----
EXIT_SECONDS  = float(os.getenv("EXIT_SECONDS", "3.0"))
DWELL_SECONDS = 10.0
SEATS_JSON_PATH = os.getenv("SEATS_CONFIG", "tripwire_perp.json")
DWELL_JSON_PATH = os.getenv("DWELL_CONFIG", "dwell.json")
EVENT_BASE = os.getenv("EVENT_SERVER_URL", "http://localhost:3002")
EVENT_PATH = os.getenv("EVENT_EVENTS_PATH", "/events")

try:
    from ultralytics import YOLO
    _YOLO_OK = True
except Exception:
    YOLO = None
    _YOLO_OK = False


# ---- (3) 결과 구조체 ----
@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    cls: Optional[int] = None
    score: Optional[float] = None

@dataclass
class InferenceResult:
    frame: np.ndarray
    detections: List[Detection] = field(default_factory=list)
    intrusion_started: bool = False
    intrusion_active: bool = False
    seat_id: Optional[int] = None
    identity: Optional[str] = None
    identity_conf: Optional[float] = None
    phone_capture: Optional[bool] = None
    meta: Dict[str, Any] = field(default_factory=dict)


# ---- (4) 엔진 클래스 ----
class InferenceEngine:
    def __init__(self,
                 blur_model=None, blur_conf=0.5,
                 person_model=None, person_conf=0.4,
                 face_cam_source="0"):
        self._lock = threading.Lock()
        self.blur = BlurEngine(weights=blur_model or os.getenv("BLUR_MODEL", "../runs/detect/train11/weights/best.pt"),
                               conf_thr=float(os.getenv("BLUR_CONF", blur_conf)))

        # 사람 감지용 YOLO
        self.person_model = YOLO(person_model or os.getenv("PERSON_MODEL", "yolov8n.pt")) if _YOLO_OK else None
        self.person_conf = float(os.getenv("PERSON_CONF", person_conf))

        # 침입 감지 엔진
        self.tripwire_app = self._load_from_files()

        # 스마트폰 감지기 (실제 or 더미)
        self.phone = _load_phone_detector()

        # 상태 변수
        self._last_phone_capture = None
        self._phone_busy = False
        self._last_ts = time.time()
        self._corr_id = None
        self._last_identity = None

        # 추가 - 멀티-얼굴 / 오너 보임 세션 플래그
        self._last_identities: list[tuple[str, float]] = []  # [(label, score), ...]
        self._owner_hold = False
        self._active_prev = False

        # 추가 - 얼굴+폰 스레드 실행 회차 카운터
        self._run_seq = 0

        #추가 - YOLO 사람 수 몇 명이었는지 이전 값 저장 (초기화)
        self._prev_persons = None

    def _load_from_files(self) -> TripwireApp:
    # 좌석 정보 로드 (NULL/옛 키 정리)
        seats = []
        p_seats = Path(SEATS_JSON_PATH)
        if p_seats.exists():
            try:
                raw = json.loads(p_seats.read_text(encoding="utf-8"))
                if not isinstance(raw, list):
                    raw = []
                cleaned = []
                for d in raw:
                    if not isinstance(d, dict):
                        continue
                    # 구버전 키 제거
                    for k in ("ref_w", "ref_h", "ref_aspect"):
                        d.pop(k, None)

                    # 좌표 필수
                    p1 = d.get("p1"); p2 = d.get("p2")
                    if not (isinstance(p1, (list, tuple)) and isinstance(p2, (list, tuple)) and len(p1) == 2 and len(p2) == 2):
                        continue

                    # None → 기본값 치환
                    def f_or(v, default):
                        try:
                            return float(v) if v is not None else float(default)
                        except Exception:
                            return float(default)

                    def i_or(v, default):
                        try:
                            return int(v) if v is not None else int(default)
                        except Exception:
                            return int(default)
                        
                    xmax = max(float(p1[0]), float(p2[0]))
                    ymax = max(float(p1[1]), float(p2[1]))
                    normalized = (xmax <= 1.01 and ymax <= 1.01)

                    if normalized:
                        item = {
                            "p1": (float(p1[0]), float(p1[1])),
                            "p2": (float(p2[0]), float(p2[1])),
                            "d_near": f_or(d.get("d_near"), 180.0),
                            "d_far":  f_or(d.get("d_far"), 120.0),
                            "inward_sign": 1 if i_or(d.get("inward_sign"), 1) >= 0 else -1,
                            "seat_id": i_or(d.get("seat_id"), len(cleaned)),
                            "b_near": f_or(d.get("b_near"), 20.0),
                            "b_far":  f_or(d.get("b_far"), 8.0),
                        }
                    else:
                        # [EXISTING L74~L79] ← 이 블록을 통째로 else: 안으로 들여쓰기
                        item = {
                            "p1": (int(float(p1[0])), int(float(p1[1]))),
                            "p2": (int(float(p2[0])), int(float(p2[1]))),
                            "d_near": f_or(d.get("d_near"), 180.0),
                            "d_far":  f_or(d.get("d_far"), 120.0),
                            "inward_sign": 1 if i_or(d.get("inward_sign"), 1) >= 0 else -1,
                            "seat_id": i_or(d.get("seat_id"), len(cleaned)),
                            "b_near": f_or(d.get("b_near"), 20.0),
                            "b_far":  f_or(d.get("b_far"), 8.0),
                        }

                    # Tripwire에 넘길 SeatWire로 변환(정의에 없는 키는 무시)
                    cleaned.append(SeatWire(
                        p1=item["p1"], p2=item["p2"],
                        b_near=item["b_near"], b_far=item["b_far"],
                        inward_sign=item["inward_sign"],
                        d_near=item["d_near"], d_far=item["d_far"],
                        # SeatWire에 seat_id 필드가 없으면 저장은 meta로:
                        # state/dwell_s/exit_s는 런타임에서 갱신
                    ))
                seats = cleaned
            except Exception as e:
                print("[LOAD SEATS FAIL]", e)

        # dwell 시간 로드
        dwell_sec = DWELL_SECONDS
        p_dwell = Path(DWELL_JSON_PATH)
        if p_dwell.exists():
            try:
                data = json.loads(p_dwell.read_text(encoding="utf-8"))
                dwell_sec = float(data.get("seconds", DWELL_SECONDS))
            except Exception as e:
                print("[LOAD DWELL FAIL]", e)

        return TripwireApp(cam=os.getenv("CAMERA_ID", "cam0"), seats=seats, dwell_sec=dwell_sec, on_intrusion=self._post_intrusion_event)

    # --- API용 getter/setter ---
    def get_seats(self): return self.tripwire_app.seats
    def set_seats(self, seats): self.tripwire_app.set_config(seats=seats)
    def get_dwell_time(self): return self.tripwire_app.dwell_sec
    def set_dwell_time(self, dwell_sec): self.tripwire_app.set_config(dwell_sec=dwell_sec)

    # --- YOLO 사람 탐지 ---
    def _detect_persons(self, frame: np.ndarray) -> List[Detection]:
        dets = []
        if self.person_model is None: return dets
        r = self.person_model.predict(frame, verbose=False, conf=self.person_conf, classes=[0])[0]
        if not r or r.boxes is None: return dets
        h, w = frame.shape[:2]
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            x1, x2 = np.clip([x1, x2], 0, w-1)
            y1, y2 = np.clip([y1, y2], 0, h-1)
            if x2 > x1 and y2 > y1:
                dets.append(Detection(bbox=(x1, y1, x2, y2), cls=0, score=float(b.conf)))
        return dets

    def _kick_phone_thread(self, corr_id: str, seat_id: int | None):
        if self._phone_busy:
            # 추가 - 스레드 실행중 로그 
            print("[FACE] ⏩ ing (busy 스레드 실행중)")   
            return
        self._phone_busy = True
        self._run_seq += 1 # 추가 - 스레드 회차 증가
        print(f"[FACE] 🚀 start run={self._run_seq}") # 추가 - 스레드 시작 로그

        def _run(): 
            try: # 얼굴 + 휴대폰 검사 
                # 1) 얼굴 인식 - 웹캠 10초간
                face_timeout = float(os.getenv("FACE_TIMEOUT", "10.0"))
                cam1 = os.getenv("CAM1_SOURCE", "1")

                label = conf = None
                skip_phone = False  # 주인 감지 시 폰 스캔 스킵 플래그 (초기값)

                try: # 함수 호출부
                    face_res = recognize_identity(timeout_sec=face_timeout, cam_source=cam1)
                    if isinstance(face_res, tuple) and len(face_res) >= 2:
                        #반환값은 튜플 형태이므로 정상적으로 이렇게 들어옴
                        label, conf = face_res[0], face_res[1] 
                    elif isinstance(face_res, str): #문자열로 들어왔을 경우 라벨 판정
                        label = face_res
                    # else: None 또는 형식 불일치 -> 그대로 None 유지

                    # ✅ 추가: 멀티-얼굴 리스트 반영 + OWNER 판정 로그
                    identities = []
                    if isinstance(face_res, list):
                        identities = [(str(l or ""), float(c or 0.0)) for (l, c) in face_res if l is not None]
                    elif isinstance(face_res, tuple) and len(face_res) >= 2:
                        identities = [(str(face_res[0] or ""), float(face_res[1] or 0.0))]
                    elif isinstance(face_res, str):
                        identities = [(face_res, 0.0)]

                    with self._lock:
                        self._last_identities = identities

                    has_owner = any((l == OWNER_ID) for (l, _) in identities)
                    if has_owner:
                        self._owner_hold = True
                        print("주인 O")   # 확인용 로그
                    else:
                        print("주인 X")

                    # 주인 보이면 폰 스캔 스킵
                    skip_phone = has_owner
                    

                except Exception as fe:
                    print("[FACE RECOG ERR]", fe)

                with self._lock:
                    self._last_identity = (label, conf) if label else None

                # 2) 휴대폰 후면 감지(옵션)
                timeout = float(os.getenv("PHONE_SCAN_TIMEOUT", "10.0"))
                ok = False
                try:
                    if not skip_phone:
                        ok = bool(self.phone.scan(timeout_sec=timeout))  # False/True/None 방어
                    else:
                        print("[PHONE] skip (owner present)") # 스킵하는 부분
                except Exception as pe:
                    print("[PHONE SCAN ERR]", pe)
                with self._lock:
                    self._last_phone_capture = ok

            except Exception as e:
                print("[PHONE THREAD ERR]", e)
            finally:
                with self._lock:
                    self._phone_busy = False
                    # 추가 - 스레드 종료 로그
                    print(f"[FACE] ✅ end   run={self._run_seq}")


        threading.Thread(target=_run, daemon=True).start()


    # --- 이벤트 전송 ---
    def _post_intrusion_event(self, seat_id, timestamp, event_data=None):
        """
        TripwireApp.update()에서 완결 이벤트(payload dict)를 전달하면 그대로 전송한다.
        event_data가 없을 경우 기존 방식(내부 계산)으로 fallback.
        """
        try:
            # case ① TripwireApp.update()에서 완성된 dict가 온 경우
            if isinstance(event_data, dict):
                payload = dict(event_data)
                with self._lock:
                    ident = self._last_identity
                if ident:
                    label, conf = ident
                    payload["person_id"] = label
                    payload["confidence"] = conf
                    meta = dict(payload.get("meta") or {})
                    meta["user_label"] = label
                    payload["meta"] = meta

            else:
                # case ② 예전처럼 내부에서 조립 (fallback)
                with self._lock:
                    ident = self._last_identity
                user_label = ident[0] if ident else None
                user_conf  = ident[1] if ident else None

                end_ts = timestamp or time.time()
                start_ts = end_ts - self.tripwire_app.dwell_sec
                started_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_ts))
                ended_iso   = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end_ts))

                payload = {
                    "event_type": "intrusion",
                    "seat_id": int(seat_id) if seat_id is not None else None,
                    "camera_id": os.getenv("CAMERA_ID", "cam0"),
                    "person_id": user_label,
                    "confidence": user_conf,
                    "started_at": started_iso,
                    "ended_at": ended_iso,
                    "duration_sec": int(round(end_ts - start_ts)),
                    "meta": {
                        "seat_no": int(seat_id) if seat_id is not None else None,
                        "device_id": os.getenv("CAMERA_ID", "cam0"),
                        "user_label": user_label,
                        "dwell_sec": self.tripwire_app.dwell_sec,
                    },
                }
            # OWNER 보일 시 이벤트 스킵
            if self._owner_hold:
                print("[EVENT] skip (owner present during session)  # 주인 O")
                return
                
            # --- 전송 ---
            url = f"{EVENT_BASE.rstrip('/')}/{EVENT_PATH.lstrip('/')}"
            ai_secret = os.getenv("AI_JWT_SECRET", "changeme")      # ✅ 이벤트서버 .env의 AI_JWT_SECRET과 동일해야 함
            camera_id = os.getenv("CAMERA_ID", "cam0")

            now = int(time.time())
            token_payload = {
                "sub": "ai",
                "role": "ai",            # 이벤트서버 verifyAI가 요구: 소문자 'ai'
                "camera_id": camera_id,
                "iat": now,
                "exp": now + 300,        # 5분
            }

            jwt_token = jwt.encode(token_payload, ai_secret, algorithm="HS256")
            if isinstance(jwt_token, bytes):
                jwt_token = jwt_token.decode("utf-8")

            headers = {
                "Authorization": f"Bearer {jwt_token}",             # ✅ 핵심
                "Content-Type": "application/json",
                # "X-AI-Token": jwt_token,  # (선택) 미들웨어가 읽는다면 보조로
            }

            with httpx.Client(timeout=5.0, headers=headers) as c:
                r = c.post(url, json=payload)
            r.raise_for_status()
            print(f"[EVENT] intrusion logged seat={seat_id} dur={payload.get('duration_sec')}s")
                
        except Exception as e:
            print("[EVENT POST FAIL]", e)


    # --- 메인 추론부 ---
    def process_frame(self, frame: np.ndarray, camera_id="cam2", do_blur=True, do_intrusion=True) -> InferenceResult:
        now = time.time()
        dt = min(0.2, max(0.0, now - self._last_ts))
        self._last_ts = now

        if do_blur:
            frame, _ = self.blur.process(frame)

        started = active = False
        seat_id = None
        person_boxes = []

        if do_intrusion:
            person_boxes = self._detect_persons(frame)
            n = len(person_boxes) #추가 - 감지된 사람 수 저장

            # 추가 - 사람 수가 달라졌을 때만 로그 출력하기
            if n != self._prev_persons:
                print(f"[YOLO] persons: {self._prev_persons} -> {n}")
                self._prev_persons = n

            boxes = [d.bbox for d in person_boxes]
            try:
                ret = self.tripwire_app.update(boxes, dt)
                if not isinstance(ret, tuple) or len(ret) != 3:
                    raise TypeError("Tripwire.update returned invalid")
                started, active, seat_id = ret
            except Exception as e:
                print("[inference] Tripwire.update() failed:", e)
                started, active, seat_id = False, False, None
            if started:
                # 아래 한 줄 추가 - 로그 
                print(f"[TRIP] started seat={seat_id} (dwell={self.tripwire_app.dwell_sec}s) → FACE thread")
                self._corr_id = str(uuid.uuid4())
                self._kick_phone_thread(self._corr_id, seat_id)

        with self._lock:
            phone_flag = self._last_phone_capture
            ident = self._last_identity
        
        # 수정 - 좌석 점유 종료 시 owner hold 해제
        if not active and self._active_prev:
            with self._lock:
                self._owner_hold = False
                self._last_identities = []
                self._last_identity = None
        self._active_prev = bool(active)

        return InferenceResult(
            frame=frame,
            detections=person_boxes,
            intrusion_started=started,
            intrusion_active=active,
            seat_id=seat_id,
            identity=ident[0] if ident else None,
            identity_conf=ident[1] if ident else None,
            phone_capture=phone_flag,
            meta={"camera_id": camera_id, "correlation_id": self._corr_id, "seats": self.tripwire_app.seats},
        )


# --- 진입점 함수 ---
_ENGINE: Optional[InferenceEngine] = None

def _engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = InferenceEngine()
    return _ENGINE

def run_inference_on_image(frame, camera_id="cam2", do_blur=True, do_intrusion=True):
    return _engine().process_frame(frame, camera_id=camera_id, do_blur=do_blur, do_intrusion=do_intrusion)
