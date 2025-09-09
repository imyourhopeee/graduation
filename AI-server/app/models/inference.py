# 필요한 기능들을 불러오는 통합된 코드

from __future__ import annotations
import os, time, uuid, threading, json
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from app.models.photo import PhoneBackDetector
from app.models.face_rec import FaceRecognizer
from app.models.face_rec_adapter import FaceRecognizer
import cv2
import numpy as np

# 블러 모듈 (네 블러 로직 그대로 들어있는 Visualizer/BlurEngine)
from app.models.blur import BlurEngine

# ====== YOLO 로더 (person 검출 용) ======
try:
    from ultralytics import YOLO
    _YOLO_OK = True
except Exception:
    YOLO = None
    _YOLO_OK = False

# ====== 침입 트립와이어: 네 코드와 동일한 규칙/구조 ======
DWELL_SECONDS = float(os.getenv("DWELL_SECONDS", "8.0"))
EXIT_SECONDS  = float(os.getenv("EXIT_SECONDS", "1.5"))

def feet_point_xyxy(x1, y1, x2, y2) -> Tuple[int, int]:
    return int((x1 + x2) / 2), int(y2)

@dataclass
class SeatWire:
    p1: Tuple[int,int]
    p2: Tuple[int,int]
    b_near: float = 20.0
    b_far:  float = 8.0
    inward_sign: int = +1
    d_near: float = 180.0
    d_far:  float = 120.0
    # 상태/타이머
    state: str = "OUTSIDE"
    dwell_s: float = 0.0
    exit_s: float = 0.0
    seat_id: int = 0

    def y_near_far(self):
        y1, y2 = self.p1[1], self.p2[1]
        return (max(y1, y2), min(y1, y2))

    def depth_at_y(self, fy: int) -> float:
        y_near, y_far = self.y_near_far()
        if y_near == y_far:
            return self.d_near
        fyc = int(np.clip(fy, y_far, y_near))
        t = (fyc - y_far) / max(1, (y_near - y_far))
        return self.d_far * (1 - t) + self.d_near * t

    def band_at_y(self, fy: int) -> float:
        y_near, y_far = self.y_near_far()
        if y_near == y_far:
            return self.b_near
        fyc = int(np.clip(fy, y_far, y_near))
        t = (fyc - y_far) / max(1, (y_near - y_far))
        return self.b_far * (1 - t) + self.b_near * t

    def intruded(self, feet: Tuple[int, int]) -> bool:
        fx, fy = feet
        a = np.array(self.p1, np.float32); b = np.array(self.p2, np.float32)
        v = b - a
        if np.linalg.norm(v) < 1e-3:
            return False

        n = np.array([-v[1], v[0]], np.float32)
        n /= (np.linalg.norm(n) + 1e-6)

        p = np.array([fx, fy], np.float32)
        w = p - a
        len_v = np.linalg.norm(v)
        u = float(np.dot(w, v) / (len_v + 1e-6))
        s = float(np.dot(w, n)) * self.inward_sign

        in_span = (-10.0 <= u <= (len_v + 10.0))
        # 네 최신 코드 기준: margin 없이 s>0.0 만 사용
        max_depth = self.depth_at_y(fy)
        inside_strip = (s > 0.0) and (s <= max_depth)
        return in_span and inside_strip

def update_seat_fsm(seat: SeatWire, any_in_core: bool, dt: float,
                    dwellSeconds=DWELL_SECONDS, exitSeconds=EXIT_SECONDS):
    if seat.state == "OUTSIDE":
        seat.dwell_s = 0.0; seat.exit_s = 0.0
        if any_in_core:
            seat.state = "ENTERING"

    elif seat.state == "ENTERING":
        if any_in_core:
            seat.dwell_s += dt
            if seat.dwell_s >= dwellSeconds:
                seat.state = "INTRUDED"
                seat.exit_s = 0.0
        else:
            seat.exit_s += dt
            if seat.exit_s >= 0.5:
                seat.state = "OUTSIDE"

    elif seat.state == "INTRUDED":
        if not any_in_core:
            seat.exit_s += dt
            if seat.exit_s >= exitSeconds:
                seat.state = "OUTSIDE"
                seat.dwell_s = 0.0
        else:
            seat.exit_s = 0.0

class EMA:
    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.val = None
    def update(self, p):
        v = np.array(p, np.float32)
        if self.val is None: self.val = v
        else: self.val = self.alpha*v + (1-self.alpha)*self.val
        return tuple(map(int, self.val))

# ====== 결과 스키마 ======
@dataclass
class Detection:
    bbox: Tuple[int,int,int,int]
    cls: Optional[int] = None
    score: Optional[float] = None

@dataclass
class InferenceResult:
    frame: np.ndarray
    detections: List[Detection] = field(default_factory=list)    # person 박스만 포함
    intrusion_started: bool = False
    intrusion_active: bool = False
    seat_id: Optional[int] = None
    identity: Optional[str] = None
    identity_conf: Optional[float] = None
    phone_capture: Optional[bool] = None
    meta: Dict[str, Any] = field(default_factory=dict)

# ====== 좌석 FSM 엔진 (다좌석 지원) ======
class SeatIntrusionEngine:
    def __init__(self,
                 seats: Optional[List[SeatWire]] = None,
                 dwell_sec: float = DWELL_SECONDS,
                 exit_sec: float = EXIT_SECONDS,
                 ema_alpha: float = 0.25):
        self.seats = seats or [SeatWire((200,400),(440,400), inward_sign=+1, seat_id=0)]
        self.dwell_sec = dwell_sec
        self.exit_sec = exit_sec
        self.ema = EMA(alpha=ema_alpha)

    @staticmethod
    def _feet_from_bbox(b: Tuple[int,int,int,int]) -> Tuple[int,int]:
        return feet_point_xyxy(*b)

    def update(self, person_bboxes: List[Tuple[int,int,int,int]], dt: float):
        started = False
        active = False
        intruded_seat = None

        feet_points = [ self.ema.update(self._feet_from_bbox(b)) for b in person_bboxes ]

        for s in self.seats:
            any_core = any(s.intruded(fp) for fp in feet_points)
            prev = s.state
            update_seat_fsm(s, any_core, dt, dwellSeconds=self.dwell_sec, exitSeconds=self.exit_sec)

            if prev != "INTRUDED" and s.state == "INTRUDED":
                started = True
                intruded_seat = s.seat_id

            if s.state == "INTRUDED":
                active = True
                if intruded_seat is None:
                    intruded_seat = s.seat_id

        return started, active, intruded_seat

# ====== 오케스트레이터 ======
class InferenceEngine:
    def __init__(self,
                 blur_model: Optional[str] = None,
                 blur_conf: float = 0.5,
                 person_model: Optional[str] = None,
                 person_conf: float = 0.4,
                 seats_config: Optional[str] = None,
                 face_cam_source: str = "0",
                 on_event=None):
        
        self._lock = getattr(self, "_lock", threading.Lock())
        # 1) 블러 엔진 (blur.py)
        self.blur = BlurEngine(model_path=blur_model or os.getenv("BLUR_MODEL", "runs/detect/train11/weights/best.pt"),
                               conf=float(os.getenv("BLUR_CONF", blur_conf)))

        # 2) 사람 검출 모델 (COCO person=0)
        self.person_model = None
        if _YOLO_OK:
            try:
                pm = person_model or os.getenv("PERSON_MODEL", "yolov8n.pt")
                self.person_model = YOLO(pm)
            except Exception:
                self.person_model = None
        self.person_conf = float(os.getenv("PERSON_CONF", person_conf))

        # 3) 좌석 구성 로드 (intrusion_tripwire.py)
        seats = self._load_seats(seats_config or os.getenv("SEATS_CONFIG", "tripwire_perp.json"))
        self.seat_engine = SeatIntrusionEngine(seats=seats,
                                               dwell_sec=DWELL_SECONDS,
                                               exit_sec=EXIT_SECONDS,
                                               ema_alpha=float(os.getenv("EMA_ALPHA","0.25")))

        # 4) 스마트폰 후면 촬영 감지 (photo.py)
        self.phone = PhoneBackDetector(
            weights=os.getenv("PHONE_MODEL", "runs/detect/train12/weights/best.pt"),
            cam_source=os.getenv("CAM1_SOURCE", face_cam_source),
            conf=float(os.getenv("PHONE_CONF", "0.8")),
            dwell=float(os.getenv("PHONE_DWELL", "1.5")),
            hold=float(os.getenv("PHONE_HOLD", "2.0")),
            imgsz=int(os.getenv("PHONE_IMGSZ", "640")),
        )
        self._last_phone_capture: bool | None = None
        self._phone_busy: bool = False  # 중복 트리거 방지 플래그

        # 타이밍
        self._last_ts = time.time()
        self._corr_id: Optional[str] = None
        self._last_identity: Optional[Tuple[str,float]] = None

    def _load_seats(self, path: Optional[str]) -> List[SeatWire]:
        if not path:
            return [SeatWire((200,400),(440,400), inward_sign=+1, seat_id=0)]
        try:
            if not os.path.exists(path):
                return [SeatWire((200,400),(440,400), inward_sign=+1, seat_id=0)]
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            seats: List[SeatWire] = []
            for i, d in enumerate(data, start=0):
                seats.append(SeatWire(
                    p1=tuple(d["p1"]),
                    p2=tuple(d["p2"]),
                    b_near=float(d.get("b_near", 20.0)),
                    b_far=float(d.get("b_far", 8.0)),
                    inward_sign=int(d.get("inward_sign", 1)),
                    d_near=float(d.get("d_near", 180.0)),
                    d_far=float(d.get("d_far", 120.0)),
                    seat_id=int(d.get("seat_id", i)),
                ))
            return seats or [SeatWire((200,400),(440,400), inward_sign=+1, seat_id=0)]
        except Exception:
            return [SeatWire((200,400),(440,400), inward_sign=+1, seat_id=0)]

    # ---- 사람 검출 (person only) ----
    def _detect_persons(self, frame: np.ndarray) -> List[Detection]:
        dets: List[Detection] = []
        if self.person_model is None:
            return dets
        r = self.person_model.predict(frame, verbose=False, conf=self.person_conf, classes=[0])[0]
        if r and r.boxes is not None:
            h, w = frame.shape[:2]
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                # 프레임 경계로 클램프
                x1 = int(np.clip(x1, 0, w-1)); x2 = int(np.clip(x2, 0, w-1))
                y1 = int(np.clip(y1, 0, h-1)); y2 = int(np.clip(y2, 0, h-1))
                if x2 > x1 and y2 > y1:
                    dets.append(Detection(bbox=(x1,y1,x2,y2), cls=0, score=float(b.conf[0])))
        return dets
    
    # 침입 전이 시, 모니터 웹캠으로 스마트폰 후면 촬영 감지를 비동기 수행
    def _kick_phone_thread(self, corr_id: str, seat_id: int | None):
        if self._phone_busy:
            return  # 이미 동작 중이면 중복 실행 방지
        self._phone_busy = True

        def _run():
            try:
                timeout = float(os.getenv("PHONE_SCAN_TIMEOUT", "3.0"))
                ok = self.phone.scan(timeout_sec=timeout)  # True/False
                with self._lock:
                    self._last_phone_capture = bool(ok)
            finally:
                with self._lock:
                    self._phone_busy = False

            # (선택) 이벤트 서버로 알림 보내고 싶으면 on_event 사용
            if self.on_event:
                self.on_event({
                    "type": "phone_capture",
                    "correlation_id": corr_id,
                    "seat_id": seat_id,
                    "detected": bool(self._last_phone_capture),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })

        threading.Thread(target=_run, daemon=True).start()


    def process_frame(self, frame: np.ndarray, camera_id: str = "cam2",
                      do_blur: bool = True, do_intrusion: bool = True) -> InferenceResult:
        now = time.time()
        dt = min(0.2, max(0.0, now - getattr(self, "_last_ts", now)))
        self._last_ts = now

        # 1) 블러 적용 (네 blur.py 로직 그대로)
        if do_blur:
            frame, _ = self.blur.process(frame)

        # 2) 사람 검출 → 좌석 FSM
        dets = self._detect_persons(frame) if do_intrusion else []
        started = active = False
        seat_id = None
        if do_intrusion:
            person_bboxes = [d.bbox for d in dets]
            started, active, seat_id = self.seat_engine.update(person_bboxes, dt)

            if started:
                self._corr_id = str(uuid.uuid4())
                # TODO: 여기서 얼굴 인식/이벤트 포스트 등을 트리거할 수 있음.
            
                # [ADD] 침입 "전이" 순간에만 모니터 웹캠 스캔 시작(비동기)
                self._kick_phone_thread(self._corr_id, seat_id)

        with self._lock:
            phone_flag = self._last_phone_capture

        return InferenceResult(
            frame=frame,
            detections=dets,
            intrusion_started=started,
            intrusion_active=active,
            seat_id=seat_id,
            identity=None,
            identity_conf=None,
            phone_capture=phone_flag,
            meta={"camera_id": camera_id, "correlation_id": self._corr_id},
        )

# ====== 외부 진입점 ======
_ENGINE: Optional[InferenceEngine] = None
def _engine() -> InferenceEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = InferenceEngine()
    return _ENGINE

def run_inference_on_image(frame: np.ndarray,
                           camera_id: str = "cam2",
                           do_blur: bool = True,
                           do_intrusion: bool = True) -> InferenceResult:
    """
    stream.py 프레임 루프에서 사용:
        res = run_inference_on_image(frame, camera_id="cam2", do_blur=True, do_intrusion=True)
        jpg = cv2.imencode(".jpg", res.frame, [...])
    """
    return _engine().process_frame(frame, camera_id=camera_id,
                                   do_blur=do_blur, do_intrusion=do_intrusion)
