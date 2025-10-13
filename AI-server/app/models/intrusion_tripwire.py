# intrusion_tripwire.py
import argparse, json, time, os
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ==== 상수 설정 ====
DWELL_SECONDS = 5.0   # 코어존 내 연속 체류해야 하는 시간(초) - 도구 기본
EXIT_SECONDS  = 1.5   # 코어존 밖 연속 이탈 시간(초)

def now_ts(): return time.time()
def draw_text(img, text, org, color=(255,255,255), scale=0.6, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
def feet_point_xyxy(x1,y1,x2,y2): return int((x1+x2)/2), int(y2)

@dataclass
class SeatWire:
    p1: Tuple[int,int]
    p2: Tuple[int,int]
    b_near: float = 20.0  # 아래쪽(가까운 곳) 버퍼(px)
    b_far: float  = 8.0   # 위쪽(먼 곳) 버퍼(px)
    inward_sign: int = +1 # 수직이등분선의 "안쪽" 방향(뒤집으려면 -1)
    d_near: float = 180.0  # ★ 안쪽 ‘최대 깊이’(아래쪽/가까운 곳)
    d_far:  float = 120.0  # ★ 안쪽 ‘최대 깊이’(위쪽/먼 곳)
    # 상태/타이머
    state: str = "OUTSIDE"
    dwell_s: float = 0.0
    exit_s: float = 0.0
    # 선택: seat_id를 외부에서 지정해줄 수 있도록 확장
    seat_id: Optional[int] = None

    def y_near_far(self):
        y1, y2 = self.p1[1], self.p2[1]
        return (max(y1,y2), min(y1,y2))
    
    def depth_at_y(self, fy: int) -> float:
        """화면 아래로 갈수록 깊이를 더 넓게(원근 보정)."""
        y_near, y_far = self.y_near_far()
        if y_near == y_far:
            return self.d_near
        fyc = int(np.clip(fy, y_far, y_near))
        t = (fyc - y_far) / max(1, (y_near - y_far))
        return self.d_far * (1 - t) + self.d_near * t

    def band_at_y(self, fy: int) -> float:
        y_near, y_far = self.y_near_far()
        if y_near == y_far: return self.b_near
        fyc = int(np.clip(fy, y_far, y_near))
        t = (fyc - y_far) / max(1, (y_near - y_far))
        return self.b_far * (1 - t) + self.b_near * t

    def intruded(self, feet: Tuple[int,int]) -> bool:
        fx, fy = feet
        a = np.array(self.p1, np.float32); b = np.array(self.p2, np.float32)
        v = b - a
        if np.linalg.norm(v) < 1e-3:
            return False

        # 선분 좌표계: 축 v(선 방향), n(수직/안쪽)
        n = np.array([-v[1], v[0]], np.float32)
        n /= (np.linalg.norm(n) + 1e-6)

        # p를 a를 원점으로 투영
        p = np.array([fx, fy], np.float32)
        w = p - a
        len_v = np.linalg.norm(v)
        u = float(np.dot(w, v) / (len_v + 1e-6))        # 선분 방향 거리(픽셀)
        s = float(np.dot(w, n)) * self.inward_sign      # 안쪽(+) 방향 수직 거리

        # 1) 선분 범위 안쪽인지: u가 [약간의 여유 포함 0~|v|]에 들어와야 함
        in_span = (-10.0 <= u <= (len_v + 10.0))

        # 2) 안쪽 사이드인지 + 0 < s <= 최대 깊이
        max_depth = self.depth_at_y(fy)
        inside_strip = (s > 0.0) and (s <= max_depth)

        return in_span and inside_strip

# 좌석 FSM
def update_seat_fsm(seat: SeatWire, any_in_core: bool, dt: float,
                    dwellSeconds: float = DWELL_SECONDS, exitSeconds: float = EXIT_SECONDS):
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
            if seat.exit_s >= 0.5:    # 바운스 롤백
                seat.state = "OUTSIDE"

    elif seat.state == "INTRUDED":
        if not any_in_core:
            seat.exit_s += dt
            if seat.exit_s >= exitSeconds:
                seat.state = "OUTSIDE"
                seat.dwell_s = 0.0
        else:
            seat.exit_s = 0.0

# 발좌표 EMA
class EMA:
    def __init__(self, alpha=0.25):
        self.alpha = float(alpha)
        self.val = None
    def update(self, p):
        v = np.array(p, np.float32)
        if self.val is None: self.val = v
        else: self.val = self.alpha*v + (1-self.alpha)*self.val
        return tuple(map(int, self.val))


# =========================
# 런타임용 TripwireApp (inference.py에서 사용)
# class TripwireApp:
#     def __init__(self, cam, seats, dwell_sec, on_intrusion):
#         self.cam = 0 if str(cam).strip() == "0" else cam
#         self.seats: List[SeatWire] = self._normalize_seats(seats)
#         self.dwell_sec = float(dwell_sec)
#         self.exit_sec = float(EXIT_SECONDS)
#         self.on_intrusion = on_intrusion
#         self.ema = EMA(alpha=float(os.getenv("EMA_ALPHA", "0.25")))
#         self.person_model = self._load_person_model()
class TripwireApp:
    def __init__(self, cam, seats, dwell_sec, on_intrusion):
        self.cam = 0 if str(cam).strip() == "0" else cam
        self.seats = self._normalize_seats(seats)
        self.dwell_sec = float(dwell_sec)
        self.exit_sec = float(EXIT_SECONDS)
        self.on_intrusion = on_intrusion
        self.ema = EMA(alpha=float(os.getenv("EMA_ALPHA", "0.25")))
        self.person_model = self._load_person_model()

        # 침입 상태 저장
        self.active_intrusion = {} # { seat_id: {"start": ts, "person": str, "conf": float} }

    def _load_person_model(self):
        model_path = os.getenv("PERSON_MODEL", "yolov8n.pt")
        try:
            return YOLO(model_path)
        except Exception as e:
            print(f"[WARN] YOLO load failed: {e}")
            return None

    def _normalize_seats(self, seats):
        out: List[SeatWire] = []
        if not seats:
            return out
        for i, s in enumerate(seats):
            if isinstance(s, SeatWire):
                if s.seat_id is None: s.seat_id = i
                out.append(s)
            else:
                # dict → SeatWire
                out.append(SeatWire(
                    p1=tuple(s["p1"]),
                    p2=tuple(s["p2"]),
                    b_near=float(s.get("b_near", 20.0)),
                    b_far=float(s.get("b_far", 8.0)),
                    inward_sign=int(s.get("inward_sign", 1)),
                    d_near=float(s.get("d_near", 180.0)),
                    d_far=float(s.get("d_far", 120.0)),
                    seat_id=int(s.get("seat_id", i)),
                ))
        return out
    
    def update(self, person_bboxes: List[Tuple[int,int,int,int]], dt: float):
        started = False
        active = False
        intruded_seat_id = None

        try:
            # 1) 바운딩박스 → 발 좌표
            feet_points = []
            for (x1, y1, x2, y2) in (person_bboxes or []):
                fx = int((x1 + x2) / 2)
                fy = int(y2)
                feet_points.append((fx, fy))

            if feet_points:
                if not hasattr(self, "_ema") or self._ema is None:
                    self._ema = EMA(alpha=0.25)
                feet_points = [self._ema.update(fp) for fp in feet_points]

            # 2) 좌석별 FSM
            for idx, s in enumerate(self.seats or []):
                any_core = any(s.intruded(fp) for fp in feet_points)
                prev = s.state

                update_seat_fsm(s, any_core, dt,
                                dwellSeconds=self.dwell_sec,
                                exitSeconds=self.exit_sec)

                # --- 침입 시작: INTRUDED 진입 순간 시작시각만 기록 ---
                if prev != "INTRUDED" and s.state == "INTRUDED":
                    seat = s.seat_id if s.seat_id is not None else idx
                    self.active_intrusion[seat] = {"start": time.time(), "person": None, "conf": None}
                    started = True
                    intruded_seat_id = seat

                # --- 침입 중: active 플래그 유지 ---
                if s.state == "INTRUDED":
                    active = True
                    if intruded_seat_id is None:
                        intruded_seat_id = s.seat_id if s.seat_id is not None else idx

                # --- 침입 종료: OUTSIDE 전이 시 완결 이벤트 1회 콜백 ---
                if prev == "INTRUDED" and s.state == "OUTSIDE":
                    seat = s.seat_id if s.seat_id is not None else idx
                    info = self.active_intrusion.pop(seat, None)
                    if info:
                        start_t = info.get("start", time.time())
                        end_t   = time.time()
                        duration = round(end_t - start_t, 1)
                        person   = info.get("person") or "Unknown"
                        conf     = info.get("conf")

                        if callable(self.on_intrusion):
                            try:
                                self.on_intrusion(
                                    seat, end_t,
                                    {
                                        "type": "intrusion",
                                        "seat_id": seat,
                                        "camera_id": str(self.cam),
                                        "person_id": person,
                                        "confidence": conf,
                                        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_t)),
                                        "ended_at":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end_t)),
                                        "duration_sec": duration,
                                        "meta": {"seat_no": seat, "device_id": str(self.cam), "user_label": person},
                                    }
                                )
                            except Exception as e:
                                print(f"[Tripwire] on_intrusion failed: {e}")
        except Exception as e:
            print(f"[Tripwire.update] EXC {e}")

        # ✅ 어떤 경로로 와도 항상 튜플 반환
        return started, active, intruded_seat_id



    def set_config(self, seats=None, dwell_sec=None):
        if seats is not None:
            self.seats = self._normalize_seats(seats)
            print(f"[TripwireApp] seats updated -> {len(self.seats)}")
        if dwell_sec is not None:
            self.dwell_sec = float(dwell_sec)
            print(f"[TripwireApp] dwell updated -> {self.dwell_sec}s")

    def _detect_person_bboxes(self, frame):
        if self.person_model is None:
            return []
        r = self.person_model.predict(frame, verbose=False, conf=float(os.getenv("PERSON_CONF", "0.4")), classes=[0])[0]
        h, w = frame.shape[:2]
        boxes = []
        if r and r.boxes is not None:
            for b in r.boxes:
                x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                x1 = int(np.clip(x1,0,w-1)); x2 = int(np.clip(x2,0,w-1))
                y1 = int(np.clip(y1,0,h-1)); y2 = int(np.clip(y2,0,h-1))
                if x2 > x1 and y2 > y1:
                    boxes.append((x1,y1,x2,y2))
        return boxes

# 도구/GUI용 TripwireTool (예전 TripwireApp 이름)

class TripwireTool:
    def __init__(self, args):
        self.args = args
        self.model = YOLO(args.model)
        self.person_cls = 0
        self.seats: List[SeatWire] = []
        self.temp_pts: List[Tuple[int,int]] = []
        self.add_mode = False
        self.debug = False
        self.help_on = True
        self.cooldown = 0.35
        self.last_fire = {}

        # undistort(옵션)
        self.K = self.dist = None
        if args.camera_K and args.camera_dist:
            pK, pD = Path(args.camera_K), Path(args.camera_dist)
            if pK.exists() and pD.exists():
                try:
                    self.K = np.load(str(pK)); self.dist = np.load(str(pD))
                    print("[INFO] Undistort enabled.]")
                except Exception as e:
                    print(f"[WARN] Undistort load failed: {e}")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(s) for s in self.seats], f, ensure_ascii=False, indent=2)
        print(f"[SAVE] {path}")

    def load(self, path):
        p = Path(path)
        if not p.exists(): 
            print("[LOAD] no file"); return
        data = json.loads(p.read_text(encoding="utf-8"))
        self.seats = [
            SeatWire(
                p1=tuple(d["p1"]),
                p2=tuple(d["p2"]),
                b_near=float(d.get("b_near", 20.0)),
                b_far=float(d.get("b_far", 8.0)),
                inward_sign=int(d.get("inward_sign", 1)),
                d_near=float(d.get("d_near", 180.0)),
                d_far=float(d.get("d_far", 120.0)),
            )
            for d in data
        ]
        print(f"[LOAD] {path} (seats={len(self.seats)})")
