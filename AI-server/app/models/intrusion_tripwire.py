# intrusion_tripwire.py
import argparse, json, time, os
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# 사람 욜로랑 50%이상 기준 변경
COVERAGE_THR = float(os.getenv("COVERAGE_THR", "0.5"))  # 사람박스의 50% 이상이 좌석 안이면 True env우선, 기본 0.5
PERSON_BAND_FRAC = float(os.getenv("PERSON_BAND_FRAC", "0.60"))  # 사람 박스 아래쪽 60%만
HALO_PX          = int(os.getenv("HALO_PX", "120"))           # 좌석 주변 후보 필터
SEAT_PAD_PX      = int(os.getenv("SEAT_PAD_PX", "8"))         # 좌석 패딩
# ==== 상수 설정 ====
DWELL_SECONDS = 5.0   # 코어존 내 연속 체류해야 하는 시간(초) - 도구 기본
EXIT_SECONDS  = float(os.getenv("EXIT_SECONDS", "1.5"))   # 코어존 밖 연속 이탈 시간(초) env우선, 기본 1.5s


def _rect_norm(p1, p2):
    (x1, y1), (x2, y2) = p1, p2
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))  # (sx1, sy1, sx2, sy2)

def _rect_intersection(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return (ix2 - ix1) * (iy2 - iy1)


##
def _as_int(v, default=None):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default

def _as_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default




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
    coord_norm: Optional[bool] = None
    ref_w: Optional[int] = None
    ref_h: Optional[int] = None
    # 상태/타이머
    state: str = "OUTSIDE"
    dwell_s: float = 0.0
    exit_s: float = 0.0
    # 선택: seat_id를 외부에서 지정해줄 수 있도록 확장
    seat_id: Optional[int] = None

   # 좌석 사각형과 겹치는 '사람 하단 밴드'가 하나라도 thr 이상이면 True.
    # 부가로 self._cov_ratio, self._cov_box 저장(디버그/오버레이용)
    def inside_by_coverage(self, person_boxes_xyxy, thr: float = COVERAGE_THR) -> bool:
        sx1, sy1, sx2, sy2 = _rect_norm(self.p1, self.p2)
        # 좌석 확장(팔걸이/기울기 여유)
        sx1 -= SEAT_PAD_PX; sy1 -= SEAT_PAD_PX
        sx2 += SEAT_PAD_PX; sy2 += SEAT_PAD_PX

        # 좌석 주변 HALO (먼 사람 배제)
        hx1, hy1, hx2, hy2 = (sx1 - HALO_PX, sy1 - HALO_PX, sx2 + HALO_PX, sy2 + HALO_PX)

        best_ratio = 0.0
        best_box = None

        band_frac = max(0.2, min(PERSON_BAND_FRAC, 1.0))

        for (x1, y1, x2, y2) in (person_boxes_xyxy or []):
            if x2 <= x1 or y2 <= y1:
                continue
            # 사람 중심이 HALO 안에 없다 → 스킵
            cx = 0.5 * (x1 + x2); cy = 0.5 * (y1 + y2)
            if not (hx1 <= cx <= hx2 and hy1 <= cy <= hy2):
                continue

            # 사람 하단 밴드(rect)
            h = y2 - y1
            by1 = int(y2 - h * band_frac)               # 아래쪽 band_frac만 남김
            pbx1, pby1, pbx2, pby2 = x1, max(y1, by1), x2, y2
            band_area = float(max(0, pbx2 - pbx1) * max(0, pby2 - pby1))
            if band_area <= 0:
                continue

            inter = _rect_intersection(pbx1, pby1, pbx2, pby2, sx1, sy1, sx2, sy2)
            if inter <= 0:
                continue

            cover_ratio = inter / band_area
            if cover_ratio > best_ratio:
                best_ratio, best_box = cover_ratio, (pbx1, pby1, pbx2, pby2)

        # 디버그용 저장(대시보드 오버레이에 씀)
        self._cov_ratio = best_ratio
        self._cov_box = best_box
        return best_ratio >= thr


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


class TripwireApp:
    def __init__(self, cam, seats, dwell_sec, on_intrusion):
        self.cam = 0 if str(cam).strip() == "0" else cam
        self.seats = self._normalize_seats(seats)
        self.dwell_sec = float(dwell_sec)
        self.exit_sec = float(EXIT_SECONDS)
        self.on_intrusion = on_intrusion
        self.ema = EMA(alpha=float(os.getenv("EMA_ALPHA", "0.25")))
        self.person_model = self._load_person_model()

        # 침입 상태 저장(stream_h추가함)
        self.active_intrusion = {} # { seat_id: {"start": ts, "person": str, "conf": float} }
        self.stream_w = None
        self.stream_h = None
        self._seats_pixelized = False

    def _load_person_model(self):
        model_path = os.getenv("PERSON_MODEL", "yolov8n.pt")
        try:
            return YOLO(model_path)
        except Exception as e:
            print(f"[WARN] YOLO load failed: {e}")
            return None
        
    def set_stream_size(self, w: int, h: int):
        self.stream_w = int(w); self.stream_h = int(h)
        self._seats_pixelized = False  # 크기가 바뀌면 다시 픽셀화 필요

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
                seat_id = _as_int(s.get("seat_id"), i)

                out.append(SeatWire(
                    p1=tuple(s["p1"]),
                    p2=tuple(s["p2"]),
                    b_near=_as_float(s.get("b_near"), 20.0),
                    b_far=_as_float(s.get("b_far"), 8.0),
                    inward_sign=_as_int(s.get("inward_sign"), 1),
                    d_near=_as_float(s.get("d_near"), 180.0),
                    d_far=_as_float(s.get("d_far"), 120.0),
                    coord_norm=bool(s.get("coord_norm", False)),
                    ref_w=_as_int(s.get("ref_w")),   # None이면 그대로 None
                    ref_h=_as_int(s.get("ref_h")),   # None이면 그대로 None
                    seat_id=seat_id,
                ))

        return out
    
    def _pixelize_seats_if_needed(self):
        # 이미 저장되어 있는 정규화 메타정보를 엔진에 전달하는 코드
        if not self.stream_w or not self.stream_h:
            return
        if self._seats_pixelized:
            return

        W, H = self.stream_w, self.stream_h
        pix_seats: List[SeatWire] = []

        for i, s in enumerate(self.seats or []):
            # dict/SeatWire 모두 처리 → 공통 dict d로 맞춤
            d = s if isinstance(s, dict) else {
                "p1": s.p1, "p2": s.p2,
                "d_near": s.d_near, "d_far": s.d_far,
                "inward_sign": s.inward_sign,
                "seat_id": s.seat_id,
                "coord_norm": (s.coord_norm if s.coord_norm is not None else False),
                "ref_w": s.ref_w,
                "ref_h": s.ref_h,
            }

            p1, p2 = d.get("p1"), d.get("p2")
            coord_norm = bool(d.get("coord_norm", False))
            ref_w = d.get("ref_w") or W
            ref_h = d.get("ref_h") or H

            # 좌표가 0~1 범위면 정규화로 간주 (백업 조건)
            try:
                mx = max(float(p1[0]), float(p2[0]))
                my = max(float(p1[1]), float(p2[1]))
                coord_norm = coord_norm or (mx <= 1.01 and my <= 1.01)
            except Exception:
                coord_norm = False

            # 1) 정규화라면 픽셀로 변환
            if coord_norm:
                P1 = (int(round(float(p1[0]) * W)), int(round(float(p1[1]) * H)))
                P2 = (int(round(float(p2[0]) * W)), int(round(float(p2[1]) * H)))
                # 깊이 값은 저장 당시 높이에 대한 픽셀 → 현재 프레임 높이로 비율 보정
                scale_y = (float(H) / float(ref_h)) if float(ref_h) > 1 else 1.0
                d_near = float(d.get("d_near", 0.0)) * scale_y
                d_far  = float(d.get("d_far",  0.0)) * scale_y
            else:
                # 원래부터 픽셀 단위
                P1 = (int(p1[0]), int(p1[1]))
                P2 = (int(p2[0]), int(p2[1]))
                d_near = float(d.get("d_near", 0.0))
                d_far  = float(d.get("d_far",  0.0))

            # 2) (선택) 저장 자체가 같은 점인지 경고 (정규화/픽셀 무관)
            try:
                if abs(float(p1[0]) - float(p2[0])) < 1e-6 and abs(float(p1[1]) - float(p2[1])) < 1e-6:
                    print(f"[SEAT] ⚠ same points in source; seat disabled: p1n={p1}, p2n={p2}")
                    # 그래도 픽셀 좌표로 길이가 생길 수도 있으니, 여기서 바로 continue 하지 말고 아래에서 픽셀 길이로 최종 판단
            except Exception:
                pass

            # 3) ★ 픽셀 좌표로 길이 계산/판정 (핵심)
            vx = P2[0] - P1[0]
            vy = P2[1] - P1[1]
            seg_len = (vx*vx + vy*vy) ** 0.5
            if seg_len < 5.0:
                print(f"[SEAT] ❌ invalid segment (len={seg_len:.2f}px). seat disabled: p1={P1}, p2={P2}")
                # 이 좌석은 건너뜀
                continue

            # 4) inward_sign 자동 보정 (화면 아래(y+)를 안쪽으로)
            nx, ny = -vy, vx
            nlen = (nx*nx + ny*ny) ** 0.5 or 1.0
            nx, ny = nx/nlen, ny/nlen
            inward_auto = +1 if ny >= 0 else -1

            seat_id_safe = _as_int(d.get("seat_id"), i)
            inward_safe  = _as_int(d.get("inward_sign"), inward_auto)

            # 5) SeatWire 생성 및 저장
            s_obj = SeatWire(
                p1=P1, p2=P2,
                inward_sign=inward_safe,
                d_near=d_near, d_far=d_far,
                seat_id=seat_id_safe,
            )
            pix_seats.append(s_obj)

            # 6) 디버그 (정규화/픽셀 모두 보여주기)
            print(f"[SEAT] pix seat#{s_obj.seat_id} p1n={p1} p2n={p2}  p1={s_obj.p1} p2={s_obj.p2}  len={seg_len:.2f}px inward={s_obj.inward_sign}")

        # 교체
        self.seats = pix_seats
        self._seats_pixelized = True
        print(f"[TripwireApp] seats pixelized → W={W} H={H}, count={len(self.seats)}")
        for s in self.seats:
            print(f"[TripwireApp] seat#{s.seat_id} p1={s.p1} p2={s.p2} d_near={s.d_near:.1f} d_far={s.d_far:.1f}")


    
    def update(self, person_bboxes: List[Tuple[int,int,int,int]], dt: float, frame_wh: Optional[Tuple[int,int]] = None):
    # ↓ 추가: 프레임 크기 반영 → 좌석 픽셀화
        if frame_wh and (self.stream_w != frame_wh[0] or self.stream_h != frame_wh[1]):
            self.set_stream_size(frame_wh[0], frame_wh[1])
        self._pixelize_seats_if_needed()

        # 1-1) 좌석 선분 유효성/부호 자동 교정
        valid_seats = []
        for s in (self.seats or []):
            # 선분 길이 가드
            vx = float(s.p2[0] - s.p1[0]); vy = float(s.p2[1] - s.p1[1])
            len_v = (vx**2 + vy**2) ** 0.5
            if len_v < 5.0:
                print(f"[SEAT] ❌ invalid segment (len={len_v:.2f}px). seat disabled: p1={s.p1}, p2={s.p2}")
                continue
            # 안쪽 부호 자동 교정: 화면 아래(y+)를 안쪽으로
            nx, ny = -vy, vx
            nlen = (nx**2 + ny**2) ** 0.5 or 1.0
            nx, ny = nx / nlen, ny / nlen
            s.inward_sign = +1 if ny >= 0 else -1
            valid_seats.append(s)
        if not valid_seats:
            return (False, False, None)
        
        started = False
        active = False
        intruded_seat_id = None

        try:
        #     feet_points = []
        #     for (x1, y1, x2, y2) in (person_bboxes or []):
        #         fx = int((x1 + x2) / 2)
        #         fy = int(y2)
        #         feet_points.append((fx, fy))

        #     if feet_points:
        #         if not hasattr(self, "_ema") or self._ema is None:
        #             self._ema = EMA(alpha=0.25)
        #         feet_points = [self._ema.update(fp) for fp in feet_points]

        #         fx, fy = feet_points[0]
        #         for idx, s in enumerate(valid_seats):
        #             a = np.array(s.p1, np.float32); b = np.array(s.p2, np.float32)
        #             v = b - a
        #             n = np.array([-v[1], v[0]], np.float32)
        #             n /= (np.linalg.norm(n) + 1e-6)
        #             w = np.array([fx, fy], np.float32) - a
        #             len_v = np.linalg.norm(v)
        #             u = float(np.dot(w, v) / (len_v + 1e-6))  # 선분 방향 거리
        #             s_val = float(np.dot(w, n)) * s.inward_sign  # 안쪽(+)
        #             max_depth = s.depth_at_y(fy)                 # 해당 y에서 밴드 폭
        #             in_span = (-10.0 <= u <= (len_v + 10.0))     # 여유 포함 범위
        #             inside = (s_val > 0.0) and (s_val <= max_depth)
        # 발 좌표 기준 코드 대체. 
            pass         

            # 좌석별 FSM
            for idx, s in enumerate(valid_seats or []):
                # any_core = any(s.intruded(fp) for fp in feet_points) #발 기준
                any_core = s.inside_by_coverage(person_bboxes, thr=COVERAGE_THR)#사람 기준
                prev = s.state
                update_seat_fsm(s, any_core, dt,
                                dwellSeconds=self.dwell_sec,
                                exitSeconds=EXIT_SECONDS)
                # ✅ 여기 추가: dwell 카운트 시작(OUTSIDE → ENTERING) 로그
                if prev == "OUTSIDE" and s.state == "ENTERING":
                    print(f"[DWELL] counting start, dwell={self.dwell_sec}s")
                # === 침입 시작 ===
                if prev != "INTRUDED" and s.state == "INTRUDED":
                    started = True
                    intruded_seat_id = s.seat_id or idx
                    # 내부 기록만 남김
                    self.active_intrusion[intruded_seat_id] = {
                        "start": time.time(),
                        "person": None,
                        "conf": None
                    }

                # === 침입 중 ===
                if s.state == "INTRUDED":
                    active = True
                    if intruded_seat_id is None:
                        intruded_seat_id = s.seat_id or idx

                # === 침입 종료 ===
                if prev == "INTRUDED" and s.state == "OUTSIDE":
                    seat = s.seat_id or idx
                    info = self.active_intrusion.pop(seat, None)
                    if info:
                        start_t = info["start"]
                        end_t = time.time()
                        duration = round(end_t - start_t, 1)

                        if callable(self.on_intrusion):
                            try:
                                self.on_intrusion(
                                    seat, end_t,
                                    {
                                        "event_type": "intrusion",                     # ← 여기만 변경!
                                        "seat_id": seat,
                                        "camera_id": str(self.cam),
                                        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_t)),
                                        "ended_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end_t)),
                                        "duration_sec": duration,
                                        "meta": {
                                            "seat_no": seat,                           # DB에는 seat_id 컬럼이 없고 meta.seat_no 사용
                                            "device_id": str(self.cam),               # meta.device_id 사용
                                            "dwell_sec": self.dwell_sec
                                        }
                                    }
                                )
                            except Exception as e:
                                print(f"[Tripwire.on_intrusion] failed: {e}")
        except Exception as e:
            print("[Tripwire.update] EXC", e)

        # 반환값 보장
        return started, active, intruded_seat_id



    def set_config(self, seats=None, dwell_sec=None):
        if seats is not None:
            self.seats = self._normalize_seats(seats)
            self._seats_pixelized = False
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
