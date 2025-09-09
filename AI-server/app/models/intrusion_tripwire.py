# intrusion_tripwire.py
import argparse, json, time
from dataclasses import dataclass, asdict
from typing import List, Tuple
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ==== 상수 설정 (요청사항) ====
DWELL_SECONDS = 8.0   # 코어존 내 연속 체류해야 하는 시간(초)
EXIT_SECONDS  = 1.5   # 코어존 밖 연속 이탈 시간(초) — 원하면 변경

def now_ts(): return time.time()
def draw_text(img, text, org, color=(255,255,255), scale=0.6, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
def feet_point_xyxy(x1,y1,x2,y2): return int((x1+x2)/2), int(y2)

def dist_to_segment(p, a, b):
    ax, ay = a; bx, by = b; px, py = p
    v = np.array([bx-ax, by-ay], np.float32)
    w = np.array([px-ax, py-ay], np.float32)
    v2 = float(v.dot(v))
    if v2 < 1e-6: return float(np.linalg.norm(w)), 0.0
    t = float(w.dot(v) / v2)
    t_clamped = max(0.0, min(1.0, t))
    closest = np.array([ax, ay], np.float32) + t_clamped * v
    dist = float(np.linalg.norm(np.array([px,py], np.float32) - closest))
    return dist, t

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

    def draw(self, img, debug=False):
        cv2.line(img, self.p1, self.p2, (0,255,255), 2)  # 노란 좌석선
        a = np.array(self.p1, np.float32); b = np.array(self.p2, np.float32)
        v = b - a
        if np.linalg.norm(v) > 1e-3:
            n = np.array([-v[1], v[0]], np.float32)
            n /= (np.linalg.norm(n) + 1e-6)
            # 각 끝점 y에 맞는 깊이 (원근 보정)
            d1 = self.depth_at_y(int(self.p1[1]))
            d2 = self.depth_at_y(int(self.p2[1]))
            # 안쪽 방향으로 사다리꼴 만들기
            n_in = n * float(self.inward_sign)
            poly = np.array([
                (a + n_in * 0), (b + n_in * 0),
                (b + n_in * d2), (a + n_in * d1)
            ], np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [poly], (200, 0, 200))     # 보라색
            cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
            # 화살표(안쪽)
            mid = ((self.p1[0]+self.p2[0])//2, (self.p1[1]+self.p2[1])//2)
            tip = (int(mid[0] + n_in[0]*40), int(mid[1] + n_in[1]*40))
            cv2.arrowedLine(img, mid, tip, (0,0,255), 2, tipLength=0.3)
        if debug:
            draw_text(img, f"inward={self.inward_sign}  depth(n/f)={self.d_near:.0f}/{self.d_far:.0f}",
                    (self.p1[0], max(20,self.p1[1]-8)), (100,255,100), 0.55, 2)

# 좌석 FSM
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
        self.alpha = alpha
        self.val = None
    def update(self, p):
        v = np.array(p, np.float32)
        if self.val is None: self.val = v
        else: self.val = self.alpha*v + (1-self.alpha)*self.val
        return tuple(map(int, self.val))

class TripwireApp:
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

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.add_mode:
            self.temp_pts.append((int(x), int(y)))
            if len(self.temp_pts) == 2:
                p1, p2 = self.temp_pts
                self.seats.append(SeatWire(p1, p2))
                self.temp_pts.clear()
                self.add_mode = False
                print(f"[ADD] seat #{len(self.seats)}: {p1}->{p2}")

    def run(self):
        src = 0 if self.args.source.strip()=="0" else self.args.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened(): raise RuntimeError(f"소스를 열 수 없습니다: {self.args.source}")

        win = "Tripwire (perpendicular-inside)"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, self.on_mouse)

        fps = cap.get(cv2.CAP_PROP_FPS); 
        if not fps or fps < 1: fps = 25.0
        last_ts = time.time()

        ema = EMA(alpha=0.25)

        while True:
            ok, frame = cap.read()
            if not ok: break
            now = time.time()
            dt = max(1.0/fps, now - last_ts)
            last_ts = now
            if self.K is not None and self.dist is not None:
                frame = cv2.undistort(frame, self.K, self.dist)

            h, w = frame.shape[:2]
            vis = frame.copy()

            # YOLO(person)
            results = self.model.predict(frame, verbose=False, conf=self.args.conf, classes=[self.person_cls])

            feet_points = []
            if results and results[0].boxes is not None:
                for b in results[0].boxes:
                    x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                    x1 = int(np.clip(x1,0,w-1)); x2 = int(np.clip(x2,0,w-1))
                    y1 = int(np.clip(y1,0,h-1)); y2 = int(np.clip(y2,0,h-1))
                    if x2 > x1 and y2 > y1:
                        fx, fy = feet_point_xyxy(x1,y1,x2,y2)
                        fx, fy = ema.update((fx, fy))  # EMA 적용
                        feet_points.append((fx, fy))
                        if self.debug:
                            cv2.circle(vis, (fx,fy), 5, (0,255,0), -1)
                        cv2.rectangle(vis, (x1,y1), (x2,y2), (120,120,120), 2)

            if self.add_mode and self.temp_pts:
                cv2.circle(vis, self.temp_pts[0], 5, (0,200,255), -1)

            intruded_seats = set()

            for i, s in enumerate(self.seats, 1):
                s.draw(vis, debug=self.debug)

                # 이 좌석 코어존 안에 '누군가' 있는가?
                any_core = any(s.intruded(fp) for fp in feet_points)

                prev = s.state
                update_seat_fsm(s, any_core, dt, dwellSeconds=DWELL_SECONDS, exitSeconds=EXIT_SECONDS)

                # === 상태/HUD 표시 (요청사항: 7초까지 카운트) ===
                if s.state in ["OUTSIDE", "ENTERING"]:
                    draw_text(
                        vis,
                        f"Seat {i} [{s.state}] {s.dwell_s:.1f}s / {DWELL_SECONDS:.1f}s",
                        (s.p1[0], s.p1[1]+40),
                        (0, 200, 255), 0.6, 2
                    )
                elif s.state == "INTRUDED":
                    draw_text(
                        vis,
                        f"Seat {i} INTRUDED ({s.dwell_s:.1f}s)",
                        (s.p1[0], s.p1[1]+40),
                        (0, 0, 255), 0.7, 2
                    )

                # 상태 전이가 'INTRUDED'로 바뀌는 순간 1회 이벤트
                if prev != "INTRUDED" and s.state == "INTRUDED":
                    print(f"[EVENT] {time.strftime('%H:%M:%S')}  SEAT#{i} INTRUSION")

                if s.state == "INTRUDED":
                    intruded_seats.add(i-1)

            if intruded_seats:
                draw_text(vis, "INTRUSION!", (10, h-20), (0,0,255), 0.9, 3)

            # help
            if self.help_on:
                draw_text(vis, "a:Add seat (click 2pts) | t:flip inward | x:delete last | s:save | l:load | d:debug | h:help | q:quit",
                          (10, 26), (30,220,30), 0.55, 2)

            cv2.imshow(win, vis)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            elif k == ord('a'):
                self.add_mode = True; self.temp_pts.clear()
                print("[MODE] click two points along the yellow line")
            elif k == ord('t'):
                if self.seats:
                    self.seats[-1].inward_sign *= -1
                    print(f"[TOGGLE] seat#{len(self.seats)} inward_sign -> {self.seats[-1].inward_sign}")
            elif k == ord('x'):
                if self.seats:
                    removed = self.seats.pop()
                    print(f"[DEL] removed seat {removed.p1}->{removed.p2}")
            elif k == ord('s'):
                self.save(self.args.config)
            elif k == ord('l'):
                self.load(self.args.config)
            elif k == ord('d'):
                self.debug = not self.debug
            elif k == ord('h'):
                self.help_on = not self.help_on

        cap.release(); cv2.destroyAllWindows()

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default="0")
    ap.add_argument("--model", type=str, default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default="0.4")
    ap.add_argument("--camera_K", type=str, default="")
    ap.add_argument("--camera_dist", type=str, default="")
    ap.add_argument("--config", type=str, default="tripwire_perp.json")
    return ap.parse_args()

if __name__ == "__main__":
    TripwireApp(parse_args()).run()
