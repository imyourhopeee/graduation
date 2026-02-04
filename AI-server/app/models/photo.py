# ====== 파일: app/models/photo.py ======
"""
프레임 스트리밍 연동형 PhoneBackDetector (카메라 직접 오픈 제거 버전)

목표
- face_rec.py(또는 다른 모듈)에서 이미 열어 둔 '웹캠 프레임'을 그대로 받아 촬영 포즈를 동시에 감지
- 별도 창/표시 없음, YOLO는 1회 로드
- conf 이상 상태가 persist_sec 이상 '연속' 유지되면 촬영(True) 판정

사용 패턴:
-------------------------------------------------
from app.models.photo import PhoneBackDetector

det = PhoneBackDetector(weights=..., conf_thr=0.65, persist_sec=1.0)

# face_rec의 프레임 루프 내부:
ok, frame = cap.read()
pose = det.update(frame)   # 매 프레임 호출
if pose.captured:
    # 촬영 포즈 성립 처리
    ...

# (옵션) 일정 시간 동안 감시하고 싶다면 frame_supplier로만 사용:
ok = det.scan(timeout_sec=5.0, frame_supplier=lambda: latest_frame_or_None)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Callable, Any

from ultralytics import YOLO

# ===== 기본 설정 (필요시 .env로 치환 가능) =====
WEIGHTS     = "C:/Users/DS/Documents/graduation/SOAT_main/runs/detect/train18/weights/best.pt"
CONF_THRES  = 0.65      # 감지 신뢰도 기준
PERSIST_SEC = 1.0       # '연속 감지' 유지 시간

# --------------------------------------------
# (1) 결과 스냅샷
# --------------------------------------------
@dataclass
class PoseResult:
    conf: float          # 이번 프레임 최대 confidence
    detected: bool       # conf >= conf_thr
    held_sec: float      # 연속 감지 누적 시간(초)
    captured: bool       # 연속 감지 시간이 persist_sec 이상
    ts: float            # 판정 시각 (time.time)

# --------------------------------------------
# (2) 본체
# --------------------------------------------
class PhoneBackDetector:
    """
    - YOLO 가중치 1회 로드
    - update(frame): 프레임 1장에 대해 촬영 포즈 판정(누적 타이머 기반)
    - reset(): 상태 초기화
    - scan(): (옵션) frame_supplier로만 일정 시간 감시 (카메라 직접 오픈 없음)
    """
    def __init__(self, weights: str = WEIGHTS, conf_thr: float = CONF_THRES, persist_sec: float = PERSIST_SEC):
        self.model = YOLO(weights)
        self.conf_thr = float(conf_thr)
        self.persist_sec = float(persist_sec)

        # 연속 감지 시작 시각 (미감지면 None)
        self._dwell_start: Optional[float] = None
        self._last_ts: float = 0.0

        print(f"[PhoneBackDetector] ✅ Model loaded: '{weights}' "
              f"(conf_thr={self.conf_thr}, persist_sec={self.persist_sec})")

    # ---- 내부 유틸: 프레임 → 최대 conf ----
    def _max_conf_from_frame(self, frame: Any) -> float:
        """
        단일 프레임에 대한 YOLO 추론을 수행하고,
        탐지 박스들 중 최대 confidence를 반환 (없으면 0.0)
        """
        det = self.model.predict(frame, verbose=False)[0]
        if det.boxes is not None and len(det.boxes) > 0:
            return float(det.boxes.conf.max().item())
        return 0.0

    # ---- 핵심 API: 스트리밍 프레임 1장 처리 ----
    def update(self, frame: Any, ts: Optional[float] = None) -> PoseResult:
        """
        face_rec.py의 루프에서 매 프레임 호출.
        반환: PoseResult(conf, detected, held_sec, captured, ts)
        """
        now = time.time() if ts is None else float(ts)
        conf = self._max_conf_from_frame(frame)
        detected = (conf >= self.conf_thr)

        if detected:
            if self._dwell_start is None:
                self._dwell_start = now
            held = now - self._dwell_start
        else:
            self._dwell_start = None
            held = 0.0

        captured = (detected and held >= self.persist_sec)
        self._last_ts = now
        return PoseResult(conf=conf, detected=detected, held_sec=held, captured=captured, ts=now)

    # ---- 상태 초기화 ----
    def reset(self) -> None:
        """연속 감지 타이머 등 내부 상태 초기화."""
        self._dwell_start = None
        self._last_ts = 0.0

    # ---- (옵션) 프레임 공급자 기반 감시 (카메라 직접 오픈 제거) ----
    def scan(
        self,
        timeout_sec: float,
        frame_supplier: Optional[Callable[[], Optional[Any]]] = None
    ) -> bool:
        """
        timeout 동안 촬영 포즈 성립(True) 여부를 반환.
        - frame_supplier: 콜러블; 호출 시 최신 프레임(ndarray) 또는 None 반환
        - 주의: 카메라를 직접 열지 않으며, frame_supplier가 없으면 예외 발생
        """
        if not callable(frame_supplier):
            raise ValueError("frame_supplier callable is required (camera auto-open is not supported).")

        t0 = time.time()
        self.reset()

        while (time.time() - t0) < float(timeout_sec):
            frame = frame_supplier()
            if frame is None:
                time.sleep(0.01)
                continue

            res = self.update(frame)
            if res.captured:
                print(f"[PhoneBackDetector] 📸 via supplier (conf={res.conf:.2f}, held={res.held_sec:.2f}s)")
                return True

            # CPU 과점유 방지
            time.sleep(0.005)

        print("[PhoneBackDetector] ⏰ Timeout via supplier → No photo pose detected.")
        return False
