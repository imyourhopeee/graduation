# app/models/blur.py
from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

__all__ = ["BlurEngine", "run_inference_on_image"]

# ===== 설정 =====
BLUR_YOLO_WEIGHTS = os.getenv(
    "BLUR_YOLO_WEIGHTS",
    r"C:/Users/DS/Documents/graduation/SOAT_main/runs/detect/train11/weights/best.pt"
)
PIXELATE = 0  # 0이면 가우시안, 아래에서 conf 설정
CONF_THR = float(os.getenv("BLUR_CONF", "0.3"))
TARGET_LABELS = set((os.getenv("BLUR_LABELS", "cell phone,laptop,tv")).split(","))

# ===== 내부 유틸 =====
def _pixelate_roi(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, block: int = 16):
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    h, w = roi.shape[:2]
    bw = max(1, block)
    small = cv2.resize(roi, (max(1, w // bw), max(1, h // bw)), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    img[y1:y2, x1:x2] = pixelated

def _gaussian_blur_roi(img: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    k = max(9, ((min(roi.shape[0], roi.shape[1]) // 10) * 2 + 1))
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    img[y1:y2, x1:x2] = blurred

# ===== 클래스 인터페이스 (inference.py 호환) =====
class BlurEngine:
    """
    Lazy YOLO loader + blur pipeline.
    inference.py가 기대하는 클래스를 충족하기 위해 제공.
    """
    def __init__(
        self,
        weights: str | None = None,
        conf_thr: float = CONF_THR,
        pixelate: int = PIXELATE,
        target_labels: set[str] | None = None,
    ):
        self.weights = Path(weights) if weights else Path(BLUR_YOLO_WEIGHTS)
        self.conf_thr = conf_thr
        self.pixelate = pixelate
        self.target_labels = set(target_labels) if target_labels else set(TARGET_LABELS)
        self._model: YOLO | None = None

    def _get_model(self) -> YOLO:
        if self._model is not None:
            return self._model
        if YOLO is None:
            raise RuntimeError("ultralytics 가 필요합니다. (pip install ultralytics)")
        if not self.weights.exists():
            raise FileNotFoundError(f"YOLO 가중치 파일이 없습니다: {self.weights}")
        self._model = YOLO(str(self.weights))
        return self._model

    def process(self, frame_bgr: np.ndarray, return_boxes: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """
        frame -> (blurred_frame, dets)
        dets: [{"label": str, "conf": float, "bbox": [x1,y1,x2,y2]}, ...]
        """
        model = self._get_model()
        res = model(frame_bgr, conf=self.conf_thr, verbose=False)
        dets: List[Dict] = []
        if not res or res[0].boxes is None:
            return frame_bgr, dets

        names = res[0].names if hasattr(res[0], "names") else {}
        H, W = frame_bgr.shape[:2]

        for b in res[0].boxes:
            conf = float(b.conf)
            cls_id = int(b.cls) if b.cls is not None else -1
            label = names.get(cls_id, str(cls_id))
            if label not in self.target_labels:
                continue

            x1, y1, x2, y2 = map(int, b.xyxy[0])
            x1 = max(0, min(W - 1, x1)); y1 = max(0, min(H - 1, y1))
            x2 = max(0, min(W - 1, x2)); y2 = max(0, min(H - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            
            # 좌표 보정 끝난 직후
            w, h = x2 - x1, y2 - y1
            mx, my = int(0.05 * w), int(0.05 * h)  # 5% 마진
            xx1, yy1 = max(0, x1 - mx), max(0, y1 - my)
            xx2, yy2 = min(W - 1, x2 + mx), min(H - 1, y2 + my)

            # 이제 블러는 이 좌표(xx1..yy2)로 적용
            _gaussian_blur_roi(frame_bgr, xx1, yy1, xx2, yy2)

            if return_boxes:
                dets.append({"label": label, "conf": conf, "bbox": [x1, y1, x2, y2]})

        return frame_bgr, dets

# ===== 함수 인터페이스 (stream.py가 함수형을 쓸 수도 있어서 유지) =====
# 내부적으로 싱글톤 엔진을 사용
_engine_singleton: BlurEngine | None = None

def _get_engine() -> BlurEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = BlurEngine()
    return _engine_singleton

def run_inference_on_image(frame_bgr: np.ndarray, return_boxes: bool = True) -> Tuple[np.ndarray, List[Dict]]:
    return _get_engine().process(frame_bgr, return_boxes=return_boxes)
