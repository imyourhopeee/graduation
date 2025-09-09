# app/models/blur.py
import os
import numpy as np
import cv2 as cv
from ultralytics import YOLO
from typing import List, Tuple

# === (원본과 동일) 클래스 이름 — 실제 사용 X ===
CLASS_NAMES = {
    0: "tv",
    1: "laptop",
    2: "cell phone"
}

# === (원본과 동일) 시각화/블러 클래스 ===
class Visualizer:
    def __init__(self, is_obb=False):
        self.is_obb = is_obb

    def id_to_color(self, id):
        np.random.seed(id)
        return tuple(np.random.randint(0, 255, size=3).tolist())

    def plot_box_on_img(
        self,
        img: np.ndarray,
        box: tuple,
        conf: float,
        cls: int,
        id: int,
        thickness: int = 2,
        fontscale: float = 0.5,
    ) -> np.ndarray:
        # --- 좌표 읽기 ---
        x1, y1, x2, y2 = map(int, box)

        # --- 좌표 뒤집힘 보정 ---
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1

        # --- 프레임 경계로 클램프 ---
        H, W = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        # --- 블러 처리: 커스텀(0,1,2) + COCO(62,63,67) 모두 허용 ---
        if int(cls) in [0, 1, 2, 62, 63, 67]:
            w, h = x2 - x1, y2 - y1
            if w > 1 and h > 1:
                # 약간의 마진 추가(경계 자연스럽게)
                mx, my = int(0.05 * w), int(0.05 * h)
                xx1, yy1 = max(0, x1 - mx), max(0, y1 - my)
                xx2, yy2 = min(W, x2 + mx), min(H, y2 + my)

                roi = img[yy1:yy2, xx1:xx2]
                # ROI 크기에 비례한 가변 커널 (항상 홀수)
                k = max(15, ((max(w, h) // 10) * 2 + 1))
                blurred = cv.GaussianBlur(roi, (k, k), 0)
                img[yy1:yy2, xx1:xx2] = blurred

        # --- 텍스트 출력 제거 (id/class/conf 표시 안 함) ---
        return img

# === 변경점: 모듈화된 엔진 클래스만 추가 (웹캠/창 표시는 없음) ===
class BlurEngine:
    """YOLO를 1회 로드하고, 매 프레임 블러만 수행."""
    def __init__(self, model_path: str | None = None, conf: float = 0.5):
        self.conf = conf
        self.model = YOLO(model_path or os.getenv("BLUR_MODEL", "runs/detect/train11/weights/best.pt"))
        self.viz = Visualizer(is_obb=False)

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, List[Tuple[int,int,int,int]]]:
        """
        입력: frame (BGR, np.ndarray)
        출력: (블러가 적용된 frame, 감지된 bbox 리스트[(x1,y1,x2,y2), ...])
        """
        bboxes: List[Tuple[int,int,int,int]] = []
        results = self.model(frame, conf=self.conf)[0]  # 원본과 동일 conf default=0.5
        if results and results.boxes is not None:
            for i, det in enumerate(results.boxes):
                box = det.xyxy[0].cpu().numpy()
                conf = float(det.conf[0])
                cls  = int(det.cls[0])
                # 원본과 동일한 블러 적용
                frame = self.viz.plot_box_on_img(frame, box, conf, cls, i)
                x1, y1, x2, y2 = map(int, box)
                bboxes.append((x1, y1, x2, y2))
        return frame, bboxes
