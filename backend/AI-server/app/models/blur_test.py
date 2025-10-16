#blur_visualize_test 최종
import cv2 as cv
import numpy as np
from ultralytics import YOLO
import cv2
from collections import deque

# 클래스 이름 정의 (커스텀 모델 기준) — 현재 텍스트 비표시라 실사용은 없음
CLASS_NAMES = {
    0: "tv",
    1: "laptop",
    2: "cell phone"
}

# id별 색상 고정 - seed로 색상 생성 (텍스트 미표시라 실사용은 없음)
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

# 메인 함수 - 모델 로드 및 웹캠 입력 받기
def main():
    model = YOLO("../runs/detect/train11/weights/best.pt")  # 경로 조정
    # model = YOLO("C:/Users/DS/Documents/graduation/SOAT_main/yolov8n.pt")  # COCO 가중치 사용할 때

    # cap = cv.VideoCapture("runs/detect/test5.mp4")  # 영상 파일 테스트
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 웹캠

    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return

    visualizer = Visualizer(is_obb=False)
    recent_boxes = deque(maxlen=5)  # 최근 5프레임 박스 저장 (현재 미사용)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ✅ conf=0.5 적용 (0.5 미만은 모델에서 걸러짐)
        results = model(frame, conf=0.4)[0]

        # 감지된 결과들 각각 처리
        for i, det in enumerate(results.boxes):
            box = det.xyxy[0].cpu().numpy()
            conf = float(det.conf[0])
            cls = int(det.cls[0])
            track_id = i  # 임시 ID (추적 아님)
            frame = visualizer.plot_box_on_img(frame, box, conf, cls, track_id)

        # 🔹 비율 유지 확대 - 화면 창 키워보려고
        h, w = frame.shape[:2]
        new_w = int(w * 2)      # 가로 2배
        new_h = int(1.5 * h)    # 세로 1.5배
        frame = cv.resize(frame, (new_w, new_h), interpolation=cv.INTER_LINEAR)

        cv.imshow("Blurred Detection", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
