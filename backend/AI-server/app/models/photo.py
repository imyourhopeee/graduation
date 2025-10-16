# ====== 파일: app/models/photo.py ======
import time
import cv2
from ultralytics import YOLO

# ===== 설정 =====
WEIGHTS = "C:/Users/DS/Documents/graduation/SOAT_main/runs/detect/train17/weights/best.pt"
CAMERA_INDEX = 1
CONF_THRES = 0.65        # 감지 신뢰도 기준 (이 이상이면 감지로 인정)
PERSIST_SEC = 1.0        # 연속 감지 유지 시간
HOLD_SEC = 2.0
FONT = cv2.FONT_HERSHEY_SIMPLEX

# OpenCV 창 크기 설정 (수동 테스트용)
WINDOW_W, WINDOW_H = 1280, 960

# ================== ⬇⬇⬇ 추가: 라이브러리용 감지 클래스 ⬇⬇⬇ ==================
class PhoneBackDetector:
    """
    오케스트레이터(inference.py)에서 임포트되어 사용되는 경량 감지기.
    - 화면 출력/윈도우 없음
    - YOLO 한 번만 로드
    - timeout 동안 감시 → CONF_THRES 이상 감지 상태가 PERSIST_SEC 연속이면 True
    """
    def __init__(self, weights: str = WEIGHTS, conf_thr: float = CONF_THRES, persist_sec: float = PERSIST_SEC):
        self.model = YOLO(weights)
        self.conf_thr = float(conf_thr)
        self.persist_sec = float(persist_sec)
        #디버깅용
        print(f"[PhoneBackDetector] ✅ Loaded model '{weights}' (conf={self.conf_thr}, persist={self.persist_sec}s)")

    def scan(self, timeout_sec: float, cam_index: int | None = None) -> bool:
        """timeout 동안 감시하여 '촬영 포즈' 성립 시 True, 아니면 False"""
        idx = CAMERA_INDEX if cam_index is None else int(cam_index)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            # 카메라가 안 열려도 False로 안전 복귀 (오케스트레이터 쪽은 예외 없이 계속 동작)
            print(f"[PhoneBackDetector] ❌ Failed to open camera index {idx}")
            return False

        t0 = time.time()
        dwell_start = None

        try:
            while (time.time() - t0) < float(timeout_sec):
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue

                # YOLO 추론
                det = self.model.predict(frame, verbose=False)[0]

                # 최대 confidence 추출
                if det.boxes is not None and len(det.boxes) > 0:
                    last_conf = float(det.boxes.conf.max().item())
                else:
                    last_conf = 0.0

                detected = (last_conf >= self.conf_thr)
                now = time.time()

                if detected:
                    if dwell_start is None:
                        dwell_start = now
                    # 연속 구간이 PERSIST_SEC 이상이면 성공
                    if (now - dwell_start) >= self.persist_sec:
                        # ✅ 디버깅용 콘솔 로그 추가
                        print(f"[PhoneBackDetector] 📸 Photo pose detected! (conf={last_conf:.2f}, held {now - dwell_start:.2f}s)")
                        return True
                else:
                    dwell_start = None

            # timeout 종료 → 실패
            print("[PhoneBackDetector] ⏰ Timeout reached → No photo pose detected.")
            return False
        finally:
            cap.release()
# ================== ⬆⬆⬆ 추가 끝 ⬆⬆⬆ ==================


def draw_center_text(img, text, scale=1.6, thickness=4, color=(40, 220, 40)):
    """중앙에 강조된 텍스트를 그림 (OpenCV 화면용)"""
    h, w = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    org = (int((w - tw) / 2), int((h + th) / 2))
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)

def main():
    model = YOLO(WEIGHTS)
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERR] 카메라 {CAMERA_INDEX} 를 열 수 없습니다.")
        return

    dwell_start = None
    trigger_until = 0.0
    last_conf = 0.0

    cv2.namedWindow("Phone-Back Pose (cam 1)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Phone-Back Pose (cam 1)", WINDOW_W, WINDOW_H)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            det = model.predict(frame, verbose=False)[0]
            if det.boxes is not None and len(det.boxes) > 0:
                last_conf = float(det.boxes.conf.max().item())
            else:
                last_conf = 0.0

            detected = last_conf >= CONF_THRES
            now = time.time()

            if detected:
                if dwell_start is None:
                    dwell_start = now
                if (now - dwell_start) >= PERSIST_SEC and now > trigger_until:
                    trigger_until = now + HOLD_SEC
                    print("[EVENT] 촬영포즈 1.5초 연속 감지 → 촬영 되었습니다!")
            else:
                dwell_start = None

            if dwell_start is not None and now < trigger_until:
                draw_center_text(frame, f"Photo Captured")
            elif dwell_start is not None:
                remain = max(0.0, PERSIST_SEC - (now - dwell_start))
                txt = f"Capturing pose... {remain:.1f}s left (conf {last_conf:.2f})"
                cv2.putText(frame, txt, (20, 40), FONT, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, txt, (20, 40), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            elif now < trigger_until:
                draw_center_text(frame, f"Photo Captured")
            else:
                txt = f"Current conf {last_conf:.2f}"
                cv2.putText(frame, txt, (20, 40), FONT, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, txt, (20, 40), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(frame, "Press 'q' to quit", (20, frame.shape[0] - 20),
                        FONT, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, "Press 'q' to quit", (20, frame.shape[0] - 20),
                        FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Phone-Back Pose (cam 1)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
