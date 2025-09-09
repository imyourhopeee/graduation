import time
import cv2
from ultralytics import YOLO

# ===== 설정 =====
WEIGHTS = "C:/Users/DS/Documents/graduation/SOAT_main/runs/detect/train12/weights/best.pt"
CAMERA_INDEX = 1
CONF_THRES = 0.8        # 감지 신뢰도 기준 (이 이상이면 감지로 인정)
PERSIST_SEC = 1.5       # 1.5초 연속 감지 시 이벤트 발생
HOLD_SEC = 2.0          # 이벤트 발생 후 문구 유지 시간(초)
FONT = cv2.FONT_HERSHEY_SIMPLEX

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

    dwell_start = None   # 감지가 시작된 시각
    trigger_until = 0.0  # 이벤트 문구를 표시할 종료 시각
    last_conf = 0.0      # 최근 프레임의 최대 confidence

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            # YOLO 추론
            det = model.predict(frame, verbose=False)[0]

            # 박스가 있다면 최대 confidence 추출
            if det.boxes is not None and len(det.boxes) > 0:
                last_conf = float(det.boxes.conf.max().item())
            else:
                last_conf = 0.0

            # 감지 여부: conf가 기준 이상일 때만 True
            detected = last_conf >= CONF_THRES
            now = time.time()

            # ---------------- 조건문 동작 ----------------
            if detected:
                if dwell_start is None:
                    dwell_start = now
                if (now - dwell_start) >= PERSIST_SEC and now > trigger_until:
                    trigger_until = now + HOLD_SEC
                    print("[EVENT] 촬영포즈 1.5초 연속 감지 → 촬영 되었습니다!")
            else:
                dwell_start = None

            # ---------------- 화면 표시 ----------------
            if dwell_start is not None and now < trigger_until:
                # 이벤트 발생 상태
                draw_center_text(frame, f"Photo Captured (conf {last_conf:.2f})")
            elif dwell_start is not None:
                # 이벤트 전 카운트다운
                remain = max(0.0, PERSIST_SEC - (now - dwell_start))
                txt = f"Capturing pose... {remain:.1f}s left (conf {last_conf:.2f})"
                cv2.putText(frame, txt, (20, 40), FONT, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, txt, (20, 40), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            elif now < trigger_until:
                # 이벤트 유지 시간
                draw_center_text(frame, f"Photo Captured (conf {last_conf:.2f})")
            else:
                # 감지도 없고 이벤트도 없을 때 → conf만 표시
                txt = f"Current conf {last_conf:.2f}"
                cv2.putText(frame, txt, (20, 40), FONT, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, txt, (20, 40), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # 종료 안내
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
