# app/models/photo_adapter.py
import time, cv2
from ultralytics import YOLO
from app.models import photo as legacy  # 기존 photo.py를 그대로 재사용

class PhoneBackDetector:
    """
    기존 photo.py의 하드코딩(WEIGHTS, CONF_THRES, PERSIST_SEC, HOLD_SEC, CAMERA_INDEX)을
    그대로 기본값으로 가져와서 사용. 호출자는 필요할 때만 덮어쓰기 가능.
    """
    def __init__(self,
                 weights=None,
                 cam_source=None,
                 conf=None,
                 dwell=None,
                 hold=None,
                 imgsz=640):
        self.weights = weights if weights is not None else legacy.WEIGHTS
        self.cam_source = cam_source if cam_source is not None else legacy.CAMERA_INDEX
        self.conf = conf if conf is not None else legacy.CONF_THRES
        self.dwell = dwell if dwell is not None else legacy.PERSIST_SEC
        self.hold = hold if hold is not None else legacy.HOLD_SEC
        self.imgsz = imgsz

        self.model = YOLO(self.weights)

    def scan(self, timeout_sec: float = 3.0) -> bool:
        """
        timeout_sec 동안 카메라에서 프레임을 읽어
        기존 로직(연속 dwell ≥ self.dwell)으로 '촬영포즈' 감지 여부만 True/False로 반환.
        화면 출력/키입력은 없음(서버 연동용).
        """
        cap = cv2.VideoCapture(self.cam_source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"[PhoneBackDetector] 카메라 {self.cam_source} 를 열 수 없습니다.")
            return False

        dwell_start = None
        trigger_until = 0.0
        deadline = time.time() + float(timeout_sec)
        detected_flag = False

        try:
            while time.time() < deadline:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue

                r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
                has_box = (r is not None and r.boxes is not None and len(r.boxes) > 0)
                now = time.time()

                if has_box:
                    if dwell_start is None:
                        dwell_start = now
                    elif (now - dwell_start) >= self.dwell and now > trigger_until:
                        trigger_until = now + self.hold
                        detected_flag = True
                        break
                else:
                    dwell_start = None

                time.sleep(0.02)
        finally:
            cap.release()
            # 화면창을 띄우지 않으므로 destroyAllWindows 불필요

        return detected_flag
