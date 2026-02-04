import cv2

for idx in range(3):  # 0,1,2 까지 테스트
    cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)  # MSMF 백엔드 강제
    if cap.isOpened():
        print("✅ Camera index", idx, "opened OK")
        cap.release()
    else:
        print("❌ Camera index", idx, "cannot open")
