import os, time, cv2, numpy as np
from collections import deque, Counter

# 네가 준 face_rec.py의 전역 객체/함수들을 그대로 활용
# (모델을 다시 로드하지 않으니 빠르고 동일한 추론 경로)
from app.models import face_rec as FR  # <-- face_rec.py 그대로 둔 상태여야 함

class FaceRecognizer:
    """
    CAM1(모니터 웹캠)에서 timeout_sec 동안 얼굴을 탐지/식별하고
    가장 안정적인 라벨과 점수를 반환한다. (GUI/창 없이)
    """
    def __init__(self, cam_source="1",
                 conf=0.5, dist_thr=0.6,
                 min_stable=3, buffer_len=7):
        self.src = os.getenv("CAM1_SOURCE", cam_source)
        self.conf = float(os.getenv("FACE_CONF", conf))
        self.dist_thr = float(os.getenv("FACE_DIST_THR", dist_thr))
        self.min_stable = int(os.getenv("FACE_MIN_STABLE", min_stable))
        self.buffer_len = int(os.getenv("FACE_BUFFER_LEN", buffer_len))

    def recognize(self, timeout_sec=3.0):
        # 카메라 오픈
        cap = cv2.VideoCapture(int(self.src), cv2.CAP_DSHOW) if str(self.src).isdigit() else cv2.VideoCapture(self.src)
        if not cap.isOpened():
            return None, None

        deadline = time.time() + float(timeout_sec)

        # 라벨 안정화 버퍼
        labels_buf = deque(maxlen=self.buffer_len)
        scores_buf = deque(maxlen=self.buffer_len)

        best_label, best_score = None, 0.0

        while time.time() < deadline:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01); continue

            H, W = frame.shape[:2]

            # YOLO 얼굴 검출 (face 모델)
            results = FR.yolo_model(frame, verbose=False)[0]
            faces = []
            for box in results.boxes:
                conf = float(box.conf)
                if conf < self.conf:  # 기본 0.5
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(W-1, x2); y2 = min(H-1, y2)
                if x2 <= x1 or y2 <= y1: 
                    continue
                faces.append((x1, y1, x2, y2))

            # 가장 큰 얼굴 1개만 사용(간단 안정화)
            if faces:
                x1, y1, x2, y2 = max(faces, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                # 정렬 + 임베딩
                pil = FR.Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                pil = FR.align_face(pil)
                t = FR.transform(pil).unsqueeze(0).to(FR.device)
                with FR.torch.no_grad():
                    emb = FR.facenet(t).cpu().numpy().flatten()
                emb = FR.l2_normalize(emb)

                # KNN 추론 (거리 기반 Unknown)
                distances, idxs = FR.knn_clf.kneighbors([emb], n_neighbors=2, return_distance=True)
                d1 = float(distances[0][0])
                pred_label = FR.knn_clf.predict([emb])[0]
                if d1 > self.dist_thr:
                    pred_label = "Unknown"
                    score = 0.0
                else:
                    score = 1.0 / (1.0 + d1)

                labels_buf.append(pred_label)
                scores_buf.append(score)

                # 즉시 최고값 갱신
                if pred_label != "Unknown" and score > best_score:
                    best_label, best_score = pred_label, score

            time.sleep(0.01)

        cap.release()

        # 다수결 + 평균 점수로 최종 안정화
        if len(labels_buf) >= self.min_stable:
            counts = Counter([l for l in labels_buf if l and l != "Unknown"])
            if counts:
                winner = counts.most_common(1)[0][0]
                mean_score = float(np.mean([s for l, s in zip(labels_buf, scores_buf) if l == winner]))
                return winner, mean_score

        # 백업: 스캔 중 최고값
        return (best_label, best_score) if best_label else (None, None)
