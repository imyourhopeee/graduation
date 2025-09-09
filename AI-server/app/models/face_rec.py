import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import joblib
from ultralytics import YOLO
import face_recognition
from collections import deque, Counter

# 딥소트 임포트 위해 추가
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 가중치 경로 인식 

from deep_sort import rpath
FACENET_WEIGHTS = rpath("facenet_triplet_hard_pk.pth")

knn_clf = joblib.load("C:/Users/DS/Documents/graduation/SOAT_main/backend/AI-server/deep_sort/knn_face_classifier_pk.pkl")

#딥소트 인식 
from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.detection import Detection


# 이 코드에서 라벨 결정 방식 보완 ing
# ===================== 기본 세팅 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FaceNet (하드마이닝 학습 가중치)
facenet = InceptionResnetV1(pretrained=None, classify=False).to(device)
state = torch.load("C:/Users/DS/Documents/graduation/SOAT_main/backend/AI-server/deep_sort/facenet_triplet_hard_pk.pth", map_location=device)  # ← 절대경로
facenet.load_state_dict(state, strict=False)
facenet.eval()

# KNN .pkl
from deep_sort import rpath  # 이미 있다면 재사용
KNN_PATH = rpath("knn_face_classifier_pk.pkl")
knn_clf = joblib.load(str(KNN_PATH))

print("[KNN]", KNN_PATH, "exists:", KNN_PATH.exists())

# YOLO 가중치
yolo_model = YOLO("C:/Users/DS/Documents/graduation/SOAT_main/backend/AI-server/deep_sort/yolov8n-face.pt")


# 전처리
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# DeepSORT 세팅
# max_cosine_distance는 appearance 유사도 임계값. 0.2~0.4 사이에서 튜닝

# metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, None)
# deepsort_tracker = tracker.Tracker(metric)
# 교체->
metric = NearestNeighborDistanceMetric("cosine", 0.2, None)
deepsort_tracker = Tracker(metric)


# ===================== 유틸 =====================
# 눈 위치로 회전 보정 - 얼굴 정렬
def align_face(pil_img: Image.Image) -> Image.Image:
    np_img = np.array(pil_img)
    lms = face_recognition.face_landmarks(np_img)
    if not lms: return pil_img
    lm = lms[0]
    if 'left_eye' not in lm or 'right_eye' not in lm:
        return pil_img
    left_eye = np.mean(lm['left_eye'], axis=0)
    right_eye = np.mean(lm['right_eye'], axis=0)
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    eye_center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
    h, w = np_img.shape[:2]
    M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    aligned = cv2.warpAffine(np_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(aligned)

# 검출 박스를 비율만큼 확장
def expand_box(x1, y1, x2, y2, scale, W, H):
    cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
    w = (x2 - x1) * scale; h = (y2 - y1) * scale
    nx1 = int(max(0, cx - w/2)); ny1 = int(max(0, cy - h/2))
    nx2 = int(min(W-1, cx + w/2)); ny2 = int(min(H-1, cy + h/2))
    return nx1, ny1, nx2, ny2

#임베딩 L2 정규화 
def l2_normalize(v):
    n = np.linalg.norm(v) + 1e-8
    return v / n

#두 박스의 ioU 계산 , 교집합 합집합 면적 계산
def iou(bb_test, bb_gt):
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[0] + bb_test[2], bb_gt[0] + bb_gt[2])
    yy2 = min(bb_test[1] + bb_test[3], bb_gt[1] + bb_gt[3])
    w = max(0., xx2 - xx1); h = max(0., yy2 - yy1)
    inter = w * h
    union = bb_test[2]*bb_test[3] + bb_gt[2]*bb_gt[3] - inter
    return inter/union if union > 0 else 0

# ===================== 메인 루프 =====================
def main():

    # #영상으로 입력받기
    # video_path = "C:/Users/DS/Downloads/유진외2인중복.mp4"
    # cap = cv2.VideoCapture(video_path)
    # if not cap.isOpened():
    #     print(f"Error: Could not open video file {video_path}")
    #     return

    #웹캠으로 입력받기
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # 0 = 기본 웹캠
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    #웹캠 - 해상도 부분 (선택)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 


    # KNN distance 임계값 (작을수록 유사); 
    # 여기서 가장 비슷한 애랑 거리가 0.80 차이나면 unknown인 거임 현재!!!!!!!!!!!
    DIST_THR = 0.6

    # 라벨 안정화
    BUFFER = 7
    MIN_STABLE = 3
    SWITCH_MARGIN = 0.15  # top-2 margin 보수화에 쓰고 싶으면 확장 가능

    # ★ ADD: 배타 재할당 파라미터(2순위 허용/스틸 기준)
    MIN_ACCEPT = 0.55     # 2순위로 갈아탈 때 최소 신뢰도
    STEAL_MARGIN = 0.03   # 이미 배정된 라벨을 빼앗으려면 이만큼 더 높아야 함

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        H, W = frame.shape[:2]

        # YOLO 얼굴 탐지
        results = yolo_model(frame, verbose=False)[0]
        detections = []

        # 현재 프레임의 det 정보(박스, 라벨 등) 저장해 추적 후 트랙에 붙일 것
        det_pack = []  # (tlwh, label, score)
        for box in results.boxes:
            conf = float(box.conf)
            if conf < 0.5: 
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, scale=1.25, W=W, H=H)
            if x2 <= x1 or y2 <= y1: 
                continue

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            try:
                # 정렬 + 임베딩
                pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                pil = align_face(pil)
                t = transform(pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = facenet(t).cpu().numpy().flatten()
                emb = l2_normalize(emb)

                # KNN 추론
                distances, idxs = knn_clf.kneighbors([emb], n_neighbors=2, return_distance=True)
                d1, d2 = distances[0]
                pred_label = knn_clf.predict([emb])[0]
                # 거리 기반 Unknown 처리
                if d1 > DIST_THR:
                    pred_label = "Unknown"

                # DeepSORT detection 객체 생성 (tlwh, confidence, feature)
                tlwh = [x1, y1, x2 - x1, y2 - y1]
                detections.append(Detection(tlwh, conf, emb))
                det_pack.append((tlwh, pred_label, 1.0/(1.0 + d1)))  # score는 거리 역수로 간단 표기
            except Exception as e:
                # 실패시 무시
                continue

        # DeepSORT 추적 업데이트
        deepsort_tracker.predict()
        deepsort_tracker.update(detections)

        # ★ ADD: 이번 프레임에서 그리기 전에 트랙별 표시 후보를 먼저 모은다.
        proposals = []  # 각 원소: dict(track_id, tlwh, label, score, alt_label, alt_score, _status)

        # 트랙별 라벨 버퍼/히스테리시스
        for track_obj in deepsort_tracker.tracks:
            if not track_obj.is_confirmed() or track_obj.time_since_update > 1:
                continue

            # 트랙 초기 안정화: 너무 어린 트랙은 패스
            if track_obj.age < 5:
                continue

            # 트랙에 버퍼가 없으면 생성
            if not hasattr(track_obj, "label_buffer"):
                track_obj.label_buffer = deque(maxlen=BUFFER)
                track_obj.score_buffer = deque(maxlen=BUFFER)
                track_obj.current_label = None

            # 이번 프레임에서 이 트랙과 IoU가 가장 큰 detection 라벨을 버퍼에 추가
            t_tlwh = track_obj.to_tlwh()
            best = None; best_iou = 0.0
            for tlwh, lab, scr in det_pack:
                i = iou(t_tlwh, tlwh)
                if i > best_iou:
                    best_iou = i
                    best = (lab, scr)
            if best and best_iou > 0.2:
                track_obj.label_buffer.append(best[0])
                track_obj.score_buffer.append(best[1])

            name_to_draw = ""
            avg_score = 0.0
            if len(track_obj.label_buffer) >= MIN_STABLE:

                # 다수결 + 평균점수
                mc = Counter(track_obj.label_buffer).most_common(1)[0][0]
                avg_score = np.mean([s for l, s in zip(track_obj.label_buffer, track_obj.score_buffer) if l == mc]) if track_obj.score_buffer else 0.0

                # 히스테리시스: 라벨이 바뀌려면 평균점수도 일정 이상일 때만
                if track_obj.current_label is None:
                    track_obj.current_label = mc
                    track_obj.current_score = avg_score
                else:
                    if (mc != track_obj.current_label) and (avg_score >= max(0.55, track_obj.current_score + 0.05)):
                        track_obj.current_label = mc
                        track_obj.current_score = avg_score
                    else:
                        # 유지하면서 점수는 천천히 업데이트
                        track_obj.current_score = 0.7 * track_obj.current_score + 0.3 * avg_score

                name_to_draw = track_obj.current_label

            # ★ ADD: 버퍼에 기반한 top2 후보 계산(배타 재할당용)
            label_scores = {}
            for l, s in zip(getattr(track_obj, "label_buffer", []), getattr(track_obj, "score_buffer", [])):
                if l and l != "Unknown":
                    label_scores.setdefault(l, []).append(s)
            label_means = {l: float(np.mean(v)) for l, v in label_scores.items()}

            top1_label = name_to_draw if name_to_draw else ""
            top1_score = float(getattr(track_obj, "current_score", avg_score))

            top2_label, top2_score = "", 0.0
            if label_means:
                alt = [(l, sc) for l, sc in label_means.items() if l != top1_label]
                if alt:
                    alt.sort(key=lambda x: x[1], reverse=True)
                    top2_label, top2_score = alt[0]

            # ★ CHANGE: '바로 그리기' 대신 proposals에 적재
            x, y, w, h = track_obj.to_tlwh()
            proposals.append({
                "track_id": track_obj.track_id,
                "tlwh": (int(x), int(y), int(w), int(h)),
                "label": top1_label,
                "score": top1_score,
                "alt_label": top2_label,
                "alt_score": float(top2_score),
                "_status": "pending"
            })

        # ===================== 라벨 배타성 강제 구간 =====================
        # ★ ADD: 1라운드 - 같은 라벨끼리 경쟁(최고 점수 1개만 유지, 나머지 패자 표시)
        from collections import defaultdict
        by_label = defaultdict(list)
        for i, p in enumerate(proposals):
            if p["label"] and p["label"] != "Unknown":
                by_label[p["label"]].append(i)

        assigned = {}  # label -> proposal index
        for label, idxs in by_label.items():
            if len(idxs) == 1:
                assigned[label] = idxs[0]
                proposals[idxs[0]]["_status"] = "won"
            else:
                idxs.sort(key=lambda k: proposals[k]["score"], reverse=True)
                winner = idxs[0]
                assigned[label] = winner
                proposals[winner]["_status"] = "won"
                for k in idxs[1:]:
                    proposals[k]["_status"] = "lost"
                    proposals[k]["label"] = "Unknown"  # 잠정 강등

        # ★ ADD: 2라운드(a) - 패자 재할당(대안 라벨이 비어 있고 점수가 기준 이상이면 즉시 배정)
        for p in proposals:
            if p["_status"] == "lost" and p["alt_label"] and p["alt_score"] >= MIN_ACCEPT:
                if p["alt_label"] not in assigned:  # 아직 누구도 차지하지 않은 라벨
                    p["label"] = p["alt_label"]
                    p["_status"] = "won"
                    p["assigned_score"] = p["alt_score"]
                    assigned[p["alt_label"]] = proposals.index(p)

        # ★ ADD: 2라운드(b) - 스틸(이미 배정된 라벨도 margin만큼 더 높으면 교체)
        changed = True
        while changed:
            changed = False
            for label, winner_idx in list(assigned.items()):
                winner = proposals[winner_idx]
                current_assigned_score = winner.get("assigned_score", winner["score"])
                # 이 라벨을 노리는 패자/보류자들 중 스틸 가능한 후보 찾기
                stealers = [ (i, p) for i, p in enumerate(proposals)
                             if p["_status"] in ("lost", "pending")
                             and p["alt_label"] == label
                             and p["alt_score"] >= MIN_ACCEPT
                             and p["alt_score"] > current_assigned_score + STEAL_MARGIN ]
                if stealers:
                    # 가장 점수 높은 스틸러 선택
                    stealers.sort(key=lambda kv: kv[1]["alt_score"], reverse=True)
                    thief_idx, thief = stealers[0]

                    # 기존 우승자 강등
                    winner["label"] = "Unknown"
                    winner["_status"] = "lost"

                    # 새 우승자 지정
                    thief["label"] = label
                    thief["_status"] = "won"
                    thief["assigned_score"] = thief["alt_score"]
                    assigned[label] = thief_idx
                    changed = True

        # ===================== 최종 그리기 =====================
        for p in proposals:
            x, y, w, h = p["tlwh"]
            x1, y1, x2, y2 = x, y, x + w, y + h
            ld = p["label"]
            color = (0, 255, 0) if ld and ld != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            tag = f'ID:{p["track_id"]} {ld}' if ld else f'ID:{p["track_id"]}'
            cv2.putText(frame, tag, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        #frame = cv2.resize(frame, (640, 960))
        cv2.imshow("YOLO + FaceNet + KNN + DeepSORT", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.detection import Detection


# 이 코드에서 라벨 결정 방식 보완 ing
# ===================== 기본 세팅 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FaceNet (하드마이닝 학습 가중치)
facenet = InceptionResnetV1(pretrained=None, classify=False).to(device)
state = torch.load("facenet_triplet_hard_pk.pth", map_location=device)
facenet.load_state_dict(state, strict=False)
facenet.eval()

# KNN 분류기
knn_clf = joblib.load("knn_face_classifier_pk.pkl")

# YOLO 얼굴 검출 (커스텀 가중치 경로 또는 동일 폴더)
yolo_model = YOLO("yolov8n-face.pt")

# 전처리
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# DeepSORT 세팅
# max_cosine_distance는 appearance 유사도 임계값. 0.2~0.4 사이에서 튜닝
metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, None)
deepsort_tracker = tracker.Tracker(metric)

# ===================== 유틸 =====================
# 눈 위치로 회전 보정 - 얼굴 정렬
def align_face(pil_img: Image.Image) -> Image.Image:
    np_img = np.array(pil_img)
    lms = face_recognition.face_landmarks(np_img)
    if not lms: return pil_img
    lm = lms[0]
    if 'left_eye' not in lm or 'right_eye' not in lm:
        return pil_img
    left_eye = np.mean(lm['left_eye'], axis=0)
    right_eye = np.mean(lm['right_eye'], axis=0)
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    eye_center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
    h, w = np_img.shape[:2]
    M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    aligned = cv2.warpAffine(np_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(aligned)

# 검출 박스를 비율만큼 확장
def expand_box(x1, y1, x2, y2, scale, W, H):
    cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
    w = (x2 - x1) * scale; h = (y2 - y1) * scale
    nx1 = int(max(0, cx - w/2)); ny1 = int(max(0, cy - h/2))
    nx2 = int(min(W-1, cx + w/2)); ny2 = int(min(H-1, cy + h/2))
    return nx1, ny1, nx2, ny2

#임베딩 L2 정규화 
def l2_normalize(v):
    n = np.linalg.norm(v) + 1e-8
    return v / n

#두 박스의 ioU 계산 , 교집합 합집합 면적 계산
def iou(bb_test, bb_gt):
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[0] + bb_test[2], bb_gt[0] + bb_gt[2])
    yy2 = min(bb_test[1] + bb_test[3], bb_gt[1] + bb_gt[3])
    w = max(0., xx2 - xx1); h = max(0., yy2 - yy1)
    inter = w * h
    union = bb_test[2]*bb_test[3] + bb_gt[2]*bb_gt[3] - inter
    return inter/union if union > 0 else 0

# ===================== 메인 루프 =====================
def main():

    # #영상으로 입력받기
    # video_path = "C:/Users/DS/Downloads/유진외2인중복.mp4"
    # cap = cv2.VideoCapture(video_path)
    # if not cap.isOpened():
    #     print(f"Error: Could not open video file {video_path}")
    #     return

    #웹캠으로 입력받기
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # 0 = 기본 웹캠
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    #웹캠 - 해상도 부분 (선택)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 


    # KNN distance 임계값 (작을수록 유사); 
    # 여기서 가장 비슷한 애랑 거리가 0.80 차이나면 unknown인 거임 현재!!!!!!!!!!!
    DIST_THR = 0.6

    # 라벨 안정화
    BUFFER = 7
    MIN_STABLE = 3
    SWITCH_MARGIN = 0.15  # top-2 margin 보수화에 쓰고 싶으면 확장 가능

    # ★ ADD: 배타 재할당 파라미터(2순위 허용/스틸 기준)
    MIN_ACCEPT = 0.55     # 2순위로 갈아탈 때 최소 신뢰도
    STEAL_MARGIN = 0.03   # 이미 배정된 라벨을 빼앗으려면 이만큼 더 높아야 함

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        H, W = frame.shape[:2]

        # YOLO 얼굴 탐지
        results = yolo_model(frame, verbose=False)[0]
        detections = []

        # 현재 프레임의 det 정보(박스, 라벨 등) 저장해 추적 후 트랙에 붙일 것
        det_pack = []  # (tlwh, label, score)
        for box in results.boxes:
            conf = float(box.conf)
            if conf < 0.5: 
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, scale=1.25, W=W, H=H)
            if x2 <= x1 or y2 <= y1: 
                continue

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            try:
                # 정렬 + 임베딩
                pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                pil = align_face(pil)
                t = transform(pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = facenet(t).cpu().numpy().flatten()
                emb = l2_normalize(emb)

                # KNN 추론
                distances, idxs = knn_clf.kneighbors([emb], n_neighbors=2, return_distance=True)
                d1, d2 = distances[0]
                pred_label = knn_clf.predict([emb])[0]
                # 거리 기반 Unknown 처리
                if d1 > DIST_THR:
                    pred_label = "Unknown"

                # DeepSORT detection 객체 생성 (tlwh, confidence, feature)
                tlwh = [x1, y1, x2 - x1, y2 - y1]
                detections.append(Detection(tlwh, conf, emb))
                det_pack.append((tlwh, pred_label, 1.0/(1.0 + d1)))  # score는 거리 역수로 간단 표기
            except Exception as e:
                # 실패시 무시
                continue

        # DeepSORT 추적 업데이트
        deepsort_tracker.predict()
        deepsort_tracker.update(detections)

        # ★ ADD: 이번 프레임에서 그리기 전에 트랙별 표시 후보를 먼저 모은다.
        proposals = []  # 각 원소: dict(track_id, tlwh, label, score, alt_label, alt_score, _status)

        # 트랙별 라벨 버퍼/히스테리시스
        for track_obj in deepsort_tracker.tracks:
            if not track_obj.is_confirmed() or track_obj.time_since_update > 1:
                continue

            # 트랙 초기 안정화: 너무 어린 트랙은 패스
            if track_obj.age < 5:
                continue

            # 트랙에 버퍼가 없으면 생성
            if not hasattr(track_obj, "label_buffer"):
                track_obj.label_buffer = deque(maxlen=BUFFER)
                track_obj.score_buffer = deque(maxlen=BUFFER)
                track_obj.current_label = None

            # 이번 프레임에서 이 트랙과 IoU가 가장 큰 detection 라벨을 버퍼에 추가
            t_tlwh = track_obj.to_tlwh()
            best = None; best_iou = 0.0
            for tlwh, lab, scr in det_pack:
                i = iou(t_tlwh, tlwh)
                if i > best_iou:
                    best_iou = i
                    best = (lab, scr)
            if best and best_iou > 0.2:
                track_obj.label_buffer.append(best[0])
                track_obj.score_buffer.append(best[1])

            name_to_draw = ""
            avg_score = 0.0
            if len(track_obj.label_buffer) >= MIN_STABLE:

                # 다수결 + 평균점수
                mc = Counter(track_obj.label_buffer).most_common(1)[0][0]
                avg_score = np.mean([s for l, s in zip(track_obj.label_buffer, track_obj.score_buffer) if l == mc]) if track_obj.score_buffer else 0.0

                # 히스테리시스: 라벨이 바뀌려면 평균점수도 일정 이상일 때만
                if track_obj.current_label is None:
                    track_obj.current_label = mc
                    track_obj.current_score = avg_score
                else:
                    if (mc != track_obj.current_label) and (avg_score >= max(0.55, track_obj.current_score + 0.05)):
                        track_obj.current_label = mc
                        track_obj.current_score = avg_score
                    else:
                        # 유지하면서 점수는 천천히 업데이트
                        track_obj.current_score = 0.7 * track_obj.current_score + 0.3 * avg_score

                name_to_draw = track_obj.current_label

            # ★ ADD: 버퍼에 기반한 top2 후보 계산(배타 재할당용)
            label_scores = {}
            for l, s in zip(getattr(track_obj, "label_buffer", []), getattr(track_obj, "score_buffer", [])):
                if l and l != "Unknown":
                    label_scores.setdefault(l, []).append(s)
            label_means = {l: float(np.mean(v)) for l, v in label_scores.items()}

            top1_label = name_to_draw if name_to_draw else ""
            top1_score = float(getattr(track_obj, "current_score", avg_score))

            top2_label, top2_score = "", 0.0
            if label_means:
                alt = [(l, sc) for l, sc in label_means.items() if l != top1_label]
                if alt:
                    alt.sort(key=lambda x: x[1], reverse=True)
                    top2_label, top2_score = alt[0]

            # ★ CHANGE: '바로 그리기' 대신 proposals에 적재
            x, y, w, h = track_obj.to_tlwh()
            proposals.append({
                "track_id": track_obj.track_id,
                "tlwh": (int(x), int(y), int(w), int(h)),
                "label": top1_label,
                "score": top1_score,
                "alt_label": top2_label,
                "alt_score": float(top2_score),
                "_status": "pending"
            })

        # ===================== 라벨 배타성 강제 구간 =====================
        # ★ ADD: 1라운드 - 같은 라벨끼리 경쟁(최고 점수 1개만 유지, 나머지 패자 표시)
        from collections import defaultdict
        by_label = defaultdict(list)
        for i, p in enumerate(proposals):
            if p["label"] and p["label"] != "Unknown":
                by_label[p["label"]].append(i)

        assigned = {}  # label -> proposal index
        for label, idxs in by_label.items():
            if len(idxs) == 1:
                assigned[label] = idxs[0]
                proposals[idxs[0]]["_status"] = "won"
            else:
                idxs.sort(key=lambda k: proposals[k]["score"], reverse=True)
                winner = idxs[0]
                assigned[label] = winner
                proposals[winner]["_status"] = "won"
                for k in idxs[1:]:
                    proposals[k]["_status"] = "lost"
                    proposals[k]["label"] = "Unknown"  # 잠정 강등

        # ★ ADD: 2라운드(a) - 패자 재할당(대안 라벨이 비어 있고 점수가 기준 이상이면 즉시 배정)
        for p in proposals:
            if p["_status"] == "lost" and p["alt_label"] and p["alt_score"] >= MIN_ACCEPT:
                if p["alt_label"] not in assigned:  # 아직 누구도 차지하지 않은 라벨
                    p["label"] = p["alt_label"]
                    p["_status"] = "won"
                    p["assigned_score"] = p["alt_score"]
                    assigned[p["alt_label"]] = proposals.index(p)

        # ★ ADD: 2라운드(b) - 스틸(이미 배정된 라벨도 margin만큼 더 높으면 교체)
        changed = True
        while changed:
            changed = False
            for label, winner_idx in list(assigned.items()):
                winner = proposals[winner_idx]
                current_assigned_score = winner.get("assigned_score", winner["score"])
                # 이 라벨을 노리는 패자/보류자들 중 스틸 가능한 후보 찾기
                stealers = [ (i, p) for i, p in enumerate(proposals)
                             if p["_status"] in ("lost", "pending")
                             and p["alt_label"] == label
                             and p["alt_score"] >= MIN_ACCEPT
                             and p["alt_score"] > current_assigned_score + STEAL_MARGIN ]
                if stealers:
                    # 가장 점수 높은 스틸러 선택
                    stealers.sort(key=lambda kv: kv[1]["alt_score"], reverse=True)
                    thief_idx, thief = stealers[0]

                    # 기존 우승자 강등
                    winner["label"] = "Unknown"
                    winner["_status"] = "lost"

                    # 새 우승자 지정
                    thief["label"] = label
                    thief["_status"] = "won"
                    thief["assigned_score"] = thief["alt_score"]
                    assigned[label] = thief_idx
                    changed = True

        # ===================== 최종 그리기 =====================
        for p in proposals:
            x, y, w, h = p["tlwh"]
            x1, y1, x2, y2 = x, y, x + w, y + h
            ld = p["label"]
            color = (0, 255, 0) if ld and ld != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            tag = f'ID:{p["track_id"]} {ld}' if ld else f'ID:{p["track_id"]}'
            cv2.putText(frame, tag, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        #frame = cv2.resize(frame, (640, 960))
        cv2.imshow("YOLO + FaceNet + KNN + DeepSORT", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
