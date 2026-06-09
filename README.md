# OFFEYE: 사무실 내 CCTV 보안 솔루션(CCTV 관리자용)

[꾸이꾸이 팀]덕성여자대학교\_사이버보안전공 졸업프로젝트

## 개발 목적 및 필요성

최근 언론에서는 기업 내부 정보 유출 및 사생활 정보 유출 사고가 지속적으로 증가하고 있음을 지적하고 있다.
(참고자료: “경찰, 올해 ‘산업스파이’ 317명 검거” _한국경제, 2022.11 / “NIA, 자문위원 개인정보 외부 유출… 잇단 사고 근절 안돼” _보안뉴스, 2025.03)

이러한 문제는 가장 일상적이자, 보안이 철저히 유지되어야 할 공간인 사무실에서 쉽게 발생할 수 있다. 기업들은 출입 통제나 보안 교육 등 다양한 보안 조치를 마련하고 있지만, CCTV를 통한 화면 노출이나 비인가자의 자리 접근으로 인한 기밀 유출 등으로 여전히 사각지대가 존재한다.

따라서 정확한 좌석 단위 감시, 사생활 침해가 적은 감시 시스템, 실시간 자동 판별 및 알림 시스템이 필요하다고 생각되어 10인 이내 소규모 사무실의 CCTV보안을 위한 솔루션 프로그램인 OFFEYE를 개발하였다.
YOLO 기반 객체 탐지, Facenet 기반 얼굴 인식, DeepSORT&OSNet을 활용한 얼굴 추적 및 ReID등의 기술을 결합하여 사무실 내 특정 좌석에 허가되지 않은 인원이 일정 시간 이상 머무는 상황을 자동으로 감지하고,
이를 관리자가 실시간으로 알아차릴 수 있게 함으로써 보안 사고를 신속하게 탐지하는 것을 목표로 한다.

## 기대효과

OFFEYE는 관리자에게 보다 안전하고 효율적인 업무 환경을 제공하며, 다음과 같은 기대 효과가 있다.

- **직원 사생활 보호 강화**

  CCTV 영상에서 개인의 모니터·휴대폰·노트북이 블러 처리되고, 좌석 침입 시 얼굴도 저장하는 것이 아니라 침입 여부 판정에만 사용되기 때문에 직원들이 사생활 침해에 대한 걱정 없이 업무에 임할 수 있다.

- **기업 내부 정보 유출 예방**

  특정 자리에 인가되지 않은 인물이 접근하면 즉시 감지되므로 정보 유출을 시도하기 어려운 환경이 조성되어 내부자로 인한 유출 가능성이 감소한다.

- **유출 사고 즉각 대응 체계 구축**

  침입이 발생하면 실시간으로 관리자 페이지에 배너가 발생하며, 이벤트 로그가 기록되어 신속하게 이상 상황을 탐지할 수 있다. 따라서 사고 발생 시 즉각 대응이 가능해진다.

## 프로젝트 폴더 구조

```text
graduation/
├── README.md
├── .gitignore
├── AI-server/
│   ├── main.py                       # FastAPI 앱 진입점
│   ├── requirements.txt              # Python 의존성 목록
│   ├── .env.example                  # AI 서버 환경변수 예시
│   ├── check_seats.py                # 좌석 설정 확인용 스크립트
│   ├── tripwire_perp.json            # 침입 감지 기준 좌표/설정 파일
│   ├── yolov8n.pt                    # YOLOv8 모델 가중치 파일
│   └── app/
│       ├── clients/
│       │   └── event_client.py        # Event 서버와 통신하는 클라이언트
│       ├── core/
│       │   ├── config.py              # 환경설정 관리
│       │   └── logging.py             # 로깅 설정
│       ├── models/
│       │   ├── blur.py                # 영상/이미지 블러 처리
│       │   ├── inference.py           # AI 추론 엔진
│       │   ├── intrusion_tripwire.py  # 좌석 침입 감지 로직
│       │   └── photo.py               # 사진/프레임 처리 로직
│       ├── routers/
│       │   ├── analyze.py             # 분석 API 라우터
│       │   ├── config_seats.py        # 좌석 영역 설정 API 라우터
│       │   ├── health.py              # 서버 상태 확인 API
│       │   └── stream.py              # 실시간 스트리밍 API 라우터
│       └── schemas/
│           └── detection.py           # 감지 결과 데이터 스키마
│
└── Event-server/
    ├── app.js                         # Express 서버 진입점
    ├── README.md
    ├── .env.example                   # Event 서버 환경변수 예시
    ├── controllers/
    │   ├── authController.js          # 인증 관련 요청 처리
    │   ├── eventController.js         # 이벤트 관련 요청 처리
    │   └── logController.js           # 로그 관련 요청 처리
    ├── db/
    │   ├── index.js                   # DB 연결 및 공통 쿼리 실행
    │   ├── init.js                    # DB 초기화 로직
    │   └── queries.js                 # SQL 쿼리 정의
    ├── middleware/
    │   ├── auth.js                    # 인증 미들웨어
    │   ├── authMiddleware.js          # JWT/권한 검증 미들웨어
    │   ├── authMiddleware copy.js     # 백업/복사본
    │   └── authMiddleware copy 2.js   # 백업/복사본
    ├── models/
    │   ├── eventModel.js              # 이벤트 데이터 모델
    │   ├── seats.js                   # 좌석 데이터 모델
    │   └── userModel.js               # 사용자 데이터 모델
    ├── routes/
    │   ├── auth.js                    # 인증 API 라우터
    │   ├── events.js                  # 이벤트 API 라우터
    │   ├── events copy.js             # 이벤트 라우터 백업/복사본
    │   ├── logs.js                    # 로그 API 라우터
    │   └── seats.js                   # 좌석 API 라우터
    └── socket/
        └── socket.js                  # Socket.IO 실시간 이벤트 전송
```

## 기능 소개

| 기능 | 설명 | 사용 기술 및 데이터 |
| --- | --- | --- |
| **CCTV 영상 내 객체 블러 처리** | CCTV 영상 내 데스크톱 모니터, 노트북, 스마트폰을 탐지하고, 탐지한 영역에는 자동 블러를 적용하며, 모든 프레임에 대해 실시간으로 수행 | CCTV의 촬영 각도와 다양한 조명 조건을 반영한 총 1,800장 이상의 이미지(6000개 이상의 객체)로 자체 데이터셋 구축 및 YOLOv8n 모델 학습 |
| **모니터 화면 촬영 감지** | 타인의 좌석에 침입해 스마트폰으로 모니터 화면을 촬영하려는 시도를 탐지하고, 웹캠에서 촬영을 감지해 대시보드에 로그를 출력 | 스마트폰을 쥔 다양한 자세를 포함한 약 500장의 정면 이미지를 자체적으로 수집하여 YOLOv8n 모델 학습 |
| **자리 침입 감지** | CCTV 영상 내에서 특정 좌석 영역에 일정 시간 이상 머무를 시 ‘INTRUDED’로 판단하고, 이후 모니터 웹캠이 켜지며 얼굴인식과 후면 촬영 감지 기능들을 호출 | 대시보드의 Seats 페이지에서 탐지할 영역을 그려 좌표 값 저장. OUTSIDE, ENTERING, INTRUDED 3가지 상태를 바탕으로 사람의 YOLO 박스 하단 60% 영역과 좌석 영역의 교집합 비율을 계산하고, 설정값 이상일 때 해당 좌석에 사람이 ‘INTRUDED’라고 판단 |
| **얼굴 인식 및 로그 전송** | CCTV 영상 내에서 ‘INTRUDED’ 로 판단된 경우 얼굴 인식을 통해 ‘누가’ 침입했는지 판단하고, 좌석 주인이 아닐 경우 웹 대시보드에 로그를 전송 | YOLOv8-face(프레임 내 얼굴 검출), FaceNet(특징벡터 추출), KNN Classification(신원 식별), DeepSORT, Re-ID(추적). 팀원별로 약 200장(여러 각도·조명·거리 포함)의 얼굴 이미지를 수집하여 학습 진행. 학습되지 않은 인물을 Unknown으로 정확히 구분하기 위해 추가적으로 동양인 안면 데이터셋을 negative learning 방식을 활용하여 모델 보완, 데이터 전처리 후 밝기 변화, 회전, 랜덤 크롭 등 다양한 증강 기법 적용 |

<img width="640" alt="모니터 화면 촬영 감지 로그 예시" src="https://github.com/user-attachments/assets/47de2d3f-02be-44e6-b066-31204171af8c" />

## 실행 화면

1. 시작화면

   <img width="640" alt="시작화면 1" src="https://github.com/user-attachments/assets/bbd143e0-2ec3-4754-86bc-7967d4accdb0" />

   <img width="640" alt="시작화면 2" src="https://github.com/user-attachments/assets/bb1acc8e-05b6-4701-b72d-b8031f812a08" />

2. 대시보드 화면

   <img width="640" alt="대시보드 화면" src="https://github.com/user-attachments/assets/aa91a9c3-78ef-49fb-95c0-981268eeac2c" />

   - (로그) 주인이 아닌 'yoojin' 팀원이 침입한 경우

     <img width="640" alt="침입 로그 예시" src="https://github.com/user-attachments/assets/b555fa15-1cb9-4c0d-bd49-afe75eaeff21" />

   - (알림 배너)주인이 아닌 외부인이 침입한 경우

     <img width="640" alt="침입 알림 배너 예시" src="https://github.com/user-attachments/assets/cd9a49e7-eb7b-4368-987c-2414b52c552b" />

3. 설정(침입으로 간주할 시간을 관리자가 설정 가능)

   <img width="640" alt="침입 시간 설정 화면" src="https://github.com/user-attachments/assets/dd840d10-2312-49ab-8722-539538102f1c" />

4. 전체 로그

   <img width="640" alt="전체 로그 화면" src="https://github.com/user-attachments/assets/cfee1bd8-f2f5-4736-80bd-1fe9155519d1" />

## 한계점

웹캠 비활성 구간에서 발생하는 매우 짧은 촬영 행위가 누락될 수 있고, 좌석 체류 시간을 감지한 뒤 웹캠이 활성화되는 구조로 인해 얼굴 인식 단계까지 약간의 지연이 발생할 수 있다.

또한 여러 좌석이 동시에 존재하는 실제 사무 환경에 대한 검증은 아직 충분하지 않은 상황이다.

향후에는 사람 형상 감지 및 얼굴 인식 모델의 성능을 강화하여 전체 감지 시간을 줄이고 미탐을 감소시키는 방향으로 개선할 필요가 있다.

### 참고문헌

- 윤경윤, 설재형, 조영준. 「Pet Movement Analysis Service using Real-time Re-identification」. 『한국디지털콘텐츠학회논문지』 제 24권 (제3호), 2023.03: 531 - 540 (10page)
- Deepsort, Re-ID 관련 오픈소스 https://github.com/KaiyangZhou/deep-person-reid
- Kaiyang Zhou, Yongxin Yang, Andrea Cavallaro, Tao Xiang. 「Omni-Scale Feature Learning for Person Re-Identification」. arXiv:1905.00953, 2019.05.05.02
- Ultralytics YOLOv8, https://docs.ultralytics.com/ko/models/yolov8/
