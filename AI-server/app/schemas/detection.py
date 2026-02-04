# 데이터 구조 정의 (AI-server ↔ Event-server 간 JSON 포맷을 일관되게 유지)
from pydantic import BaseModel, conlist
from typing import List, Optional

class ReID(BaseModel):
    osnet: Optional[str] = None
    densenet_label: Optional[str] = None
    densenet_conf: Optional[float] = None

class Face(BaseModel):
    label: Optional[str] = None
    conf: Optional[float] = None
    embedding: Optional[str] = None  # 필요 시 base64 or hex

class OneDet(BaseModel):
    track_id: int
    bbox: conlist(float, min_length=4, max_length=4)
    cls: str
    score: float
    reid: Optional[ReID] = None
    face: Optional[Face] = None
    center: Optional[conlist(float, min_length=2, max_length=2)] = None

class DetectionBatch(BaseModel):
    camera_id: str
    timestamp: str
    fps: Optional[float] = None
    frame_idx: Optional[int] = None
    detections: List[OneDet]
