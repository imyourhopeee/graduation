# app/routers/config_seats.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Tuple, Optional
import os, json

# inference.py의 _engine()을 사용하기 위해 import 경로를 수정합니다.
try:
    from app.models.inference import _engine, SeatWire 
except ImportError:
    # 모듈 구조에 따라 경로가 다를 수 있어 예외 처리
    from app.inference import _engine 


SEATS_JSON_PATH = os.getenv("SEATS_CONFIG", "tripwire_perp.json")
ALLOWED = {"p1", "p2", "d_near", "d_far", "inward_sign", "seat_id"}

def to_engine_dict(s: dict) -> dict:
    p1 = s.get("p1", [0, 0])
    p2 = s.get("p2", [0, 0])
    return {
        "p1": [int(float(p1[0])), int(float(p1[1]))],
        "p2": [int(float(p2[0])), int(float(p2[1]))],
        "d_near": float(s.get("d_near", 0)),
        "d_far": float(s.get("d_far", 0)),
        "inward_sign": 1 if int(s.get("inward_sign", 1)) >= 0 else -1,
        "seat_id": int(s.get("seat_id", 0)),
    }

class SeatWireModel(BaseModel):
    p1: Tuple[int, int]
    p2: Tuple[int, int]
    b_near: float = 20.0
    b_far: float = 8.0
    inward_sign: int = 1
    d_near: float = 180.0
    d_far: float = 120.0
    seat_id: Optional[int] = None


    # NEW: 카메라 보정 관련 필드
    ref_w: Optional[int] = None
    ref_h: Optional[int] = None
    flip_horizontal: Optional[bool] = None

router = APIRouter(prefix="/config/seats", tags=["config-seats"])

@router.get("")
def get_seats():
    if not os.path.exists(SEATS_JSON_PATH):
        # 엔진도 비우기(선택)
        try: _engine().set_seats([]) 
        except: pass
        return []

    try:
        with open(SEATS_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)  # ← 클라이언트용 원본(메타 포함)
            # ★ 엔진에는 허용 키만 넣기
            try:
                clean = [to_engine_dict(s) for s in data]
                new_seats = [SeatWire(**c) for c in clean]
                _engine().set_seats(new_seats)
            except Exception as e:
                print(f"[WARN] Failed to load seats to engine (clean): {e}")
            return data  # 프론트엔드는 메타 포함 원본 유지
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("")
def put_seats(seats: List[SeatWireModel]):
    try:
        # 1) 파일에 그대로 저장(메타 포함)
        raw_list = [s.model_dump() for s in seats]
        with open(SEATS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(raw_list, f, ensure_ascii=False, indent=2)

        # 2) ★ 엔진 메모리에 허용 키만 반영
        try:
            clean = [to_engine_dict(s) for s in raw_list]
            new_seats = [SeatWire(**c) for c in clean]
            _engine().set_seats(new_seats)
            print(f"[SEATS SET] count={len(new_seats)}")
        except Exception as e:
            print(f"[WARN] Failed to update seats to engine (clean): {e}")

        return {"ok": True, "count": len(seats), "path": SEATS_JSON_PATH}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Dwell 시간 설정 추가
class DwellModel(BaseModel):
    seconds: float

DWELL_JSON_PATH = os.getenv("DWELL_CONFIG", "dwell.json")

@router.get("/dwell")
def get_dwell():
    """현재 dwell 시간 조회"""
    if not os.path.exists(DWELL_JSON_PATH):
        return {"seconds": 8.0} 
    try:
        with open(DWELL_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            sec = float(data.get("seconds", 8.0))
            # [추가] AI 엔진의 현재 Dwell 시간으로 동기화
            _engine().set_dwell_time(sec)
            return {"seconds": sec}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/dwell")
def put_dwell(body: DwellModel):
    """dwell 시간 저장"""
    sec = max(0.5, float(body.seconds)) 
    try:
        # 1. 파일에 저장
        with open(DWELL_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump({"seconds": sec}, f, ensure_ascii=False, indent=2)
        
        # 2. AI 엔진 메모리에 즉시 반영
        try:
            _engine().set_dwell_time(sec)
        except Exception as e:
            print(f"[WARN] Failed to update dwell time in engine: {e}")
            pass

        return {"ok": True, "seconds": sec, "path": DWELL_JSON_PATH}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))