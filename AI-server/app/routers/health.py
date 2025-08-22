# 배포된 서버가 정상인지 모니터링에 사용
from fastapi import APIRouter

router = APIRouter(prefix="/healthz", tags=["health"])

@router.get("")
async def healthz():
    return {"status": "ok"}
