from uuid import UUID
from fastapi import APIRouter, Depends
from services.cache_service import cache_service
from middleware.rate_limit import rate_limit_general_api

router = APIRouter(prefix="/session", tags=["Session State"])

@router.get("/{session_id}", dependencies=[Depends(rate_limit_general_api)])
async def get_session_state(session_id: UUID):
    """
    Retrieve session state by ID using the Cache-Aside pattern with Redis.
    Protected by general API rate limiting.
    """
    cache_key = cache_service.keys.session(str(session_id))

    # 1. Check Redis Cache
    cached_data = await cache_service.get(cache_key)
    if cached_data is not None:
        cached_data["cached"] = True
        return cached_data

    # 2. Cache Miss: Compute State
    session_data = {
        "session_id": str(session_id),
        "status": "ACTIVE",
        "language": "English",
        "context_length": 12,
        "cached": False,
    }

    # 3. Store in Redis with TTL
    await cache_service.set(cache_key, session_data, ttl_seconds=120)

    return session_data
