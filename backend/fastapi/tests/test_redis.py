import uuid
import pytest
from fastapi.testclient import TestClient
from services.cache_service import cache_service
from services.rate_limiter import rate_limiter_service

@pytest.mark.anyio
async def test_cache_set_get_and_delete():
    test_key = "nexora:test:unit_test_key"
    test_value = {"user_id": "123", "role": "ROLE_STUDENT", "score": 98.5}

    # 1. Set Cache
    success = await cache_service.set(test_key, test_value, ttl_seconds=60)
    assert success is True

    # 2. Get Cache (Hit)
    cached = await cache_service.get(test_key)
    assert cached is not None
    assert cached["user_id"] == "123"
    assert cached["score"] == 98.5

    # 3. Check Existence
    exists = await cache_service.exists(test_key)
    assert exists is True

    # 4. Delete Cache
    del_success = await cache_service.delete(test_key)
    assert del_success is True

    # 5. Get Cache (Miss)
    assert await cache_service.get(test_key) is None

@pytest.mark.anyio
async def test_cache_user_invalidation():
    user_id = "user-uuid-999"
    profile_key = cache_service.keys.user_profile(user_id)
    perm_key = cache_service.keys.user_permissions(user_id)

    await cache_service.set(profile_key, {"email": "user@nexora.ai"}, ttl_seconds=60)
    await cache_service.set(perm_key, ["CHAT_READ", "CHAT_WRITE"], ttl_seconds=60)

    assert await cache_service.get(profile_key) is not None
    assert await cache_service.get(perm_key) is not None

    # Invalidate both keys
    await cache_service.invalidate_user_cache(user_id)

    assert await cache_service.get(profile_key) is None
    assert await cache_service.get(perm_key) is None

def test_session_endpoint_cache_aside(client: TestClient):
    session_id = uuid.uuid4()

    # Request 1: Cache Miss
    resp1 = client.get(f"/api/v1/session/{session_id}")
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["session_id"] == str(session_id)
    assert data1["cached"] is False

    # Request 2: Cache Hit
    resp2 = client.get(f"/api/v1/session/{session_id}")
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["session_id"] == str(session_id)
    assert data2["cached"] is True

@pytest.mark.anyio
async def test_rate_limiter_under_limit():
    action = "test_action_allowed"
    ident = "client_ip_1"

    # Limit = 5 requests
    for i in range(5):
        allowed, remaining, retry_after = await rate_limiter_service.check_rate_limit(
            action=action, identifier=ident, limit=5, window_seconds=60
        )
        assert allowed is True
        assert remaining == 5 - (i + 1)
        assert retry_after > 0

@pytest.mark.anyio
async def test_rate_limiter_exceeded_blocks_with_retry_after():
    action = "test_action_blocked"
    ident = "client_ip_2"

    # Exhaust limit of 3
    for _ in range(3):
        await rate_limiter_service.check_rate_limit(
            action=action, identifier=ident, limit=3, window_seconds=60
        )

    # 4th request should be blocked
    allowed, remaining, retry_after = await rate_limiter_service.check_rate_limit(
        action=action, identifier=ident, limit=3, window_seconds=60
    )
    assert allowed is False
    assert remaining == 0
    assert retry_after > 0
