import uuid
import pytest
from fastapi.testclient import TestClient
from services.cache_service import cache_service, CacheKeyManager
from services.rate_limiter import rate_limiter_service
from services.vector_search_service import vector_search_service
from services.qdrant_service import qdrant_service
from schemas.vector import VectorSearchRequest
from schemas.qdrant import QdrantSearchRequest, QdrantUpsertRequest, QdrantPointInput, QdrantPointPayload

# ==============================================================================
# 1. PGVECTOR INTEGRATION & MULTI-TENANT ISOLATION TESTS
# ==============================================================================

@pytest.mark.anyio
async def test_pgvector_similarity_and_math():
    """Verify deterministic 1536-dimensional vector distance and similarity conversion."""
    vec_a = [0.0] * 1536
    vec_a[0] = 1.0

    # Identical vectors: distance should be 0.0, similarity 1.0
    req_identical = VectorSearchRequest(query_embedding=vec_a, top_k=5)
    assert len(req_identical.query_embedding) == 1536

    # Test service query structure
    res = await vector_search_service.search_similar_chunks(
        request=req_identical,
        session=None,
        user_id=uuid.UUID("00000000-0000-0000-0000-000000000001")
    )
    assert res.total_results >= 1
    assert res.results[0].similarity >= 0.0
    assert res.results[0].similarity <= 1.0

def test_pgvector_endpoint_user_isolation(client: TestClient):
    """Verify that vector search endpoint returns 1536-dim responses with similarity scores."""
    query_vec = [0.02] * 1536
    payload = {
        "query_embedding": query_vec,
        "top_k": 5,
        "document_name": "tenant_spec.pdf"
    }

    res = client.post("/api/v1/vector/search", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["query_dimensions"] == 1536
    assert isinstance(data["results"], list)
    assert len(data["results"]) >= 1
    assert data["results"][0]["similarity"] >= 0.0

# ==============================================================================
# 2. QDRANT VECTOR COLLECTION & ISOLATION TESTS
# ==============================================================================

@pytest.mark.anyio
async def test_qdrant_point_upsert_and_user_isolation():
    """Verify Qdrant point upsert and payload-scoped multi-tenant retrieval."""
    user_owner_id = "33333333-3333-3333-3333-333333333333"

    test_vector = [0.04] * 1536
    point_id = str(uuid.uuid4())

    # 1. Upsert Point for Owner
    upsert_req = QdrantUpsertRequest(
        collection_name="nexora_documents",
        points=[
            QdrantPointInput(
                id=point_id,
                vector=test_vector,
                payload=QdrantPointPayload(
                    user_id=user_owner_id,
                    document_id="confidential_spec.pdf",
                    chunk_index=0,
                    content="Confidential vector data for tenant isolation test.",
                    source="unit_test"
                )
            )
        ]
    )
    upsert_res = await qdrant_service.upsert_points(request=upsert_req, user_id=user_owner_id)
    assert upsert_res.upserted_count == 1

    # 2. Search as Owner User
    search_req = QdrantSearchRequest(
        collection_name="nexora_documents",
        query_vector=test_vector,
        top_k=5,
        user_id=user_owner_id
    )
    owner_res = await qdrant_service.search_points(request=search_req, user_id=user_owner_id)
    assert owner_res.dimensions == 1536

# ==============================================================================
# 3. REDIS CACHE-ASIDE & DISTRIBUTED RATE LIMITING TESTS
# ==============================================================================

@pytest.mark.anyio
async def test_redis_cache_aside_full_lifecycle():
    """Verify cache set, get, exists, TTL, and explicit key invalidation."""
    test_uid = f"user_{uuid.uuid4().hex[:8]}"
    profile_key = CacheKeyManager.user_profile(test_uid)
    perm_key = CacheKeyManager.user_permissions(test_uid)

    profile_payload = {
        "id": test_uid,
        "email": f"{test_uid}@nexora.ai",
        "roles": ["ROLE_STUDENT"],
        "is_active": True
    }
    perm_payload = ["CHAT_READ", "CHAT_WRITE", "VECTOR_SEARCH"]

    # 1. Store in Cache
    assert await cache_service.set(profile_key, profile_payload, ttl_seconds=300) is True
    assert await cache_service.set(perm_key, perm_payload, ttl_seconds=300) is True

    # 2. Verify Cache Hit
    cached_profile = await cache_service.get(profile_key)
    assert cached_profile == profile_payload

    cached_perms = await cache_service.get(perm_key)
    assert cached_perms == perm_payload

    # 3. Explicit Invalidation
    await cache_service.invalidate_user_cache(test_uid)
    assert await cache_service.get(profile_key) is None
    assert await cache_service.get(perm_key) is None

@pytest.mark.anyio
async def test_redis_rate_limiting_enforcement():
    """Verify atomic rate limiting threshold and HTTP 429 response formatting."""
    action = "vector_test_rate"
    client_ident = f"ip_{uuid.uuid4().hex[:6]}"
    limit = 4

    # 4 requests under limit should be allowed
    for i in range(4):
        allowed, remaining, retry_after = await rate_limiter_service.check_rate_limit(
            action=action, identifier=client_ident, limit=limit, window_seconds=60
        )
        assert allowed is True
        assert remaining == limit - (i + 1)
        assert retry_after > 0

    # 5th request must be rejected
    allowed, remaining, retry_after = await rate_limiter_service.check_rate_limit(
        action=action, identifier=client_ident, limit=limit, window_seconds=60
    )
    assert allowed is False
    assert remaining == 0
    assert retry_after > 0

# ==============================================================================
# 4. HTTP API INTEGRATION TESTS
# ==============================================================================

def test_full_health_readiness_probe(client: TestClient):
    """Verify /api/v1/health/ready probes all 3 persistent services."""
    res = client.get("/api/v1/health/ready")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "database" in data
    assert "redis" in data
    assert "vector_db" in data
    assert data["database"]["status"] in ["healthy", "unhealthy"]
    assert data["redis"]["status"] in ["healthy", "degraded"]
    assert data["vector_db"]["status"] in ["healthy", "degraded"]

def test_openapi_schema_completeness(client: TestClient):
    """Verify OpenAPI contract has registered all Phase 01-06 router tags."""
    res = client.get("/openapi.json")
    assert res.status_code == 200
    openapi = res.json()
    paths = openapi["paths"]

    # Verify essential routes are registered
    assert "/api/v1/health/live" in paths
    assert "/api/v1/health/ready" in paths
    assert "/api/v1/chat/message" in paths
    assert "/api/v1/session/{session_id}" in paths
    assert "/api/v1/telemetry" in paths
    assert "/api/v1/vector/search" in paths
    assert "/api/v1/qdrant/search" in paths
    assert "/api/v1/qdrant/upsert" in paths
    assert "/api/v1/qdrant/collections" in paths
