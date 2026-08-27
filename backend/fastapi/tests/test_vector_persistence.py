import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from documents.hybrid_ingestion_service import EnrichedChunk
from documents.persistence.postgres_vector_repository import postgres_vector_repository
from documents.persistence.qdrant_vector_repository import qdrant_vector_repository
from documents.persistence.vector_persistence_service import (
    VectorPersistenceService,
    vector_persistence_service,
    DEV_DEFAULT_USER_ID,
)
from embeddings.embedding_service import embedding_service


def create_sample_chunk(
    chunk_id: str = "chunk_001",
    document_id: str = "doc_001",
    document_name: str = "sample.txt",
    chunk_index: int = 0,
    content: str = "ನಮಸ್ಕಾರ, NEXORA AI ಪ್ಲಾಟ್‌ಫಾರ್ಮ್‌ಗೆ ಸುಸ್ವಾಗತ.",
    language: str = "kn",
    script: str = "Kannada",
    dimension: int = 1536,
) -> EnrichedChunk:
    """Helper to generate a valid EnrichedChunk with 1536d normalized vector."""
    emb = embedding_service.embed_single(content)
    vec = emb.vector if dimension == 1536 else [0.1] * dimension
    return EnrichedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_name=document_name,
        chunk_index=chunk_index,
        content=content,
        character_count=len(content),
        token_count=10,
        start_offset=0,
        end_offset=len(content),
        language=language,
        script=script,
        embedding=vec,
        embedding_dimension=len(vec),
        embedding_provider="development_deterministic",
        embedding_model="development-1536",
        metadata={"category": "sample"},
    )


def create_mock_session():
    """Creates a properly configured SQLAlchemy AsyncSession mock."""
    session = AsyncMock()
    session.add = MagicMock()  # Synchronous in SQLAlchemy
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.execute = AsyncMock()
    return session


# ==============================================================================
# 1. CORE REPOSITORY PERSISTENCE TESTS
# ==============================================================================

@pytest.mark.anyio
async def test_postgres_chunk_persistence():
    mock_session = create_mock_session()

    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.first.return_value = None
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute.return_value = mock_result

    chunk_uuid = uuid.uuid4()
    user_uuid = uuid.UUID(DEV_DEFAULT_USER_ID)
    emb = [0.05] * 1536

    chunk = await postgres_vector_repository.upsert_chunk(
        session=mock_session,
        chunk_id=chunk_uuid,
        user_id=user_uuid,
        document_name="test_doc.txt",
        chunk_index=0,
        content="Kannada chunk content",
        embedding=emb,
    )

    assert chunk.id == chunk_uuid
    assert chunk.document_name == "test_doc.txt"
    assert chunk.user_id == user_uuid
    assert mock_session.commit.called
    assert mock_session.add.called


@pytest.mark.anyio
async def test_qdrant_chunk_persistence():
    point_data = [{
        "id": str(uuid.uuid4()),
        "vector": [0.01] * 1536,
        "payload": {
            "document_id": "doc_101",
            "document_name": "overview.pdf",
            "chunk_index": 0,
            "content": "Qdrant persistence test",
            "language": "en",
            "script": "Latin",
        },
    }]

    count = await qdrant_vector_repository.upsert_chunk_points(
        points_data=point_data,
        user_id=DEV_DEFAULT_USER_ID,
    )
    assert count == 1


@pytest.mark.anyio
async def test_both_stores_receive_same_vector():
    chunk = create_sample_chunk()
    mock_session = create_mock_session()

    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.first.return_value = None
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute.return_value = mock_result

    rec = await vector_persistence_service.persist_chunk(
        session=mock_session,
        chunk=chunk,
        user_id=DEV_DEFAULT_USER_ID,
    )

    assert rec.postgres_persisted is True
    assert rec.qdrant_persisted is True
    assert rec.status == "persisted"


# ==============================================================================
# 2. VALIDATION & SECURITY TESTS
# ==============================================================================

@pytest.mark.anyio
async def test_1536_dimension_validation():
    invalid_chunk = create_sample_chunk(dimension=512)
    mock_session = create_mock_session()

    with pytest.raises(ValueError) as exc:
        await vector_persistence_service.persist_chunk(
            session=mock_session,
            chunk=invalid_chunk,
            user_id=DEV_DEFAULT_USER_ID,
        )
    assert "exactly 1536 dimensions" in str(exc.value)


@pytest.mark.anyio
async def test_empty_chunk_rejection():
    mock_session = create_mock_session()
    with pytest.raises(ValueError) as exc:
        await vector_persistence_service.persist_chunks(
            session=mock_session,
            document_id="doc_empty",
            document_name="empty.txt",
            chunks=[],
            user_id=DEV_DEFAULT_USER_ID,
        )
    assert "No chunks provided" in str(exc.value)


@pytest.mark.anyio
async def test_user_isolation_query():
    mock_session = create_mock_session()
    user_a = uuid.uuid4()

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    res = await postgres_vector_repository.get_chunk(
        session=mock_session,
        chunk_id=uuid.uuid4(),
        user_id=user_a,
    )
    assert res is None


@pytest.mark.anyio
async def test_duplicate_persistence_idempotency():
    mock_session = create_mock_session()
    existing_chunk = MagicMock()

    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.first.return_value = existing_chunk
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute.return_value = mock_result

    chunk_uuid = uuid.uuid4()
    user_uuid = uuid.UUID(DEV_DEFAULT_USER_ID)
    emb = [0.02] * 1536

    updated = await postgres_vector_repository.upsert_chunk(
        session=mock_session,
        chunk_id=chunk_uuid,
        user_id=user_uuid,
        document_name="doc.txt",
        chunk_index=0,
        content="Updated content",
        embedding=emb,
    )
    assert updated == existing_chunk
    assert existing_chunk.content == "Updated content"


# ==============================================================================
# 3. DELETION & COUNTING TESTS
# ==============================================================================

@pytest.mark.anyio
async def test_chunk_deletion():
    mock_session = create_mock_session()
    mock_result = MagicMock()
    mock_result.rowcount = 1
    mock_session.execute.return_value = mock_result

    deleted = await vector_persistence_service.delete_chunk(
        session=mock_session,
        chunk_id=str(uuid.uuid4()),
        user_id=DEV_DEFAULT_USER_ID,
    )
    assert deleted is True


@pytest.mark.anyio
async def test_document_deletion():
    mock_session = create_mock_session()
    mock_result = MagicMock()
    mock_result.rowcount = 3
    mock_session.execute.return_value = mock_result

    deleted = await vector_persistence_service.delete_document(
        session=mock_session,
        document_id="doc_to_delete",
        user_id=DEV_DEFAULT_USER_ID,
    )
    assert deleted is True


@pytest.mark.anyio
async def test_user_chunk_counting():
    mock_session = create_mock_session()
    mock_result = MagicMock()
    mock_result.scalar_one.return_value = 5
    mock_session.execute.return_value = mock_result

    count = await postgres_vector_repository.count_user_chunks(
        session=mock_session,
        user_id=uuid.UUID(DEV_DEFAULT_USER_ID),
    )
    assert count == 5


@pytest.mark.anyio
async def test_persistence_status():
    mock_session = create_mock_session()
    mock_result = MagicMock()
    mock_result.scalar_one.return_value = 3
    mock_session.execute.return_value = mock_result

    status_res = await vector_persistence_service.get_document_status(
        session=mock_session,
        document_id="doc_status_check",
        user_id=DEV_DEFAULT_USER_ID,
    )
    assert status_res.document_id == "doc_status_check"
    assert status_res.postgres_chunk_count == 3
    assert status_res.persisted is True


# ==============================================================================
# 4. SEMANTIC SEARCH & MULTILINGUAL TESTS
# ==============================================================================

@pytest.mark.anyio
async def test_search_by_semantic_similarity():
    mock_session = create_mock_session()
    mock_chunk = MagicMock()
    mock_chunk.id = uuid.uuid4()
    mock_chunk.document_name = "kannada_guide.pdf"
    mock_chunk.content = "ಕನ್ನಡ ವ್ಯಾಕರಣ ಮತ್ತು ನೈಸರ್ಗಿಕ ಭಾಷಾ ಸಂಸ್ಕರಣೆ."

    mock_result = MagicMock()
    mock_result.all.return_value = [(mock_chunk, 0.15)]
    mock_session.execute.return_value = mock_result

    search_res = await vector_persistence_service.search_documents(
        session=mock_session,
        query="ಕನ್ನಡ ವ್ಯಾಕರಣ",
        user_id=DEV_DEFAULT_USER_ID,
        top_k=3,
    )

    assert search_res.query == "ಕನ್ನಡ ವ್ಯಾಕರಣ"
    assert search_res.total_results >= 1
    assert search_res.results[0].similarity >= 0.80


@pytest.mark.anyio
async def test_search_user_isolation():
    mock_session = create_mock_session()
    mock_result = MagicMock()
    mock_result.all.return_value = []
    mock_session.execute.return_value = mock_result

    res = await vector_persistence_service.search_documents(
        session=mock_session,
        query="Private user content",
        user_id="11111111-1111-1111-1111-111111111111",
    )
    assert res.total_results == 0


@pytest.mark.anyio
async def test_batch_persistence():
    chunks = [
        create_sample_chunk(chunk_id="c1", chunk_index=0, content="Chunk 1 content"),
        create_sample_chunk(chunk_id="c2", chunk_index=1, content="Chunk 2 content"),
    ]
    mock_session = create_mock_session()

    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.first.return_value = None
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute.return_value = mock_result

    result = await vector_persistence_service.persist_chunks(
        session=mock_session,
        document_id="doc_batch",
        document_name="batch.txt",
        chunks=chunks,
        user_id=DEV_DEFAULT_USER_ID,
    )

    assert result.total_chunks == 2
    assert result.postgres_success_count == 2
    assert result.qdrant_success_count == 2
    assert result.status == "success"


# ==============================================================================
# 5. ERROR & FAILURE TOLERANCE TESTS
# ==============================================================================

@pytest.mark.anyio
async def test_postgres_failure_handling():
    chunk = create_sample_chunk()
    mock_session = create_mock_session()
    mock_session.execute.side_effect = RuntimeError("Postgres connection dropped")

    rec = await vector_persistence_service.persist_chunk(
        session=mock_session,
        chunk=chunk,
        user_id=DEV_DEFAULT_USER_ID,
    )
    assert rec.postgres_persisted is False
    assert rec.qdrant_persisted is True
    assert rec.status == "partial"


@pytest.mark.anyio
async def test_qdrant_failure_handling():
    chunk = create_sample_chunk()
    mock_session = create_mock_session()

    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.first.return_value = None
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute.return_value = mock_result

    with patch.object(qdrant_vector_repository, "upsert_chunk_points", side_effect=RuntimeError("Qdrant unavailable")):
        rec = await vector_persistence_service.persist_chunk(
            session=mock_session,
            chunk=chunk,
            user_id=DEV_DEFAULT_USER_ID,
        )
        assert rec.postgres_persisted is True
        assert rec.qdrant_persisted is False
        assert rec.status == "partial"


@pytest.mark.anyio
async def test_invalid_vector_rejection():
    mock_session = create_mock_session()
    bad_chunk = create_sample_chunk(dimension=100)
    with pytest.raises(ValueError) as exc:
        await vector_persistence_service.persist_chunk(
            session=mock_session,
            chunk=bad_chunk,
            user_id=DEV_DEFAULT_USER_ID,
        )
    assert "exactly 1536 dimensions" in str(exc.value)


# ==============================================================================
# 6. REST API ENDPOINT INTEGRATION TESTS
# ==============================================================================

def test_api_persist_endpoint(client: TestClient):
    chunk_sample = create_sample_chunk()
    payload = {
        "document_id": "doc_api_persist",
        "document_name": "api_persist.txt",
        "user_id": DEV_DEFAULT_USER_ID,
        "chunks": [chunk_sample.model_dump()],
    }

    response = client.post("/api/v1/documents/persist", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["document_id"] == "doc_api_persist"
    assert data["total_chunks"] == 1
    assert data["status"] in ["success", "partial_failure"]
    assert data["processing_time_ms"] >= 0.0


def test_api_delete_endpoint(client: TestClient):
    response = client.delete(f"/api/v1/documents/doc_to_delete?user_id={DEV_DEFAULT_USER_ID}")
    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == "doc_to_delete"
    assert "deleted" in data


def test_api_status_endpoint(client: TestClient):
    response = client.get(f"/api/v1/documents/doc_status_check/status?user_id={DEV_DEFAULT_USER_ID}")
    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == "doc_status_check"
    assert "postgres_chunk_count" in data
    assert "qdrant_chunk_count" in data


def test_api_search_endpoint(client: TestClient):
    payload = {
        "query": "ಕನ್ನಡ ಭಾಷೆ ಮತ್ತು ಆರ್ಟಿಫಿಷಿಯಲ್ ಇಂಟೆಲಿಜೆನ್ಸ್",
        "top_k": 3,
        "user_id": DEV_DEFAULT_USER_ID,
    }

    response = client.post("/api/v1/documents/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == payload["query"]
    assert "results" in data
    assert data["processing_time_ms"] >= 0.0
