import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from retrieval.models.retrieval_result import (
    CandidateChunk,
    RerankedChunk,
    AssembledContext,
    RetrievalResult,
)
from retrieval.hybrid_retriever import HybridRetriever, hybrid_retriever
from retrieval.reranker import DevelopmentReranker, development_reranker
from retrieval.context_assembler import ContextAssembler, context_assembler
from retrieval.retrieval_service import RetrievalService, retrieval_service, DEV_DEFAULT_USER_ID
from embeddings.embedding_service import embedding_service


def create_sample_candidate(
    chunk_id: str = "c_001",
    document_id: str = "doc_001",
    document_name: str = "kannada_guide.pdf",
    chunk_index: int = 0,
    content: str = "ಕನ್ನಡ ಭಾಷೆಯು ಕರ್ನಾಟಕದ ಅಧಿಕೃತ ಭಾಷೆಯಾಗಿದೆ.",
    similarity: float = 0.85,
    language: str = "kn",
    script: str = "Kannada",
    source: str = "qdrant",
) -> CandidateChunk:
    return CandidateChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_name=document_name,
        chunk_index=chunk_index,
        content=content,
        similarity=similarity,
        language=language,
        script=script,
        source=source,
        metadata={"category": "test"},
    )


def create_mock_session():
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.execute = AsyncMock()
    return session


# ==============================================================================
# 1. HYBRID RETRIEVER & STORE INTEGRATION TESTS
# ==============================================================================

@pytest.mark.anyio
async def test_qdrant_retrieval():
    mock_qdrant = AsyncMock()
    hit = MagicMock()
    hit.id = "point_1"
    hit.score = 0.88
    hit.payload = {
        "document_id": "doc_1",
        "document_name": "kannada.txt",
        "chunk_index": 0,
        "content": "ಕನ್ನಡ ಸಾಹಿತ್ಯ ಮತ್ತು ಇತಿಹಾಸ.",
        "language": "kn",
        "script": "Kannada",
    }
    mock_qdrant.search_similar.return_value = [hit]

    retriever = HybridRetriever(pg_repo=AsyncMock(), qdrant_repo=mock_qdrant)
    candidates = await retriever.retrieve_candidates(
        query_vector=[0.05] * 1536,
        user_id=DEV_DEFAULT_USER_ID,
        candidate_k=5,
    )

    assert len(candidates) == 1
    assert candidates[0].document_name == "kannada.txt"
    assert candidates[0].similarity == 0.88
    assert candidates[0].source == "qdrant"


@pytest.mark.anyio
async def test_postgres_retrieval():
    mock_pg = AsyncMock()
    mock_session = create_mock_session()

    mock_chunk = MagicMock()
    mock_chunk.id = uuid.uuid4()
    mock_chunk.document_name = "postgres_doc.pdf"
    mock_chunk.chunk_index = 0
    mock_chunk.content = "PostgreSQL pgvector storage chunk."

    mock_pg.search_similar.return_value = [(mock_chunk, 0.12)]

    mock_qdrant = AsyncMock()
    mock_qdrant.search_similar.return_value = []

    retriever = HybridRetriever(pg_repo=mock_pg, qdrant_repo=mock_qdrant)
    candidates = await retriever.retrieve_candidates(
        query_vector=[0.01] * 1536,
        user_id=DEV_DEFAULT_USER_ID,
        session=mock_session,
        candidate_k=5,
    )

    assert len(candidates) == 1
    assert candidates[0].document_name == "postgres_doc.pdf"
    assert candidates[0].similarity == 0.88  # 1.0 - 0.12
    assert candidates[0].source == "postgres"


@pytest.mark.anyio
async def test_hybrid_retrieval_and_result_fusion():
    mock_qdrant = AsyncMock()
    hit = MagicMock()
    hit.id = "shared_chunk"
    hit.score = 0.80
    hit.payload = {
        "document_id": "doc_shared",
        "document_name": "shared.pdf",
        "chunk_index": 0,
        "content": "Shared document chunk.",
        "language": "en",
        "script": "Latin",
    }
    mock_qdrant.search_similar.return_value = [hit]

    mock_pg = AsyncMock()
    mock_chunk = MagicMock()
    mock_chunk.id = uuid.uuid4()
    mock_chunk.document_name = "shared.pdf"
    mock_chunk.chunk_index = 0
    mock_chunk.content = "Shared document chunk."
    # Cosine distance 0.25 -> similarity 0.75
    mock_pg.search_similar.return_value = [(mock_chunk, 0.25)]

    mock_session = create_mock_session()
    retriever = HybridRetriever(pg_repo=mock_pg, qdrant_repo=mock_qdrant)

    candidates = await retriever.retrieve_candidates(
        query_vector=[0.02] * 1536,
        user_id=DEV_DEFAULT_USER_ID,
        session=mock_session,
    )

    # Must be deduplicated into 1 candidate with fused score and "hybrid" source
    assert len(candidates) == 1
    c = candidates[0]
    assert c.document_name == "shared.pdf"
    assert c.source == "hybrid"
    # Fusion formula: max(0.80, 0.75) + 0.05 * min(0.80, 0.75) = 0.80 + 0.0375 = 0.8375
    assert c.similarity == 0.8375


@pytest.mark.anyio
async def test_duplicate_removal():
    mock_qdrant = AsyncMock()
    # 2 hits representing the same chunk
    hit1 = MagicMock(id="p1", score=0.85, payload={"document_name": "doc_a.txt", "chunk_index": 0, "content": "Sample"})
    hit2 = MagicMock(id="p2", score=0.82, payload={"document_name": "doc_a.txt", "chunk_index": 0, "content": "Sample"})
    mock_qdrant.search_similar.return_value = [hit1, hit2]

    retriever = HybridRetriever(pg_repo=AsyncMock(), qdrant_repo=mock_qdrant)
    candidates = await retriever.retrieve_candidates(
        query_vector=[0.01] * 1536,
        user_id=DEV_DEFAULT_USER_ID,
    )
    assert len(candidates) == 1


# ==============================================================================
# 2. RERANKING & SCORING TESTS
# ==============================================================================

def test_reranking_lexical_overlap():
    reranker = DevelopmentReranker()
    c1 = create_sample_candidate(chunk_id="c1", content="ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಬೆಂಗಳೂರು ನಗರ.", similarity=0.70)
    c2 = create_sample_candidate(chunk_id="c2", content="ಮೈಸೂರು ಅರಮನೆ ಸುಂದರವಾಗಿದೆ.", similarity=0.70)

    # Query with exact lexical match in c1 ("ಬೆಂಗಳೂರು")
    results = reranker.rerank(
        query="ಬೆಂಗಳೂರು ಮಾಹಿತಿ",
        candidates=[c1, c2],
        query_language="kn",
        query_script="Kannada",
    )

    assert len(results) == 2
    assert results[0].chunk_id == "c1"
    assert results[0].lexical_score > 0.0
    assert results[0].rerank_score > results[1].rerank_score


def test_reranking_language_and_script_matching():
    reranker = DevelopmentReranker()
    c_kannada = create_sample_candidate(chunk_id="c_kn", content="ಕನ್ನಡ ಪಠ್ಯ", language="kn", script="Kannada", similarity=0.80)
    c_english = create_sample_candidate(chunk_id="c_en", content="English text", language="en", script="Latin", similarity=0.80)

    results = reranker.rerank(
        query="ಕನ್ನಡ ಪರೀಕ್ಷೆ",
        candidates=[c_kannada, c_english],
        query_language="kn",
        query_script="Kannada",
    )

    assert results[0].chunk_id == "c_kn"
    assert results[0].language_match is True
    assert results[0].script_match is True
    assert results[0].rerank_score > results[1].rerank_score


def test_deterministic_reranking_order():
    reranker = DevelopmentReranker()
    candidates = [
        create_sample_candidate(chunk_id="c1", similarity=0.60, content="Content 1"),
        create_sample_candidate(chunk_id="c2", similarity=0.90, content="Content 2"),
        create_sample_candidate(chunk_id="c3", similarity=0.75, content="Content 3"),
    ]

    r1 = reranker.rerank("Query", candidates, "en", "Latin")
    r2 = reranker.rerank("Query", candidates, "en", "Latin")

    assert [x.chunk_id for x in r1] == [x.chunk_id for x in r2]
    assert r1[0].chunk_id == "c2"


# ==============================================================================
# 3. CONTEXT ASSEMBLY & BUDGET TESTS
# ==============================================================================

def test_context_assembler_structure():
    assembler = ContextAssembler(max_context_chars=4000, max_context_chunks=5)
    reranked = [
        RerankedChunk(
            chunk_id="c1",
            document_id="doc1",
            document_name="kannada.txt",
            chunk_index=0,
            content="ಕನ್ನಡ ಭಾಷೆ ಅತ್ಯಂತ ಪುರಾತನವಾದದ್ದು.",
            similarity=0.90,
            rerank_score=0.92,
            lexical_score=0.5,
            language_match=True,
            script_match=True,
            language="kn",
            script="Kannada",
            matched_signals=["vector_sim_0.90", "lang_match_kn"],
        )
    ]

    ctx = assembler.assemble_context(reranked)
    assert ctx.total_chunks == 1
    assert "### [Source: kannada.txt | ID: doc1 | Chunk: 0 | Lang: kn | Relevance: 0.92]" in ctx.context_text
    assert "ಕನ್ನಡ ಭಾಷೆ ಅತ್ಯಂತ ಪುರಾತನವಾದದ್ದು." in ctx.context_text
    assert ctx.truncated is False


def test_context_assembler_character_and_chunk_limits():
    assembler = ContextAssembler(max_context_chars=120, max_context_chunks=2)
    chunks = [
        RerankedChunk(
            chunk_id=f"c_{i}",
            document_id=f"d_{i}",
            document_name=f"doc_{i}.txt",
            chunk_index=0,
            content="A" * 80,
            similarity=0.8,
            rerank_score=0.8,
            language="en",
            script="Latin",
        )
        for i in range(5)
    ]

    ctx = assembler.assemble_context(chunks, max_chars=120, max_chunks=2)
    assert ctx.total_chunks <= 2
    assert ctx.total_characters <= 120
    assert ctx.truncated is True


# ==============================================================================
# 4. FILTERS & USER ISOLATION
# ==============================================================================

@pytest.mark.anyio
async def test_retrieval_minimum_similarity_filter():
    mock_qdrant = AsyncMock()
    hit_high = MagicMock(id="h1", score=0.85, payload={"document_name": "d1.txt", "chunk_index": 0, "content": "High score"})
    hit_low = MagicMock(id="h2", score=0.30, payload={"document_name": "d2.txt", "chunk_index": 0, "content": "Low score"})
    mock_qdrant.search_similar.return_value = [hit_high, hit_low]

    retriever = HybridRetriever(pg_repo=AsyncMock(), qdrant_repo=mock_qdrant)
    candidates = await retriever.retrieve_candidates(
        query_vector=[0.01] * 1536,
        user_id=DEV_DEFAULT_USER_ID,
        minimum_similarity=0.70,
    )

    assert len(candidates) == 1
    assert candidates[0].chunk_id == "h1"


@pytest.mark.anyio
async def test_retrieval_user_isolation():
    mock_qdrant = AsyncMock()
    mock_qdrant.search_similar.return_value = []

    retriever = HybridRetriever(pg_repo=AsyncMock(), qdrant_repo=mock_qdrant)
    candidates = await retriever.retrieve_candidates(
        query_vector=[0.01] * 1536,
        user_id="user_b_isolated",
    )
    assert len(candidates) == 0


@pytest.mark.anyio
async def test_retrieval_document_filtering():
    mock_qdrant = AsyncMock()
    hit1 = MagicMock(id="h1", score=0.9, payload={"document_name": "target.pdf", "chunk_index": 0, "content": "Match"})
    hit2 = MagicMock(id="h2", score=0.9, payload={"document_name": "other.pdf", "chunk_index": 0, "content": "Other"})
    mock_qdrant.search_similar.return_value = [hit1, hit2]

    retriever = HybridRetriever(pg_repo=AsyncMock(), qdrant_repo=mock_qdrant)
    candidates = await retriever.retrieve_candidates(
        query_vector=[0.01] * 1536,
        user_id=DEV_DEFAULT_USER_ID,
        document_name="target.pdf",
    )

    assert len(candidates) == 1
    assert candidates[0].document_name == "target.pdf"


# ==============================================================================
# 5. RETRIEVAL SERVICE & MULTILINGUAL PIPELINE TESTS
# ==============================================================================

@pytest.mark.anyio
async def test_retrieval_service_empty_and_short_queries():
    res_empty = await retrieval_service.retrieve(query="")
    assert res_empty.total_candidates == 0
    assert res_empty.results == []
    assert res_empty.context == ""

    res_spaces = await retrieval_service.retrieve(query="   ")
    assert res_spaces.total_candidates == 0


@pytest.mark.anyio
async def test_retrieval_service_multilingual_kannada():
    res = await retrieval_service.retrieve(
        query="ಕನ್ನಡ ಸಾಹಿತ್ಯ ಚರಿತ್ರೆ",
        user_id=DEV_DEFAULT_USER_ID,
        top_k=3,
    )
    assert res.detected_language == "kn"
    assert res.detected_script == "Kannada"
    assert res.processing_time_ms >= 0.0


@pytest.mark.anyio
async def test_retrieval_service_multilingual_hindi():
    res = await retrieval_service.retrieve(
        query="कृत्रिम बुद्धिमत्ता और मशीन लर्निंग",
        user_id=DEV_DEFAULT_USER_ID,
    )
    assert res.detected_language == "hi"
    assert res.detected_script == "Devanagari"


@pytest.mark.anyio
async def test_retrieval_service_multilingual_tamil():
    res = await retrieval_service.retrieve(
        query="செயற்கை நுண்ணறிவு",
        user_id=DEV_DEFAULT_USER_ID,
    )
    assert res.detected_language == "ta"
    assert res.detected_script == "Tamil"


@pytest.mark.anyio
async def test_retrieval_service_romanized_kannada():
    res = await retrieval_service.retrieve(
        query="kannada bhasha bagge mahiti kodi",
        user_id=DEV_DEFAULT_USER_ID,
    )
    assert res.detected_script == "Latin"
    assert res.processing_time_ms >= 0.0


# ==============================================================================
# 6. REST API INTEGRATION TESTS
# ==============================================================================

def test_api_retrieval_search_endpoint(client: TestClient):
    payload = {
        "query": "ಕನ್ನಡ ಭಾಷೆಯ ಬಗ್ಗೆ ಮಾಹಿತಿ ನೀಡಿ",
        "top_k": 5,
        "user_id": DEV_DEFAULT_USER_ID,
    }

    response = client.post("/api/v1/retrieval/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == payload["query"]
    assert data["language"] == "kn"
    assert data["script"] == "Kannada"
    assert "results" in data
    assert "context" in data
    assert data["processing_time_ms"] >= 0.0


def test_api_retrieval_context_endpoint(client: TestClient):
    payload = {
        "query": "Explain neural networks in Kannada",
        "top_k": 3,
        "max_context_chars": 2000,
        "user_id": DEV_DEFAULT_USER_ID,
    }

    response = client.post("/api/v1/retrieval/context", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == payload["query"]
    assert "context" in data
    assert "total_chunks" in data
    assert "total_characters" in data
    assert data["processing_time_ms"] >= 0.0
