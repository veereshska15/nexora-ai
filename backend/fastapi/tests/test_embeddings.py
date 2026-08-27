import numpy as np
import pytest
from fastapi.testclient import TestClient

from embeddings.embedding_factory import embedding_factory
from embeddings.embedding_service import embedding_service
from embeddings.providers.development_embedding import DevelopmentEmbeddingProvider
from documents.hybrid_ingestion_service import hybrid_ingestion_service


# ==============================================================================
# 1. CORE EMBEDDING PROVIDER & SERVICE TESTS
# ==============================================================================

def test_single_text_embedding():
    text = "NEXORA AI Multilingual Vector Intelligence Engine"
    res = embedding_service.embed_single(text)

    assert res.text == text
    assert isinstance(res.vector, list)
    assert len(res.vector) == 1536
    assert res.dimension == 1536
    assert res.provider == "development_deterministic"
    assert res.model == "development-1536"
    assert res.processing_time_ms >= 0.0


def test_batch_embedding():
    texts = [
        "First document text chunk for vector embedding.",
        "Second text chunk analyzing Indic language tokenization.",
        "Third text chunk verifying pgvector and Qdrant readiness.",
    ]
    results = embedding_service.embed_batch(texts)

    assert len(results) == 3
    for i, r in enumerate(results):
        assert r.text == texts[i]
        assert len(r.vector) == 1536
        assert r.dimension == 1536


def test_exact_1536_dimensions():
    provider = DevelopmentEmbeddingProvider(dimension=1536)
    vec = provider.embed_text("Dimension check text")
    assert len(vec) == 1536
    assert provider.dimension == 1536


def test_deterministic_output():
    text = "Deterministic vector reproducibility check in NEXORA AI."
    res1 = embedding_service.embed_single(text)
    res2 = embedding_service.embed_single(text)

    assert res1.vector == res2.vector


def test_vector_normalization():
    text = "Vector normalization unit length validation."
    res = embedding_service.embed_single(text)
    l2_norm = np.linalg.norm(res.vector)

    assert pytest.approx(l2_norm, rel=1e-3) == 1.0


# ==============================================================================
# 2. VALIDATION & ERROR HANDLING TESTS
# ==============================================================================

def test_empty_text_rejection():
    with pytest.raises(ValueError) as exc:
        embedding_service.embed_single("")
    assert "empty" in str(exc.value).lower()

    with pytest.raises(ValueError) as exc:
        embedding_service.embed_single("   \n\t  ")
    assert "empty" in str(exc.value).lower()


def test_long_input_rejection():
    huge_text = "A" * 50001
    with pytest.raises(ValueError) as exc:
        embedding_service.embed_single(huge_text)
    assert "exceeds maximum allowed limit" in str(exc.value)


def test_batch_size_validation():
    with pytest.raises(ValueError) as exc:
        embedding_service.embed_batch([])
    assert "empty" in str(exc.value).lower()

    huge_batch = [f"Text chunk {i}" for i in range(101)]
    with pytest.raises(ValueError) as exc:
        embedding_service.embed_batch(huge_batch)
    assert "exceeds maximum allowed limit" in str(exc.value)


# ==============================================================================
# 3. FACTORY & PROVIDER METADATA TESTS
# ==============================================================================

def test_provider_factory():
    dev1 = embedding_factory.get_provider("development")
    dev2 = embedding_factory.get_provider("development_deterministic")
    dev3 = embedding_factory.get_provider("default")
    fallback = embedding_factory.get_provider("non_existent_future_provider")

    assert dev1.provider_name == "development_deterministic"
    assert dev2.provider_name == "development_deterministic"
    assert dev3.provider_name == "development_deterministic"
    assert fallback.provider_name == "development_deterministic"


def test_development_provider_metadata():
    prov = embedding_factory.get_provider("development_deterministic")
    assert prov.provider_name == "development_deterministic"
    assert prov.model_name == "development-1536"
    assert prov.dimension == 1536


# ==============================================================================
# 4. MULTILINGUAL & INDIC VECTOR TESTS
# ==============================================================================

def test_kannada_embedding():
    kannada_text = "ನಮಸ್ಕಾರ, ಹೇಗಿದ್ದೀರಾ? ಇದು ಕನ್ನಡ ವೆಕ್ಟರ್ ಎಂಬೆಡ್ಡಿಂಗ್ ಪರೀಕ್ಷೆ."
    res = embedding_service.embed_single(kannada_text)

    assert len(res.vector) == 1536
    assert pytest.approx(np.linalg.norm(res.vector), rel=1e-3) == 1.0


def test_hindi_embedding():
    hindi_text = "हिन्दी में आपका स्वागत है। यह वेक्टर एम्बेडिंग का परीक्षण है।"
    res = embedding_service.embed_single(hindi_text)

    assert len(res.vector) == 1536
    assert pytest.approx(np.linalg.norm(res.vector), rel=1e-3) == 1.0


def test_english_embedding():
    english_text = "NEXORA AI is a cutting-edge multimodal platform."
    res = embedding_service.embed_single(english_text)

    assert len(res.vector) == 1536
    assert pytest.approx(np.linalg.norm(res.vector), rel=1e-3) == 1.0


def test_mixed_kannada_english_embedding():
    mixed_text = "ನಮಸ್ಕಾರ Welcome to NEXORA AI Multilingual Neural Platform."
    res = embedding_service.embed_single(mixed_text)

    assert len(res.vector) == 1536
    assert pytest.approx(np.linalg.norm(res.vector), rel=1e-3) == 1.0


# ==============================================================================
# 5. HYBRID DOCUMENT ENRICHMENT TESTS
# ==============================================================================

def test_hybrid_document_enrichment():
    text = (
        "ಕನ್ನಡ ಭಾಷೆಯು ಭಾರತದ ಪ್ರಾಚೀನ ಭಾಷೆಗಳಲ್ಲಿ ಒಂದಾಗಿದೆ.\n\n"
        "ನೆಕ್ಸೋರಾ ಎಐ ಕನ್ನಡ ಭಾಷೆಗೆ ಪ್ರಥಮ ಆದ್ಯತೆ ನೀಡುತ್ತದೆ.\n\n"
        "This platform delivers enterprise-grade multimodal AI."
    )
    res = hybrid_ingestion_service.enrich_document(
        text=text,
        document_name="kannada_doc.txt",
        strategy="recursive",
        chunk_size=100,
        chunk_overlap=10,
    )

    assert res.total_chunks >= 2
    assert res.embedding_dimension == 1536
    assert res.embedding_provider == "development_deterministic"
    assert len(res.chunks) == res.total_chunks

    for c in res.chunks:
        assert len(c.embedding) == 1536
        assert c.token_count > 0
        assert c.language in ["kn", "en"]
        assert c.embedding_model == "development-1536"


def test_chunk_metadata_preservation():
    text = "Secure enterprise financial audit document 2026."
    custom_metadata = {"tenant_id": "tenant_42", "classification": "TOP_SECRET"}

    res = hybrid_ingestion_service.enrich_document(
        text=text,
        document_name="audit.txt",
        metadata=custom_metadata,
    )

    assert res.total_chunks >= 1
    assert res.chunks[0].metadata["tenant_id"] == "tenant_42"
    assert res.chunks[0].metadata["classification"] == "TOP_SECRET"


def test_embedding_metadata_preservation():
    text = "Vector telemetry verification text."
    res = hybrid_ingestion_service.enrich_document(
        text=text,
        document_name="telemetry.txt",
    )

    assert res.embedding_dimension == 1536
    assert res.embedding_provider == "development_deterministic"
    assert res.chunks[0].embedding_dimension == 1536
    assert res.chunks[0].embedding_provider == "development_deterministic"


# ==============================================================================
# 6. REST API ENDPOINT INTEGRATION TESTS
# ==============================================================================

def test_api_embedding_endpoint(client: TestClient):
    payload = {"text": "ನಮಸ್ಕಾರ ಹೇಗಿದ್ದೀರಾ?"}
    response = client.post("/api/v1/embeddings/embed", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["dimension"] == 1536
    assert data["provider"] == "development_deterministic"
    assert data["model"] == "development-1536"
    assert len(data["vector"]) == 1536
    assert data["processing_time_ms"] >= 0.0


def test_api_document_enrichment_endpoint(client: TestClient):
    payload = {
        "text": "ನಮಸ್ಕಾರ ಹೇಗಿದ್ದೀರಾ?\n\nNEXORA AI Multilingual Ingestion Platform.",
        "document_name": "api_enrich_test.txt",
        "strategy": "recursive",
        "chunk_size": 1000,
        "chunk_overlap": 100,
        "metadata": {"source": "integration_test"},
    }

    response = client.post("/api/v1/documents/enrich", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["document_name"] == "api_enrich_test.txt"
    assert data["embedding_dimension"] == 1536
    assert data["total_chunks"] >= 1
    assert len(data["chunks"]) == data["total_chunks"]
    assert len(data["chunks"][0]["embedding"]) == 1536
    assert data["chunks"][0]["metadata"]["source"] == "integration_test"
    assert data["processing_time_ms"] >= 0.0
