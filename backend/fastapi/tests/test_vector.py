import pytest
from fastapi.testclient import TestClient
from services.vector_search_service import vector_search_service

def test_vector_search_valid_dimensions(client: TestClient):
    # Construct a valid 1536-dimensional dummy embedding
    valid_embedding = [0.01] * 1536

    payload = {
        "query_embedding": valid_embedding,
        "top_k": 5,
        "document_name": "test_doc.pdf"
    }

    response = client.post("/api/v1/vector/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert data["query_dimensions"] == 1536
    assert isinstance(data["results"], list)

def test_vector_search_invalid_dimension_rejected(client: TestClient):
    # Construct invalid 512-dimensional embedding
    invalid_embedding = [0.01] * 512

    payload = {
        "query_embedding": invalid_embedding,
        "top_k": 5
    }

    response = client.post("/api/v1/vector/search", json=payload)
    assert response.status_code == 422
    assert "Invalid vector dimension" in response.text

def test_vector_search_top_k_boundary_validation(client: TestClient):
    valid_embedding = [0.01] * 1536

    # Test top_k > 50 (max limit)
    payload_exceed = {
        "query_embedding": valid_embedding,
        "top_k": 100
    }
    response_exceed = client.post("/api/v1/vector/search", json=payload_exceed)
    assert response_exceed.status_code == 422

    # Test top_k < 1 (min limit)
    payload_below = {
        "query_embedding": valid_embedding,
        "top_k": 0
    }
    response_below = client.post("/api/v1/vector/search", json=payload_below)
    assert response_below.status_code == 422

def test_vector_ingest_valid(client: TestClient):
    valid_embedding = [0.02] * 1536

    payload = {
        "document_name": "ai_blueprint.pdf",
        "chunk_index": 0,
        "content": "NEXORA AI high performance vector search with pgvector.",
        "embedding": valid_embedding
    }

    response = client.post("/api/v1/vector/ingest", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert "ingested" in data["message"]

def test_similarity_score_conversion():
    # Distance = 0.0 -> Similarity = 1.0 (Identical vectors)
    assert vector_search_service.calculate_similarity_score(0.0) == 1.0

    # Distance = 0.12 -> Similarity = 0.88
    assert vector_search_service.calculate_similarity_score(0.12) == 0.88

    # Distance = 1.0 -> Similarity = 0.0 (Orthogonal vectors)
    assert vector_search_service.calculate_similarity_score(1.0) == 0.0

    # Distance = 1.5 -> Clamped to 0.0
    assert vector_search_service.calculate_similarity_score(1.5) == 0.0
