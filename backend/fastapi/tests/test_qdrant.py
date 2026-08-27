import pytest
from fastapi.testclient import TestClient

def test_qdrant_search_valid_dimensions(client: TestClient):
    valid_vector = [0.03] * 1536

    payload = {
        "collection_name": "nexora_documents",
        "query_vector": valid_vector,
        "top_k": 5,
        "user_id": "test-user-uuid"
    }

    response = client.post("/api/v1/qdrant/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert data["dimensions"] == 1536
    assert data["collection_name"] == "nexora_documents"
    assert isinstance(data["results"], list)

def test_qdrant_search_invalid_dimension_rejected(client: TestClient):
    invalid_vector = [0.03] * 256  # Invalid 256 dimensions

    payload = {
        "collection_name": "nexora_documents",
        "query_vector": invalid_vector,
        "top_k": 5
    }

    response = client.post("/api/v1/qdrant/search", json=payload)
    assert response.status_code == 422
    assert "Invalid vector dimension" in response.text

def test_qdrant_search_top_k_boundary_validation(client: TestClient):
    valid_vector = [0.03] * 1536

    # Test top_k > 50
    payload_exceed = {
        "collection_name": "nexora_documents",
        "query_vector": valid_vector,
        "top_k": 100
    }
    response_exceed = client.post("/api/v1/qdrant/search", json=payload_exceed)
    assert response_exceed.status_code == 422

    # Test top_k < 1
    payload_below = {
        "collection_name": "nexora_documents",
        "query_vector": valid_vector,
        "top_k": 0
    }
    response_below = client.post("/api/v1/qdrant/search", json=payload_below)
    assert response_below.status_code == 422

def test_qdrant_upsert_valid(client: TestClient):
    valid_vector = [0.05] * 1536

    payload = {
        "collection_name": "nexora_documents",
        "points": [
            {
                "id": "11111111-1111-1111-1111-111111111111",
                "vector": valid_vector,
                "payload": {
                    "user_id": "test-user-uuid",
                    "document_id": "system_design.pdf",
                    "chunk_index": 0,
                    "content": "Qdrant vector engine supports high dimensional indexing.",
                    "source": "manual_upload"
                }
            }
        ]
    }

    response = client.post("/api/v1/qdrant/upsert", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["upserted_count"] == 1
    assert data["collection_name"] == "nexora_documents"

def test_qdrant_list_collections(client: TestClient):
    response = client.get("/api/v1/qdrant/collections")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert data[0]["name"] == "nexora_documents"

def test_qdrant_health_readiness(client: TestClient):
    response = client.get("/api/v1/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert "vector_db" in data
    assert data["vector_db"]["status"] in ["healthy", "degraded"]
