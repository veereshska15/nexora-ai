def test_health_liveness(client):
    response = client.get("/api/v1/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "NEXORA AI Engine"

def test_health_readiness(client):
    response = client.get("/api/v1/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert "database" in data
    assert "redis" in data
    assert "vector_db" in data
