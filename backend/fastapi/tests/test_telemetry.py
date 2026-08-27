def test_telemetry_snapshot(client):
    response = client.get("/api/v1/telemetry")
    assert response.status_code == 200
    data = response.json()
    assert "cpu_percent" in data
    assert "active_rag_documents" in data
    assert "active_mcp_tools" in data
    assert data["is_mock_data"] is True
