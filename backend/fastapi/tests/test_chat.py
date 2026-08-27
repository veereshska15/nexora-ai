def test_chat_message(client):
    payload = {
        "content": "Explain neural networks in Kannada.",
        "language": "English"
    }
    response = client.post("/api/v1/chat/message", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "content" in data
    assert data["sender"] == "assistant"
    assert data["is_mock"] is True
