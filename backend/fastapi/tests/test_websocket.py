def test_websocket_chat_connection(client):
    with client.websocket_connect("/api/v1/ws/chat?client_id=test_client_1") as websocket:
        # Check initial connection_ready event
        data = websocket.receive_json()
        assert data["event_type"] == "connection_ready"
        assert data["payload"]["status"] == "CONNECTED"

        # Send ping
        websocket.send_json({"event_type": "ping", "payload": {"timestamp": 123456}})
        data_pong = websocket.receive_json()
        assert data_pong["event_type"] == "pong"
