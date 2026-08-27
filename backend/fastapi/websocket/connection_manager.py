import json
from typing import Dict
from fastapi import WebSocket
from core.logging import logger
from schemas.ws_events import WSServerEvent, AIStateEnum, AIStateEventPayload

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, client_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info("WebSocket client connected", client_id=client_id, active_total=len(self.active_connections))
        
        # Send connection_ready event
        await self.send_event(client_id, WSServerEvent(
            event_type="connection_ready",
            payload={"client_id": client_id, "status": "CONNECTED"}
        ))

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info("WebSocket client disconnected", client_id=client_id, active_total=len(self.active_connections))

    async def send_event(self, client_id: str, event: WSServerEvent):
        if client_id in self.active_connections:
            ws = self.active_connections[client_id]
            await ws.send_text(event.model_dump_json())

    async def send_ai_state(self, client_id: str, state: AIStateEnum, message: str | None = None):
        payload = AIStateEventPayload(state=state, message=message).model_dump()
        await self.send_event(client_id, WSServerEvent(
            event_type="ai_state",
            payload=payload
        ))

    async def broadcast(self, event: WSServerEvent):
        for client_id in list(self.active_connections.keys()):
            await self.send_event(client_id, event)

ws_manager = ConnectionManager()
