import uuid
from fastapi import APIRouter, WebSocket
from websocket.ws_chat_handler import handle_ws_chat

router = APIRouter(prefix="/ws", tags=["Real-Time WebSocket"])

@router.websocket("/chat")
async def websocket_chat_endpoint(websocket: WebSocket, client_id: str | None = None):
    """Bidirectional WebSocket streaming endpoint for real-time chat & voice state sync."""
    cid = client_id or str(uuid.uuid4())
    await handle_ws_chat(cid, websocket)
