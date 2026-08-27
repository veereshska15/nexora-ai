import asyncio
import json
from fastapi import WebSocket, WebSocketDisconnect
from core.logging import logger
from websocket.connection_manager import ws_manager
from schemas.ws_events import WSClientEvent, WSServerEvent, AIStateEnum

async def handle_ws_chat(client_id: str, websocket: WebSocket):
    await ws_manager.connect(client_id, websocket)
    try:
        while True:
            data_str = await websocket.receive_text()
            try:
                data_json = json.loads(data_str)
                client_event = WSClientEvent(**data_json)
                await _process_client_event(client_id, client_event)
            except Exception as parse_err:
                logger.warning("Invalid WebSocket client event payload", client_id=client_id, error=str(parse_err))
                await ws_manager.send_event(client_id, WSServerEvent(
                    event_type="error",
                    payload={"message": "Invalid JSON event schema", "details": str(parse_err)}
                ))
    except WebSocketDisconnect:
        ws_manager.disconnect(client_id)
    except Exception as e:
        logger.error("WebSocket connection error", client_id=client_id, error=str(e))
        ws_manager.disconnect(client_id)

async def _process_client_event(client_id: str, event: WSClientEvent):
    event_type = event.event_type
    payload = event.payload

    if event_type == "ping":
        await ws_manager.send_event(client_id, WSServerEvent(
            event_type="pong",
            payload={"timestamp": payload.get("timestamp")}
        ))
        return

    if event_type == "connection_init":
        await ws_manager.send_ai_state(client_id, AIStateEnum.IDLE, "Connection initialized")
        return

    if event_type == "voice_start":
        await ws_manager.send_ai_state(client_id, AIStateEnum.LISTENING, "VAD Audio Listening Active")
        return

    if event_type == "voice_end":
        await ws_manager.send_ai_state(client_id, AIStateEnum.THINKING, "VAD Audio Transcribing")
        await asyncio.sleep(0.5)
        await ws_manager.send_ai_state(client_id, AIStateEnum.SPEAKING, "Synthesizing Speech")
        await asyncio.sleep(0.5)
        await ws_manager.send_ai_state(client_id, AIStateEnum.IDLE, "Ready")
        return

    if event_type == "user_message":
        user_text = payload.get("content", "")
        # 1. State -> Thinking
        await ws_manager.send_ai_state(client_id, AIStateEnum.THINKING, "LangGraph Intent Classification")
        await asyncio.sleep(0.3)

        # 2. State -> Processing
        await ws_manager.send_ai_state(client_id, AIStateEnum.PROCESSING, "RAG & MCP Tool Execution")
        await asyncio.sleep(0.3)

        # 3. Stream Response Tokens (Mock Development Stream)
        response_text = f"NEXORA AI Server received: '{user_text}'. (Phase 04 Mock Engine)"
        words = response_text.split(" ")

        await ws_manager.send_ai_state(client_id, AIStateEnum.SPEAKING, "Streaming Tokens")
        for word in words:
            await ws_manager.send_event(client_id, WSServerEvent(
                event_type="token",
                payload={"token": word + " "}
            ))
            await asyncio.sleep(0.05)

        # 4. State -> Success -> Idle
        await ws_manager.send_event(client_id, WSServerEvent(
            event_type="message_complete",
            payload={"full_text": response_text, "is_mock": True}
        ))
        await ws_manager.send_ai_state(client_id, AIStateEnum.SUCCESS, "Response Completed")
        await asyncio.sleep(0.5)
        await ws_manager.send_ai_state(client_id, AIStateEnum.IDLE, "Ready")
