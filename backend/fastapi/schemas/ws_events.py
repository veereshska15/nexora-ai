from pydantic import BaseModel, Field
from typing import Literal, Any
from enum import Enum

class AIStateEnum(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    SUCCESS = "success"
    ERROR = "error"

# Client -> Server WebSocket Events
ClientEventType = Literal[
    "connection_init",
    "user_message",
    "voice_start",
    "voice_end",
    "typing",
    "ping"
]

# Server -> Client WebSocket Events
ServerEventType = Literal[
    "connection_ready",
    "ai_state",
    "token",
    "message_complete",
    "error",
    "telemetry",
    "pong"
]

class WSClientEvent(BaseModel):
    event_type: ClientEventType
    payload: dict[str, Any] = Field(default_factory=dict)

class WSServerEvent(BaseModel):
    event_type: ServerEventType
    payload: dict[str, Any] = Field(default_factory=dict)

class AIStateEventPayload(BaseModel):
    state: AIStateEnum
    message: str | None = None
