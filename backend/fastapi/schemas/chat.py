from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from datetime import datetime

class ChatMessageRequest(BaseModel):
    conversation_id: UUID | None = Field(default_factory=uuid4)
    content: str = Field(..., min_length=1, example="Explain neural networks in Kannada.")
    language: str = Field("English", example="English")
    user_id: str | None = Field("guest_user", example="guest_user")

class ChatMessageResponse(BaseModel):
    message_id: UUID = Field(default_factory=uuid4)
    conversation_id: UUID
    sender: str = Field("assistant", example="assistant")
    content: str
    tokens_used: int = Field(0, example=42)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_mock: bool = Field(True, description="Indicates mock development response")
