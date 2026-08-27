from fastapi import APIRouter
from schemas.chat import ChatMessageRequest, ChatMessageResponse
from services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["AI Chat"])

@router.post("/message", response_model=ChatMessageResponse)
async def send_chat_message(request: ChatMessageRequest):
    """Post HTTP chat message."""
    return await ChatService.process_chat_message(request)
