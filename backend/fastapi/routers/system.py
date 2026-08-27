from fastapi import APIRouter
from core.config import settings

router = APIRouter(prefix="/system", tags=["System Information"])

@router.get("")
async def get_system_info():
    """System information & active feature matrix."""
    return {
        "platform": "NEXORA AI",
        "environment": settings.NEXORA_ENV,
        "features": {
            "langgraph_dag": "active",
            "fastmcp_tools": "active",
            "rag_vector_search": "active",
            "custom_3d_cnn": "ready",
            "webrtc_speech": "active"
        }
    }
