from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class AgentState(BaseModel):
    """
    State schema for the LangGraph cyclic workflow engine in NEXORA AI.
    Tracks state transitions across input safety checks, language detection,
    intent classification, RAG retrieval, tool execution, and output guardrails.
    """
    conversation_id: str
    user_id: str
    tenant_id: str
    input_text: str
    detected_language: str = "English"
    intent: str = "chat"
    is_safe: bool = True
    safety_flags: List[str] = Field(default_factory=list)
    rag_context: List[Dict[str, Any]] = Field(default_factory=list)
    vision_features: Optional[Dict[str, Any]] = None
    gesture_result: Optional[Dict[str, Any]] = None
    active_mcp_tool: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[Dict[str, Any]] = None
    messages: List[Dict[str, str]] = Field(default_factory=list)
    final_response: Optional[str] = None
    tokens_used: int = 0
