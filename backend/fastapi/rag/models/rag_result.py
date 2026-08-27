from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from rag.models.citation import Citation


class RAGResponse(BaseModel):
    """
    Standardized result model for End-to-End Grounded RAG Query Orchestration.
    """
    query: str = Field(..., description="User question / query")
    answer: str = Field(..., description="Grounded answer synthesized from verified retrieved context")
    language: str = Field("en", description="Predicted/detected language code")
    script: str = Field("Latin", description="Predicted/detected script")
    grounded: bool = Field(True, description="Whether answer is strictly grounded in retrieved documents")
    grounding_confidence: float = Field(..., ge=0.0, le=1.0, description="Verification confidence score")
    citations: List[Citation] = Field(default_factory=list, description="Verified source citations")
    retrieved_chunks: int = Field(0, ge=0, description="Count of retrieved chunks used in context")
    provider: str = Field("development", description="LLM provider name")
    model: str = Field("development-grounded", description="LLM model name")
    processing_time_ms: float = Field(0.0, ge=0.0, description="Total orchestration latency in milliseconds")
    warnings: List[str] = Field(default_factory=list, description="Grounding or safety guardrail warnings")
