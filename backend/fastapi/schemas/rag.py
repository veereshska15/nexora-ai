from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CitationSchema(BaseModel):
    """
    Schema for verified source document citations.
    """
    citation_id: int = Field(..., ge=1, description="1-indexed citation number")
    marker: str = Field(..., description="Citation marker string (e.g. '[1]')")
    document_id: str = Field(..., description="Document identifier")
    document_name: str = Field(..., description="Document filename/title")
    chunk_id: str = Field(..., description="Chunk identifier")
    chunk_index: int = Field(..., ge=0, description="Index of chunk in document")
    content_snippet: str = Field(..., description="Text excerpt from chunk")
    language: str = Field("en", description="Language code")
    script: str = Field("Latin", description="Script name")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Rerank relevance score")


class RAGQueryRequest(BaseModel):
    """
    Request schema for grounded RAG query generation.
    """
    query: str = Field(..., min_length=1, description="User question / search query", json_schema_extra={"example": "ಕನ್ನಡ ಭಾಷೆಯ ಬಗ್ಗೆ ವಿವರಿಸಿ"})
    top_k: int = Field(5, ge=1, le=50, description="Number of top chunks to include in context")
    candidate_k: int = Field(20, ge=1, le=100, description="Candidate search depth")
    minimum_similarity: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity threshold")
    max_context_chars: int = Field(4000, ge=100, le=20000, description="Max character budget for context")
    document_id: Optional[str] = Field(None, description="Optional document filter")
    document_name: Optional[str] = Field(None, description="Optional document name filter")
    language: Optional[str] = Field(None, description="Optional language filter")
    user_id: Optional[str] = Field(None, description="User scope (overridden by auth identity when available)")


class RAGQueryResponse(BaseModel):
    """
    Response schema for grounded RAG query generation.
    """
    query: str = Field(..., description="User query")
    answer: str = Field(..., description="Grounded answer text")
    language: str = Field(..., description="Detected query language code")
    script: str = Field(..., description="Detected query script")
    grounded: bool = Field(..., description="Whether answer is verified as grounded in context")
    grounding_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    citations: List[CitationSchema] = Field(default_factory=list, description="Verified source citations")
    retrieved_chunks: int = Field(..., ge=0, description="Count of retrieved chunks used")
    provider: str = Field(..., description="LLM provider name")
    model: str = Field(..., description="LLM model name")
    processing_time_ms: float = Field(0.0, ge=0.0, description="Latency in milliseconds")
    warnings: List[str] = Field(default_factory=list, description="Safety or grounding warnings")
