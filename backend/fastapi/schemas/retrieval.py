from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RerankedChunkSchema(BaseModel):
    """
    Schema for an individual reranked chunk in retrieval responses.
    """
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document identifier")
    document_name: str = Field(..., description="Parent document name")
    chunk_index: int = Field(..., ge=0, description="Chunk sequence index")
    content: str = Field(..., description="Text content")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Original vector similarity")
    rerank_score: float = Field(..., ge=0.0, le=1.0, description="Composite rerank score")
    lexical_score: float = Field(0.0, ge=0.0, le=1.0, description="Lexical token overlap score")
    language_match: bool = Field(False, description="Whether chunk language matches query language")
    script_match: bool = Field(False, description="Whether chunk script matches query script")
    language: str = Field("en", description="Language code")
    script: str = Field("Latin", description="Script name")
    matched_signals: List[str] = Field(default_factory=list, description="Matched relevance signals")


class RetrievalSearchRequest(BaseModel):
    """
    Request schema for hybrid retrieval and reranking.
    """
    query: str = Field(..., min_length=1, description="Search query string", json_schema_extra={"example": "ಕನ್ನಡ ಭಾಷೆಯ ಬಗ್ಗೆ ಮಾಹಿತಿ ನೀಡಿ"})
    top_k: int = Field(5, ge=1, le=50, description="Number of top reranked chunks to return")
    candidate_k: int = Field(20, ge=1, le=100, description="Number of initial candidates to pull from vector stores")
    minimum_similarity: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity threshold")
    document_id: Optional[str] = Field(None, description="Optional document ID filter")
    document_name: Optional[str] = Field(None, description="Optional document name filter")
    language: Optional[str] = Field(None, description="Optional language filter")
    script: Optional[str] = Field(None, description="Optional script filter")
    max_context_chars: int = Field(4000, ge=100, le=20000, description="Maximum characters for assembled context")
    user_id: Optional[str] = Field(None, description="User scope (defaults to authenticated user or dev default)")


class RetrievalSearchResponse(BaseModel):
    """
    Response schema for hybrid retrieval and reranking.
    """
    query: str = Field(..., description="Original search query")
    language: str = Field(..., description="Detected query language")
    script: str = Field(..., description="Detected query script")
    total_candidates: int = Field(..., ge=0, description="Candidates retrieved before reranking")
    results: List[RerankedChunkSchema] = Field(default_factory=list, description="Top-K reranked chunks")
    context: str = Field("", description="Assembled RAG context string")
    processing_time_ms: float = Field(0.0, ge=0.0, description="Total pipeline latency in ms")


class RetrievalContextRequest(BaseModel):
    """
    Request schema to retrieve only assembled RAG context for a query.
    """
    query: str = Field(..., min_length=1, description="Search query string", json_schema_extra={"example": "ಕನ್ನಡ ಭಾಷೆಯ ಇತಿಹಾಸ"})
    top_k: int = Field(5, ge=1, le=50, description="Max chunks to include in context")
    max_context_chars: int = Field(4000, ge=100, le=20000, description="Maximum character budget")
    user_id: Optional[str] = Field(None, description="User scope")


class RetrievalContextResponse(BaseModel):
    """
    Response schema containing assembled RAG context and metadata.
    """
    query: str = Field(..., description="Search query string")
    context: str = Field(..., description="Formatted markdown RAG context ready for prompt injection")
    total_chunks: int = Field(..., ge=0, description="Number of chunks included in context")
    total_characters: int = Field(..., ge=0, description="Total character count")
    truncated: bool = Field(False, description="Whether context was budget-constrained")
    processing_time_ms: float = Field(0.0, ge=0.0, description="Latency in milliseconds")
