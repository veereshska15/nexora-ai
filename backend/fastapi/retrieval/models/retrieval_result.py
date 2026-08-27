from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CandidateChunk(BaseModel):
    """
    Raw candidate chunk retrieved from PostgreSQL pgvector or Qdrant vector database.
    """
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document identifier")
    document_name: str = Field(..., description="Parent document filename/title")
    chunk_index: int = Field(..., ge=0, description="Sequence index of chunk in document")
    content: str = Field(..., description="Text content of the chunk")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Vector cosine similarity score")
    language: str = Field("en", description="Detected language code")
    script: str = Field("Latin", description="Detected script")
    source: str = Field("qdrant", description="Source store: 'qdrant', 'postgres', or 'hybrid'")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom document metadata")


class RerankedChunk(BaseModel):
    """
    Chunk that has undergone multi-signal deterministic reranking.
    """
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document identifier")
    document_name: str = Field(..., description="Parent document filename/title")
    chunk_index: int = Field(..., ge=0, description="Sequence index of chunk")
    content: str = Field(..., description="Chunk content")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Original vector similarity score")
    rerank_score: float = Field(..., ge=0.0, le=1.0, description="Composite rerank score")
    lexical_score: float = Field(0.0, ge=0.0, le=1.0, description="Lexical token overlap score")
    language_match: bool = Field(False, description="Whether chunk language matches query language")
    script_match: bool = Field(False, description="Whether chunk script matches query script")
    language: str = Field("en", description="Language code")
    script: str = Field("Latin", description="Script name")
    matched_signals: List[str] = Field(default_factory=list, description="List of matched relevance signals")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary")


class AssembledContext(BaseModel):
    """
    Structured context assembled for RAG prompt consumption.
    """
    context_text: str = Field("", description="Formatted context string ready for LLM prompt")
    total_chunks: int = Field(0, ge=0, description="Total chunks included in context")
    total_characters: int = Field(0, ge=0, description="Total character count of assembled context")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source attribution metadata")
    truncated: bool = Field(False, description="Whether context was truncated due to budget constraints")


class RetrievalResult(BaseModel):
    """
    Comprehensive result container for hybrid retrieval, reranking, and context assembly.
    """
    query: str = Field(..., description="User search query")
    detected_language: str = Field("en", description="Query language detected by NLP pipeline")
    detected_script: str = Field("Latin", description="Query script detected by NLP pipeline")
    total_candidates: int = Field(0, ge=0, description="Total candidate chunks found before reranking")
    results: List[RerankedChunk] = Field(default_factory=list, description="Top-K reranked chunks")
    context: str = Field("", description="Assembled context text")
    assembled_context: AssembledContext = Field(default_factory=AssembledContext, description="Assembled context object")
    processing_time_ms: float = Field(0.0, ge=0.0, description="Total retrieval pipeline latency")
