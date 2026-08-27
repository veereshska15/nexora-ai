from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChunkPersistenceRecord(BaseModel):
    """
    Persistence telemetry for an individual chunk across PostgreSQL and Qdrant.
    """
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document identifier")
    document_name: str = Field(..., description="Parent document title/name")
    chunk_index: int = Field(..., ge=0, description="Chunk sequence index")
    content: str = Field(..., description="Chunk text content")
    language: str = Field(..., description="Detected language code")
    script: str = Field(..., description="Detected script")
    postgres_persisted: bool = Field(..., description="Whether stored in PostgreSQL pgvector")
    qdrant_persisted: bool = Field(..., description="Whether stored in Qdrant vector database")
    status: str = Field("persisted", description="Status string: persisted, partial, failed")


class DocumentPersistenceResult(BaseModel):
    """
    Result container for document batch vector persistence.
    """
    document_id: str = Field(..., description="Document identifier")
    document_name: str = Field(..., description="Document filename/title")
    user_id: str = Field(..., description="Authenticated user scope")
    total_chunks: int = Field(..., ge=0, description="Total chunks submitted")
    postgres_success_count: int = Field(..., ge=0, description="Chunks persisted to PostgreSQL")
    qdrant_success_count: int = Field(..., ge=0, description="Chunks persisted to Qdrant")
    status: str = Field(..., description="Overall status (success, partial_failure, failed)")
    chunks: List[ChunkPersistenceRecord] = Field(default_factory=list, description="Per-chunk persistence details")
    processing_time_ms: float = Field(0.0, ge=0.0, description="Persistence latency in milliseconds")


class DocumentPersistenceStatus(BaseModel):
    """
    Current persistence status for a document.
    """
    document_id: str = Field(..., description="Document identifier")
    document_name: str = Field(..., description="Document filename/title")
    user_id: str = Field(..., description="User scope")
    postgres_chunk_count: int = Field(..., ge=0, description="Count of chunks in PostgreSQL")
    qdrant_chunk_count: int = Field(..., ge=0, description="Count of chunks in Qdrant")
    persisted: bool = Field(..., description="Whether document is persisted in vector stores")
    last_updated: Optional[str] = Field(None, description="ISO timestamp of last update")


class DocumentSearchResult(BaseModel):
    """
    Semantic search hit result.
    """
    chunk_id: str = Field(..., description="Chunk identifier")
    document_id: str = Field(..., description="Document identifier")
    document_name: str = Field(..., description="Document filename")
    content: str = Field(..., description="Chunk text content")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score [0.0, 1.0]")
    language: str = Field(..., description="Language code")
    script: str = Field(..., description="Script name")


class DocumentSearchResponse(BaseModel):
    """
    Semantic search response container.
    """
    query: str = Field(..., description="User search query text")
    total_results: int = Field(..., ge=0, description="Total results found")
    results: List[DocumentSearchResult] = Field(default_factory=list, description="List of similarity hits")
    processing_time_ms: float = Field(0.0, ge=0.0, description="Search latency in milliseconds")
