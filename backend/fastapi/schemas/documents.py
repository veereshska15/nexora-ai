from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from schemas.embeddings import EnrichedChunkSchema
from documents.persistence.models.persistence_result import (
    ChunkPersistenceRecord,
    DocumentSearchResult,
)


class DocumentExtractionResponse(BaseModel):
    """
    Response schema returned by the document extraction endpoint.
    """
    filename: str = Field(..., description="Name of the uploaded file")
    file_type: str = Field(..., description="File extension / format (txt, pdf, docx, csv)")
    file_size: int = Field(..., ge=0, description="Size of the uploaded file in bytes")
    extracted_text: str = Field(..., description="Extracted raw text content")
    character_count: int = Field(..., ge=0, description="Total characters extracted")
    page_count: Optional[int] = Field(None, description="Total pages if applicable (PDF)")
    extraction_success: bool = Field(True, description="Whether text extraction succeeded")
    processing_time_ms: float = Field(0.0, ge=0.0, description="Processing duration in milliseconds")
    error_message: Optional[str] = Field(None, description="Diagnostic error details if extraction failed")


class PersistDocumentRequest(BaseModel):
    """
    Request payload to persist enriched document chunks into PostgreSQL pgvector and Qdrant.
    """
    document_id: str = Field(..., description="Document identifier")
    document_name: str = Field(..., description="Document filename/title")
    user_id: Optional[str] = Field(None, description="Scoped user ID (defaults to authenticated identity or dev fallback)")
    chunks: List[EnrichedChunkSchema] = Field(..., min_length=1, description="List of enriched chunks with 1536d vector embeddings")


class PersistDocumentResponse(BaseModel):
    """
    Response payload for document vector persistence.
    """
    document_id: str = Field(..., description="Document identifier")
    document_name: str = Field(..., description="Document title")
    user_id: str = Field(..., description="User scope")
    total_chunks: int = Field(..., ge=0, description="Total chunks submitted")
    postgres_success_count: int = Field(..., ge=0, description="Chunks persisted to pgvector")
    qdrant_success_count: int = Field(..., ge=0, description="Chunks persisted to Qdrant")
    status: str = Field(..., description="Status (success, partial_failure, failed)")
    chunks: List[ChunkPersistenceRecord] = Field(default_factory=list, description="Per-chunk persistence details")
    processing_time_ms: float = Field(0.0, ge=0.0, description="Processing latency in milliseconds")


class DocumentStatusResponse(BaseModel):
    """
    Response schema for document persistence status.
    """
    document_id: str = Field(..., description="Document identifier")
    document_name: str = Field(..., description="Document filename")
    user_id: str = Field(..., description="User scope")
    postgres_chunk_count: int = Field(..., ge=0, description="Chunks in PostgreSQL")
    qdrant_chunk_count: int = Field(..., ge=0, description="Chunks in Qdrant")
    persisted: bool = Field(..., description="Whether document exists in vector stores")
    last_updated: Optional[str] = Field(None, description="Last update timestamp")


class DocumentSearchQueryRequest(BaseModel):
    """
    Request schema for semantic vector similarity search.
    """
    query: str = Field(..., min_length=1, description="Search query string", json_schema_extra={"example": "ಕನ್ನಡ ಭಾಷೆ"})
    top_k: int = Field(5, ge=1, le=50, description="Maximum number of nearest neighbor hits")
    user_id: Optional[str] = Field(None, description="Optional user scope (defaults to authenticated scope)")
    document_id: Optional[str] = Field(None, description="Optional document filter")


class DocumentSearchQueryResponse(BaseModel):
    """
    Response schema for semantic vector similarity search.
    """
    query: str = Field(..., description="Search query string")
    total_results: int = Field(..., ge=0, description="Total matching chunks returned")
    results: List[DocumentSearchResult] = Field(default_factory=list, description="Ranked similarity search hits")
    processing_time_ms: float = Field(0.0, ge=0.0, description="Search latency in milliseconds")
