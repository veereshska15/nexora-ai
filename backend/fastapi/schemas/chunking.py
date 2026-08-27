from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChunkSchema(BaseModel):
    """
    Representation of an individual chunk within the API response.
    """
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document identifier")
    document_name: str = Field(..., description="Parent document name")
    chunk_index: int = Field(..., ge=0, description="Chunk sequence index")
    content: str = Field(..., description="Chunk text content")
    character_count: int = Field(..., ge=1, description="Character count")
    token_count: int = Field(..., ge=0, description="Multilingual subword token count")
    start_offset: int = Field(..., ge=0, description="Start character offset")
    end_offset: int = Field(..., ge=0, description="End character offset")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Preserved metadata")


class ChunkTextRequest(BaseModel):
    """
    Request payload for document text chunking.
    """
    text: str = Field(..., min_length=1, description="Raw document text to chunk", json_schema_extra={"example": "NEXORA AI is a cutting-edge multimodal platform."})
    document_name: str = Field("document.txt", description="Document filename/title")
    document_id: Optional[str] = Field(None, description="Optional document UUID")
    strategy: str = Field("recursive", description="Chunking strategy: 'character', 'recursive', or 'token'")
    chunk_size: int = Field(1000, gt=0, description="Target maximum size per chunk (chars or tokens)")
    chunk_overlap: int = Field(100, ge=0, description="Overlap between consecutive chunks")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Custom document metadata to attach")


class ChunkTextResponse(BaseModel):
    """
    Response payload containing structured chunks and processing telemetry.
    """
    document_id: str = Field(..., description="Document identifier")
    document_name: str = Field(..., description="Document title/filename")
    strategy: str = Field(..., description="Applied chunking strategy")
    total_chunks: int = Field(..., ge=0, description="Total number of chunks produced")
    chunks: List[ChunkSchema] = Field(default_factory=list, description="Ordered list of text chunks")
    processing_time_ms: float = Field(0.0, ge=0.0, description="Chunking latency in milliseconds")
