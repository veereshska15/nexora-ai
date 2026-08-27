from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    """
    Request schema for single text vector embedding generation.
    """
    text: str = Field(..., min_length=1, description="Input text to embed", json_schema_extra={"example": "ನಮಸ್ಕಾರ ಹೇಗಿದ್ದೀರಾ?"})
    provider: Optional[str] = Field("development_deterministic", description="Embedding provider name")


class EmbeddingResponse(BaseModel):
    """
    Response schema containing generated vector embedding and metadata.
    """
    dimension: int = Field(1536, description="Embedding vector dimensionality")
    provider: str = Field(..., description="Active embedding provider name")
    model: str = Field(..., description="Active embedding model name")
    vector: List[float] = Field(..., description="1536d normalized vector array")
    processing_time_ms: float = Field(0.0, ge=0.0, description="Latency in milliseconds")


class BatchEmbeddingRequest(BaseModel):
    """
    Request schema for batch text vector embedding generation.
    """
    texts: List[str] = Field(..., min_length=1, description="List of text strings to embed")
    provider: Optional[str] = Field("development_deterministic", description="Embedding provider name")


class BatchEmbeddingResponse(BaseModel):
    """
    Response schema containing batch generated vector embeddings.
    """
    total_embeddings: int = Field(..., ge=1, description="Total number of vectors generated")
    dimension: int = Field(1536, description="Embedding vector dimensionality")
    provider: str = Field(..., description="Active embedding provider name")
    model: str = Field(..., description="Active embedding model name")
    embeddings: List[List[float]] = Field(..., description="List of 1536d normalized vector arrays")
    processing_time_ms: float = Field(0.0, ge=0.0, description="Total batch latency in milliseconds")


class EnrichedChunkSchema(BaseModel):
    """
    Schema representing a text chunk enriched with language metadata and vector embeddings.
    """
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document identifier")
    document_name: str = Field(..., description="Document filename/title")
    chunk_index: int = Field(..., ge=0, description="Chunk sequence index")
    content: str = Field(..., description="Extracted text chunk content")
    character_count: int = Field(..., ge=1, description="Character count")
    token_count: int = Field(..., ge=0, description="Subword token count")
    start_offset: int = Field(..., ge=0, description="Start character offset")
    end_offset: int = Field(..., ge=0, description="End character offset")
    language: str = Field(..., description="Detected ISO language code")
    script: str = Field(..., description="Detected Unicode script")
    embedding: List[float] = Field(..., description="1536d normalized vector embedding")
    embedding_dimension: int = Field(1536, description="Vector dimension")
    embedding_provider: str = Field(..., description="Embedding provider name")
    embedding_model: str = Field(..., description="Embedding model name")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Preserved contextual metadata")


class EnrichDocumentRequest(BaseModel):
    """
    Request schema for end-to-end hybrid document enrichment.
    """
    text: str = Field(..., min_length=1, description="Document raw text to chunk and embed", json_schema_extra={"example": "ನಮಸ್ಕಾರ ಹೇಗಿದ್ದೀರಾ?"})
    document_name: str = Field("document.txt", description="Document filename/title")
    document_id: Optional[str] = Field(None, description="Optional document UUID")
    strategy: str = Field("recursive", description="Chunking strategy: 'character', 'recursive', or 'token'")
    chunk_size: int = Field(1000, gt=0, description="Target chunk size")
    chunk_overlap: int = Field(100, ge=0, description="Chunk overlap size")
    embedding_provider: Optional[str] = Field("development_deterministic", description="Embedding provider")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Document metadata")


class EnrichDocumentResponse(BaseModel):
    """
    Response schema containing enriched document chunks with vectors and metadata.
    """
    document_id: str = Field(..., description="Document identifier")
    document_name: str = Field(..., description="Document title")
    strategy: str = Field(..., description="Applied chunking strategy")
    total_chunks: int = Field(..., ge=0, description="Total enriched chunks produced")
    embedding_dimension: int = Field(1536, description="Vector dimension")
    embedding_provider: str = Field(..., description="Active embedding provider name")
    chunks: List[EnrichedChunkSchema] = Field(default_factory=list, description="Ordered enriched chunks with vectors")
    processing_time_ms: float = Field(0.0, ge=0.0, description="Processing duration in milliseconds")
