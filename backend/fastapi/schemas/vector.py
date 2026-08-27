from uuid import UUID
from pydantic import BaseModel, Field, field_validator

VECTOR_DIMENSION = 1536

class VectorSearchRequest(BaseModel):
    query_embedding: list[float] = Field(..., description="1536-dimensional dense embedding vector")
    top_k: int = Field(5, ge=1, le=50, description="Number of closest document chunks to return (1-50)")
    document_name: str | None = Field(None, description="Optional filter by document title")

    @field_validator("query_embedding")
    @classmethod
    def validate_dimension(cls, v: list[float]) -> list[float]:
        if len(v) != VECTOR_DIMENSION:
            raise ValueError(
                f"Invalid vector dimension: expected {VECTOR_DIMENSION} dimensions, but received {len(v)}"
            )
        return v

class VectorChunkResult(BaseModel):
    id: str = Field(..., description="Document chunk UUID")
    document_name: str = Field(..., description="Source document filename")
    chunk_index: int = Field(..., description="Zero-indexed chunk position in document")
    content: str = Field(..., description="Extracted text chunk content")
    distance: float = Field(..., description="Cosine distance metric (0 = identical, 2 = opposite)")
    similarity: float = Field(..., description="Cosine similarity score (1 - distance, clamped 0 to 1)")

class VectorSearchResponse(BaseModel):
    results: list[VectorChunkResult] = Field(default_factory=list)
    total_results: int = Field(..., description="Count of chunks returned")
    query_dimensions: int = Field(VECTOR_DIMENSION, description="Query embedding dimensionality")

class VectorIngestRequest(BaseModel):
    user_id: str | None = Field(None, description="Owner user UUID (defaults to authenticated user)")
    document_name: str = Field(..., min_length=1, max_length=255, description="Document filename")
    chunk_index: int = Field(..., ge=0, description="Sequential chunk index")
    content: str = Field(..., min_length=1, description="Text chunk content")
    embedding: list[float] = Field(..., description="Pre-computed 1536-dim embedding vector")

    @field_validator("embedding")
    @classmethod
    def validate_dimension(cls, v: list[float]) -> list[float]:
        if len(v) != VECTOR_DIMENSION:
            raise ValueError(
                f"Invalid vector dimension: expected {VECTOR_DIMENSION} dimensions, but received {len(v)}"
            )
        return v

class VectorIngestResponse(BaseModel):
    id: str = Field(..., description="Created chunk UUID")
    message: str = Field("Document chunk and vector embedding ingested successfully")
