import uuid
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

QDRANT_VECTOR_DIMENSION = 1536
DEFAULT_COLLECTION_NAME = "nexora_documents"

class QdrantPointPayload(BaseModel):
    user_id: str = Field(..., description="Owner user UUID for multi-tenant isolation")
    document_id: str | None = Field(None, description="Associated document ID")
    chunk_index: int | None = Field(0, ge=0, description="Sequential chunk index")
    content: str | None = Field(None, description="Text chunk content")
    source: str | None = Field("upload", description="Data source type (upload, web, api)")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class QdrantPointInput(BaseModel):
    id: str | None = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique point UUID")
    vector: list[float] = Field(..., description="1536-dimensional embedding vector")
    payload: QdrantPointPayload = Field(..., description="Point metadata payload")

    @field_validator("vector")
    @classmethod
    def validate_dimension(cls, v: list[float]) -> list[float]:
        if len(v) != QDRANT_VECTOR_DIMENSION:
            raise ValueError(
                f"Invalid vector dimension: expected {QDRANT_VECTOR_DIMENSION} dimensions, but received {len(v)}"
            )
        return v

class QdrantUpsertRequest(BaseModel):
    collection_name: str = Field(DEFAULT_COLLECTION_NAME, description="Target Qdrant collection")
    points: list[QdrantPointInput] = Field(..., min_length=1, max_length=100, description="Batch of points to upsert")

class QdrantUpsertResponse(BaseModel):
    upserted_count: int = Field(..., description="Number of points successfully upserted")
    collection_name: str = Field(..., description="Collection name")
    status: str = Field("completed", description="Upsert operation status")

class QdrantSearchRequest(BaseModel):
    collection_name: str = Field(DEFAULT_COLLECTION_NAME, description="Target Qdrant collection")
    query_vector: list[float] = Field(..., description="1536-dimensional query vector")
    top_k: int = Field(5, ge=1, le=50, description="Number of closest points to return (1-50)")
    user_id: str | None = Field(None, description="Scoped user UUID for multi-tenant isolation")
    document_id: str | None = Field(None, description="Optional document filter")

    @field_validator("query_vector")
    @classmethod
    def validate_dimension(cls, v: list[float]) -> list[float]:
        if len(v) != QDRANT_VECTOR_DIMENSION:
            raise ValueError(
                f"Invalid vector dimension: expected {QDRANT_VECTOR_DIMENSION} dimensions, but received {len(v)}"
            )
        return v

class QdrantSearchResult(BaseModel):
    id: str = Field(..., description="Point UUID")
    score: float = Field(..., description="Cosine similarity score (higher is more similar)")
    payload: dict = Field(default_factory=dict, description="Point metadata payload")

class QdrantSearchResponse(BaseModel):
    results: list[QdrantSearchResult] = Field(default_factory=list)
    total_results: int = Field(..., description="Count of points returned")
    collection_name: str = Field(..., description="Searched collection")
    dimensions: int = Field(QDRANT_VECTOR_DIMENSION, description="Query vector dimensionality")

class QdrantCollectionInfo(BaseModel):
    name: str = Field(..., description="Collection name")
    status: str = Field(..., description="Collection status (green, yellow, red, offline)")
    vectors_count: int = Field(0, description="Total vectors indexed")
    vector_size: int = Field(QDRANT_VECTOR_DIMENSION, description="Vector embedding dimension")
    distance: str = Field("Cosine", description="Vector distance metric")
