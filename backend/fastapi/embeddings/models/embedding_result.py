from typing import List
from pydantic import BaseModel, Field


class EmbeddingResult(BaseModel):
    """
    Standardized result model for text embedding generation.
    """
    text: str = Field(..., description="Original input text embedded")
    vector: List[float] = Field(..., description="Normalized dense vector embedding")
    dimension: int = Field(..., ge=1, description="Embedding vector dimensionality")
    provider: str = Field(..., description="Active embedding provider name")
    model: str = Field(..., description="Active embedding model name")
    processing_time_ms: float = Field(0.0, ge=0.0, description="Embedding generation latency in milliseconds")
