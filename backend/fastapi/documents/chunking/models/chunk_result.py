from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """
    Standardized atomic text chunk with positional offsets, token telemetry, and metadata.
    """
    chunk_id: str = Field(..., description="Unique deterministic identifier for the chunk")
    document_id: str = Field(..., description="Parent document identifier")
    document_name: str = Field(..., description="Parent document filename/title")
    chunk_index: int = Field(..., ge=0, description="Zero-based ordinal index of the chunk")
    content: str = Field(..., min_length=1, description="Text content of the chunk")
    character_count: int = Field(..., ge=1, description="Number of characters in the chunk")
    token_count: int = Field(..., ge=0, description="Token count calculated via multilingual tokenizer")
    start_offset: int = Field(..., ge=0, description="Character start position in original text")
    end_offset: int = Field(..., ge=0, description="Character end position in original text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Preserved contextual metadata")


class ChunkingResult(BaseModel):
    """
    Container for the complete chunking operation output.
    """
    document_id: str = Field(..., description="Document ID")
    document_name: str = Field(..., description="Document name")
    strategy: str = Field(..., description="Applied chunking strategy (character, recursive, token)")
    chunk_size: int = Field(..., description="Configured chunk size parameter")
    chunk_overlap: int = Field(..., description="Configured chunk overlap parameter")
    total_chunks: int = Field(..., ge=0, description="Total number of chunks produced")
    chunks: List[Chunk] = Field(default_factory=list, description="List of generated chunks")
    processing_time_ms: float = Field(0.0, ge=0.0, description="Chunking latency in milliseconds")
