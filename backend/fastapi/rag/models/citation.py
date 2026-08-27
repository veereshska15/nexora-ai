from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """
    Source citation linking a response claim directly to a verified retrieved chunk.
    """
    citation_id: int = Field(..., ge=1, description="1-indexed citation number (e.g. 1 for [1])")
    marker: str = Field(..., description="Citation marker string in text (e.g. '[1]')")
    document_id: str = Field(..., description="Unique document identifier")
    document_name: str = Field(..., description="Document filename/title")
    chunk_id: str = Field(..., description="Retrieved chunk identifier")
    chunk_index: int = Field(..., ge=0, description="Index of chunk within document")
    content_snippet: str = Field(..., description="Relevant text excerpt from chunk")
    language: str = Field("en", description="Document language code")
    script: str = Field("Latin", description="Document script")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Reranker composite relevance score")
