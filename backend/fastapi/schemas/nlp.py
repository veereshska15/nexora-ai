from typing import List, Optional
from pydantic import BaseModel, Field


class NLPAnalyzeRequest(BaseModel):
    """Request payload for multilingual NLP analysis."""
    text: str = Field(..., min_length=1, description="Input text to analyze", json_schema_extra={"example": "ನಮಸ್ಕಾರ, ಹೇಗಿದ್ದೀರಾ?"})


class NLPAnalyzeResponse(BaseModel):
    """Comprehensive response payload for multilingual NLP analysis."""
    original_text: str = Field(..., description="Original un-normalized user input text")
    normalized_text: str = Field(..., description="Canonical Unicode normalized text")
    language: str = Field(..., description="Predicted ISO 639-1 language code")
    language_candidates: List[str] = Field(default_factory=list, description="Ranked candidate language codes")
    script: str = Field(..., description="Primary detected Unicode script")
    scripts: List[str] = Field(default_factory=list, description="All detected scripts present in input")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0.0, 1.0]")
    mixed_language: bool = Field(False, description="Whether multiple scripts or languages are present")
    romanized: bool = Field(False, description="Whether input is Romanized Indic transliterated text")
    ambiguous: bool = Field(False, description="Whether linguistic evidence is insufficient or conflicting")
    tokens: List[str] = Field(default_factory=list, description="Token string units")
    token_ids: List[int] = Field(default_factory=list, description="Encoded integer token IDs")
    token_count: int = Field(0, description="Total number of extracted tokens")
    processing_time_ms: float = Field(0.0, description="Pipeline latency in milliseconds")


class NLPBatchAnalyzeRequest(BaseModel):
    """Batch request payload for multiple text inputs."""
    texts: List[str] = Field(..., min_length=1, description="List of texts to analyze")


class NLPBatchAnalyzeResponse(BaseModel):
    """Batch response payload."""
    results: List[NLPAnalyzeResponse] = Field(..., description="List of pipeline results")
    total_texts: int = Field(..., description="Total number of processed texts")
