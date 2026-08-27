from typing import List
from pydantic import BaseModel, Field


class DisambiguationResult(BaseModel):
    """
    Strongly typed container for language disambiguation results.
    Combines Unicode script evidence, character N-gram statistical scoring,
    and Romanized Indic heuristic analysis.
    """
    language: str = Field(..., description="Predicted ISO 639-1 language code (e.g., 'kn', 'hi', 'en')")
    language_name: str = Field(..., description="Human-readable language name (e.g., 'Kannada', 'Hindi')")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Derived statistical/heuristic confidence score [0.0, 1.0]")
    candidates: List[str] = Field(default_factory=list, description="Ranked or considered candidate language codes")
    script: str = Field(..., description="Primary detected Unicode script (e.g., 'Kannada', 'Devanagari', 'Latin')")
    evidence: List[str] = Field(default_factory=list, description="List of evidence signals used for classification")
    romanized: bool = Field(False, description="Whether the text is written in Romanized Latin script")
    mixed_language: bool = Field(False, description="Whether multiple scripts/languages were detected")
    ambiguous: bool = Field(False, description="Flag indicating insufficient or conflicting linguistic evidence")
