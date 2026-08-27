from pydantic import BaseModel, Field

class LanguageMetadata(BaseModel):
    """Metadata specification for a supported natural language in NEXORA AI."""
    code: str = Field(..., description="ISO 639-1 language code (e.g. 'kn', 'hi', 'en')")
    name: str = Field(..., description="English display name of the language")
    native_name: str = Field(..., description="Autonym / native script display name")
    script: str = Field(..., description="Primary writing script (e.g. 'Kannada', 'Devanagari')")
    script_family: str = Field(..., description="Linguistic script family (e.g. 'Dravidian', 'Indo-Aryan', 'Latin')")
    unicode_start: int = Field(..., description="Unicode code-point start range (hex)")
    unicode_end: int = Field(..., description="Unicode code-point end range (hex)")
    is_indic: bool = Field(..., description="Flag denoting Indic / South Asian language family")
    tokenizer_strategy: str = Field("sentencepiece", description="Recommended subword tokenization strategy")
    enabled: bool = Field(True, description="Whether language is active in inference pipelines")

class ScriptDetectionResult(BaseModel):
    """Structured response model for deterministic Unicode script identification."""
    primary_script: str = Field(..., description="Dominant detected script in input text")
    language_candidates: list[str] = Field(default_factory=list, description="ISO language codes matching detected script")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Normalized script dominance score (0.0 to 1.0)")
    is_indic: bool = Field(..., description="True if primary script belongs to Indic family")
    scripts: list[str] = Field(default_factory=list, description="All distinct scripts detected in text")
    script_distribution: dict[str, float] = Field(default_factory=dict, description="Character count proportion per script")
    mixed: bool = Field(False, description="True if multiple writing scripts are present")
    total_characters: int = Field(..., ge=0, description="Total non-whitespace script characters analyzed")
