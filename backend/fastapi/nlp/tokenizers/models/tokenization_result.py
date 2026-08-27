from pydantic import BaseModel, Field

class TokenizationResult(BaseModel):
    """
    Strongly typed container for multilingual tokenization outputs.
    Preserves original and normalized text, token sequences, deterministic IDs,
    and diagnostic counts for context window & token usage calculation.
    """
    original_text: str = Field(..., description="Original un-normalized user text")
    normalized_text: str = Field(..., description="Canonical Unicode normalized text passed to tokenizer")
    tokens: list[str] = Field(..., description="Extracted token string units")
    token_ids: list[int] = Field(..., description="Encoded integer token IDs")
    token_count: int = Field(..., description="Total count of tokens (including special tokens if requested)")
    language: str = Field(..., description="ISO 639-1 language code of tokenizer")
    tokenizer_type: str = Field("development_fallback", description="Tokenizer strategy / engine name")
    unknown_token_count: int = Field(0, description="Count of unknown <UNK> tokens encountered")
    special_tokens_count: int = Field(0, description="Count of special structural tokens (<BOS>, <EOS>, etc.)")
