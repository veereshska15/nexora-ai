import time
import unicodedata
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from nlp.tokenizers.tokenizer_factory import tokenizer_factory
from documents.chunking.models.chunk_result import Chunk

MAX_CHUNK_SIZE_LIMIT = 50000

# Unicode Combining Mark Categories: Mn (Nonspacing Mark), Mc (Spacing Combining Mark), Me (Enclosing Mark)
COMBINING_CATEGORIES = {"Mn", "Mc", "Me"}


class BaseChunker(ABC):
    """
    Abstract Base Class for text chunking strategies in NEXORA AI.
    Guarantees Unicode Indic conjunct preservation, parameter validation,
    and multilingual token count telemetry.
    """

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name

    def validate_params(self, text: str, chunk_size: int, chunk_overlap: int) -> None:
        """Validates chunking inputs according to strict safety invariants."""
        if text is None or not text.strip():
            raise ValueError("Input text cannot be empty or whitespace-only.")

        if chunk_size <= 0:
            raise ValueError("chunk_size must be strictly greater than 0.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative (>= 0).")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size.")

        if chunk_size > MAX_CHUNK_SIZE_LIMIT:
            raise ValueError(f"chunk_size {chunk_size} exceeds maximum allowed limit of {MAX_CHUNK_SIZE_LIMIT}.")

    def calculate_tokens(self, text: str, language_code: str = "en") -> int:
        """Computes subword token count using language-aware tokenizer."""
        if not text:
            return 0
        try:
            tok = tokenizer_factory.get(language_code)
            res = tok.tokenize(text)
            return res.token_count
        except Exception:
            # Fallback estimation
            return max(1, len(text.split()))

    def adjust_for_unicode_graphemes(self, text: str, index: int) -> int:
        """
        Prevents splitting within an Indic grapheme cluster or Ottakshara conjunct.
        If index lands on a combining diacritic / virama, advances to grapheme boundary.
        """
        if index <= 0 or index >= len(text):
            return index

        # If current character is a combining mark (vowel sign, virama, anusvara), shift forward
        while index < len(text) and unicodedata.category(text[index]) in COMBINING_CATEGORIES:
            index += 1

        # Check if previous character was a virama (halant / ottakshara connector)
        # Kannada virama: \u0CCD, Devanagari: \u094D, Telugu: \u0C4D, Tamil: \u0BCD, Malayalam: \u0D4D, Bengali: \u09CD
        viramas = {'\u0ccd', '\u094d', '\u0c4d', '\u0bcd', '\u0d4d', '\u09cd', '\u200c', '\u200d'}
        if index > 0 and text[index - 1] in viramas and index < len(text):
            # Advance past the conjunct consonant following the virama
            index += 1
            while index < len(text) and unicodedata.category(text[index]) in COMBINING_CATEGORIES:
                index += 1

        return min(index, len(text))

    @abstractmethod
    def split(
        self,
        text: str,
        document_id: str,
        document_name: str,
        chunk_size: int,
        chunk_overlap: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Subclasses implement specific splitting logic."""
        pass
