from abc import ABC, abstractmethod
from nlp.disambiguation.models.disambiguation_result import DisambiguationResult

MAX_TEXT_INPUT_LIMIT = 100_000


class BaseDisambiguator(ABC):
    """
    Abstract Base Class for language disambiguators in NEXORA AI.
    Defines the contract for combining script analysis, statistical n-grams,
    and Romanized Indic heuristics.
    """

    def __init__(self, max_length: int = MAX_TEXT_INPUT_LIMIT):
        self.max_length = max_length

    def validate_input(self, text: str) -> None:
        """Defensively checks text length against buffer limits."""
        if text and len(text) > self.max_length:
            raise ValueError(
                f"Input text exceeds maximum allowed length of {self.max_length} characters (received {len(text)})"
            )

    @abstractmethod
    def disambiguate(self, text: str) -> DisambiguationResult:
        """Analyzes text and returns a comprehensive DisambiguationResult."""
        pass
