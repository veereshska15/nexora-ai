from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from nlp.tokenizers.models.tokenization_result import TokenizationResult

MAX_TEXT_INPUT_LIMIT = 100_000
MAX_TOKEN_COUNT = 100_000

# Standard Special Tokens for LLM Context Pipelines
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
BOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3

SPECIAL_TOKENS_MAP = {
    PAD_TOKEN: PAD_TOKEN_ID,
    UNK_TOKEN: UNK_TOKEN_ID,
    BOS_TOKEN: BOS_TOKEN_ID,
    EOS_TOKEN: EOS_TOKEN_ID,
}

ID_TO_SPECIAL_TOKENS = {v: k for k, v in SPECIAL_TOKENS_MAP.items()}


class BaseTokenizer(ABC):
    """
    Abstract Base Class defining the tokenizer interface for NEXORA AI.
    All language-specific and subword tokenizers (SentencePiece, BPE, WordPiece, Hugging Face)
    must implement this contract.
    """

    def __init__(self, name: str, language: str, vocabulary_size: int = 32000):
        self.name = name
        self.language = language
        self._vocabulary_size = vocabulary_size

    @property
    def vocabulary_size(self) -> int:
        """Returns the total vocabulary size supported by the tokenizer."""
        return self._vocabulary_size

    def validate_input(self, text: str) -> None:
        """Defensively validates text length against buffer limits."""
        if text and len(text) > MAX_TEXT_INPUT_LIMIT:
            raise ValueError(
                f"Input text exceeds maximum allowed tokenizer limit of {MAX_TEXT_INPUT_LIMIT} characters (received {len(text)})"
            )

    def validate_token_ids(self, token_ids: List[int]) -> None:
        """Defensively validates token ID list length and value integrity."""
        if not isinstance(token_ids, list):
            raise TypeError("Token IDs must be provided as a list of integers.")
        if len(token_ids) > MAX_TOKEN_COUNT:
            raise ValueError(
                f"Token count {len(token_ids)} exceeds maximum safe decoding limit of {MAX_TOKEN_COUNT}"
            )

    @abstractmethod
    def tokenize(self, text: str, add_special_tokens: bool = False) -> List[str]:
        """Splits normalized text into a sequence of token strings."""
        pass

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encodes text into a sequence of integer token IDs."""
        pass

    @abstractmethod
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decodes integer token IDs back into human-readable text."""
        pass

    @abstractmethod
    def tokenize_with_result(
        self, text: str, add_special_tokens: bool = False
    ) -> TokenizationResult:
        """Performs tokenization and returns a comprehensive TokenizationResult container."""
        pass
