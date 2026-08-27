import re
import unicodedata
from pydantic import BaseModel, Field

MAX_TEXT_INPUT_LIMIT = 100_000

class NormalizationResult(BaseModel):
    """Encapsulates the normalization output alongside metadata and original text."""
    original_text: str = Field(..., description="Original un-normalized user input")
    normalized_text: str = Field(..., description="Canonical Unicode normalized text")
    changed: bool = Field(..., description="Whether normalization altered the input string")
    language: str | None = Field(None, description="Associated language code if provided")
    script: str | None = Field(None, description="Primary detected script if resolved")
    warnings: list[str] = Field(default_factory=list, description="Diagnostic normalization warnings")

class BaseNormalizer:
    """
    Base multilingual text normalizer for NEXORA AI.
    Enforces Unicode NFC normalization, whitespace unification, control character sanitation,
    and length bounds.
    """

    def __init__(self, max_length: int = MAX_TEXT_INPUT_LIMIT):
        self.max_length = max_length
        # Pre-compile regex for multiple whitespace collapse while preserving single newlines
        self._whitespace_pattern = re.compile(r"[^\S\r\n]+")
        self._multinewline_pattern = re.compile(r"\n{3,}")

    def validate_length(self, text: str) -> None:
        """Validates that input length does not exceed maximum defensive limit."""
        if len(text) > self.max_length:
            raise ValueError(
                f"Input text exceeds maximum allowed length of {self.max_length} characters (received {len(text)})"
            )

    def is_valid(self, text: str) -> bool:
        """Checks if input is a valid non-empty string within length bounds."""
        if not text or not isinstance(text, str):
            return False
        return len(text) <= self.max_length

    def clean_control_characters(self, text: str) -> str:
        """
        Removes dangerous ASCII control characters and Byte Order Marks (BOM \uFEFF),
        while strictly preserving standard line breaks (\n), tabs (\t), and Unicode text.
        """
        # Strip Byte Order Mark (BOM / ZWNBSP)
        cleaned = text.replace("\ufeff", "")

        # Filter out ASCII control characters 0x00-0x1F except \t (0x09), \n (0x0A), and \r (0x0D)
        result = []
        for char in cleaned:
            cp = ord(char)
            if cp < 0x20 and char not in ("\n", "\t", "\r"):
                continue
            # Also ignore delete character 0x7F
            if cp == 0x7F:
                continue
            result.append(char)
        return "".join(result)

    def normalize_whitespace(self, text: str) -> str:
        """Collapses duplicate horizontal spaces and excessive newlines."""
        # Replace Windows CRLF with standard LF
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse multiple horizontal whitespace (spaces, tabs) into a single space
        text = self._whitespace_pattern.sub(" ", text)
        # Collapse 3+ consecutive newlines into 2 (preserving paragraph breaks)
        text = self._multinewline_pattern.sub("\n\n", text)
        # Trim leading and trailing whitespace around each line
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(lines).strip()

    def normalize(self, text: str) -> str:
        """
        Executes full canonical Unicode normalization (NFC standard),
        sanitizes control characters, and standardizes whitespace.
        Guaranteed to be idempotent: normalize(normalize(x)) == normalize(x).
        """
        if not text or not isinstance(text, str):
            return ""

        self.validate_length(text)

        # 1. Remove unwanted control characters & BOM
        text = self.clean_control_characters(text)

        # 2. Canonical Unicode NFC Decomposition & Composition
        text = unicodedata.normalize("NFC", text)

        # 3. Standardize whitespace
        text = self.normalize_whitespace(text)

        # Final NFC pass to guarantee consistency
        return unicodedata.normalize("NFC", text)

    def normalize_with_metadata(
        self,
        text: str,
        language: str | None = None,
        script: str | None = None,
    ) -> NormalizationResult:
        """Executes normalization and returns rich metadata preserving original input."""
        normalized = self.normalize(text)
        return NormalizationResult(
            original_text=text or "",
            normalized_text=normalized,
            changed=(text != normalized),
            language=language,
            script=script,
            warnings=[],
        )

base_normalizer = BaseNormalizer()
