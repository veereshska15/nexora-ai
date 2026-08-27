import re
import unicodedata
from nlp.normalizers.base_normalizer import BaseNormalizer, MAX_TEXT_INPUT_LIMIT

# Punctuation allowed across Indic and Latin text
INDIC_PUNCTUATION = {"।", "॥", "॰"}

class IndicNormalizer(BaseNormalizer):
    """
    Multilingual Indic Text Normalizer.
    Provides standard canonical normalization for Dravidian and Indo-Aryan scripts
    (Kannada, Hindi, Marathi, Tamil, Telugu, Malayalam, Bengali).
    """

    def __init__(self, max_length: int = MAX_TEXT_INPUT_LIMIT):
        super().__init__(max_length=max_length)
        # Regex to strip isolated ZWJ/ZWNJ at word boundaries or surrounded by spaces
        self._isolated_zwnj_pattern = re.compile(r"(^|[\s])[\u200c\u200d]+|[\u200c\u200d]+([\s]|$)")
        # Regex to collapse multiple consecutive ZWJ/ZWNJ characters
        self._consecutive_zw_pattern = re.compile(r"[\u200c\u200d]{2,}")

    def clean_zero_width_characters(self, text: str) -> str:
        """
        Preserves linguistically mandatory Zero-Width Joiners (\u200D) and Zero-Width
        Non-Joiners (\u200C) when attached to Indic characters, while stripping
        isolated, trailing, or duplicate zero-width markers.
        """
        # Collapse multiple consecutive ZWJ/ZWNJ down to a single character
        text = self._consecutive_zw_pattern.sub("\u200c", text)
        # Remove isolated ZWJ/ZWNJ at string or word boundaries
        text = self._isolated_zwnj_pattern.sub(r"\1\2", text)
        return text

    def normalize_indic_punctuation(self, text: str) -> str:
        """Standardizes spacing around Indic danda (।) and double danda (॥)."""
        # Ensure space after Danda if immediately followed by text
        text = re.sub(r"([।॥])([^\s\d।॥])", r"\1 \2", text)
        return text

    def normalize(self, text: str) -> str:
        """
        Normalizes Indic text via Unicode NFC, preserves valid conjuncts and ZWJ/ZWNJ ligatures,
        cleans isolated zero-width artifacts, and formats punctuation.
        Idempotency: normalize(normalize(text)) == normalize(text).
        """
        if not text or not isinstance(text, str):
            return ""

        self.validate_length(text)

        # 1. Base control character & BOM cleaning
        text = self.clean_control_characters(text)

        # 2. Unicode NFC decomposition and canonical composition
        text = unicodedata.normalize("NFC", text)

        # 3. Clean zero-width artifacts while preserving valid ligatures
        text = self.clean_zero_width_characters(text)

        # 4. Standardize whitespace
        text = self.normalize_whitespace(text)

        # 5. Indic punctuation formatting
        text = self.normalize_indic_punctuation(text)

        # Final NFC canonical composition pass
        return unicodedata.normalize("NFC", text)

indic_normalizer = IndicNormalizer()
