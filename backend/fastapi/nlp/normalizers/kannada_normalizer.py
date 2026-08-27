import re
import unicodedata
from nlp.normalizers.indic_normalizer import IndicNormalizer, MAX_TEXT_INPUT_LIMIT

KANNADA_UNICODE_START = 0x0C80
KANNADA_UNICODE_END = 0x0CFF

# Kannada Virama / Halant
KANNADA_VIRAMA = "\u0ccd"  # ್

# Set of dependent vowel signs (Matras) in Kannada
KANNADA_MATRAS = {
    "\u0cbe",  # ಾ (aa)
    "\u0cbf",  # ಿ (i)
    "\u0cc0",  # ೀ (ii)
    "\u0cc1",  # ು (u)
    "\u0cc2",  # ೂ (uu)
    "\u0cc3",  # ೃ (r)
    "\u0cc4",  # ೄ (rr)
    "\u0cc6",  # ೆ (e)
    "\u0cc7",  # ೇ (ee)
    "\u0cc8",  # ೈ (ai)
    "\u0cca",  # ೊ (o)
    "\u0ccb",  # ೋ (oo)
    "\u0ccc",  # ೌ (au)
    "\u0cd5",  # ೕ (length mark)
    "\u0cd6",  # ೖ (ai length mark)
}

class KannadaNormalizer(IndicNormalizer):
    """
    Dedicated Kannada Script Normalizer for NEXORA AI.
    Preserves Kannada conjuncts (Ottakshara), vowel signs (Matras),
    Anusvara, Visarga, and mixed Kannada-English sentences.
    """

    def __init__(self, max_length: int = MAX_TEXT_INPUT_LIMIT):
        super().__init__(max_length=max_length)
        # Regex to detect duplicate consecutive Viramas or Matras
        self._duplicate_virama = re.compile(rf"{KANNADA_VIRAMA}{{2,}}")
        # Regex for orphaned leading combining marks at the very start of text
        self._leading_orphan_marks = re.compile(r"^[\u0c82\u0c83\u0cbe-\u0ccd]+")

    def is_kannada_char(self, char: str) -> bool:
        """Returns True if the character falls within the Kannada Unicode block."""
        return KANNADA_UNICODE_START <= ord(char) <= KANNADA_UNICODE_END

    def has_kannada(self, text: str) -> bool:
        """Returns True if input text contains at least one Kannada script character."""
        return any(self.is_kannada_char(c) for c in text)

    def validate_kannada_structure(self, text: str) -> list[str]:
        """
        Performs non-destructive diagnostics on Kannada text.
        Returns a list of warnings if malformed sequences or suspicious patterns are detected.
        """
        warnings = []
        if self._leading_orphan_marks.match(text):
            warnings.append("Text starts with an orphaned Kannada combining diacritic.")
        if self._duplicate_virama.search(text):
            warnings.append("Detected duplicate consecutive Kannada Virama (್) marks.")
        return warnings

    def clean_kannada_diacritics(self, text: str) -> str:
        """
        Cleans redundant repeated Viramas or Matra combinations without altering
        valid composite consonants or conjunct formations.
        """
        # Collapse duplicate viramas (e.g. ್್ -> ್)
        text = self._duplicate_virama.sub(KANNADA_VIRAMA, text)
        return text

    def normalize(self, text: str) -> str:
        """
        Normalizes Kannada text applying Unicode NFC canonical form,
        cleaning diacritics, preserving valid Ottaksharas (conjuncts), and
        formatting mixed Kannada + Latin strings.
        Idempotency: normalize(normalize(text)) == normalize(text).
        """
        if not text or not isinstance(text, str):
            return ""

        self.validate_length(text)

        # 1. Base Indic normalization (Control chars, BOM, NFC, ZWJ/ZWNJ, Whitespace)
        text = super().normalize(text)

        # 2. Kannada-specific diacritic cleaning
        text = self.clean_kannada_diacritics(text)

        # Final NFC canonical composition
        return unicodedata.normalize("NFC", text)

kannada_normalizer = KannadaNormalizer()
