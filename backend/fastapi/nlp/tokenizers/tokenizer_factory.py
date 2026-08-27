from typing import Dict, List, Optional
from nlp.tokenizers.base_tokenizer import BaseTokenizer
from nlp.tokenizers.subword_tokenizer import SubwordTokenizer

SUPPORTED_LANGUAGES = ["en", "kn", "hi", "ta", "te", "ml", "mr", "bn"]


class TokenizerFactory:
    """
    Language-Aware Tokenizer Factory for NEXORA AI.
    Instantiates and manages language-specific tokenizers with support for
    SentencePiece, BPE, WordPiece, and development fallback engines.
    """

    def __init__(self):
        self._tokenizers: Dict[str, BaseTokenizer] = {}
        self._initialize_default_tokenizers()

    def _initialize_default_tokenizers(self) -> None:
        """Initializes default development tokenizers for all supported Indic & Latin languages."""
        for lang in SUPPORTED_LANGUAGES:
            self._tokenizers[lang] = SubwordTokenizer(
                name=f"nexora-{lang}-subword",
                language=lang,
                vocabulary_size=32000,
            )

    def register_tokenizer(self, language_code: str, tokenizer: BaseTokenizer) -> None:
        """Registers or overrides a tokenizer instance for a specific language code."""
        if not language_code or not isinstance(tokenizer, BaseTokenizer):
            raise ValueError("Valid language code and BaseTokenizer instance required.")
        self._tokenizers[language_code.lower()] = tokenizer

    def get(self, language_code: str) -> BaseTokenizer:
        """
        Retrieves the registered tokenizer for the specified ISO language code.
        If the language code is unregistered, gracefully falls back to the general fallback tokenizer.
        """
        if not language_code or not isinstance(language_code, str):
            return self._tokenizers["en"]

        code = language_code.lower().strip()
        if code in self._tokenizers:
            return self._tokenizers[code]

        # Graceful fallback for unsupported or unknown languages
        if "fallback" not in self._tokenizers:
            self._tokenizers["fallback"] = SubwordTokenizer(
                name="nexora-general-fallback",
                language="fallback",
                vocabulary_size=32000,
            )
        return self._tokenizers["fallback"]

    def list_supported_languages(self) -> List[str]:
        """Returns the list of explicitly configured language codes."""
        return [lang for lang in self._tokenizers.keys() if lang != "fallback"]


tokenizer_factory = TokenizerFactory()
