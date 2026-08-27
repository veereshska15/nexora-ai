from typing import Dict, List, Optional
from nlp.models.language_metadata import LanguageMetadata

class LanguageRegistry:
    """
    Centralized, thread-safe Language Registry for NEXORA AI.
    Manages metadata, script mappings, and tokenizer strategies for multilingual processing.
    """

    def __init__(self):
        self._registry: Dict[str, LanguageMetadata] = {}
        self._initialize_defaults()

    def _initialize_defaults(self) -> None:
        """Populates the registry with the 8 core supported languages."""
        initial_languages = [
            LanguageMetadata(
                code="en",
                name="English",
                native_name="English",
                script="Latin",
                script_family="Indo-European (Germanic)",
                unicode_start=0x0020,
                unicode_end=0x007F,
                is_indic=False,
                tokenizer_strategy="bpe",
                enabled=True,
            ),
            LanguageMetadata(
                code="kn",
                name="Kannada",
                native_name="ಕನ್ನಡ",
                script="Kannada",
                script_family="Dravidian (Southern)",
                unicode_start=0x0C80,
                unicode_end=0x0CFF,
                is_indic=True,
                tokenizer_strategy="sentencepiece",
                enabled=True,
            ),
            LanguageMetadata(
                code="hi",
                name="Hindi",
                native_name="हिन्दी",
                script="Devanagari",
                script_family="Indo-Aryan (Central)",
                unicode_start=0x0900,
                unicode_end=0x097F,
                is_indic=True,
                tokenizer_strategy="sentencepiece",
                enabled=True,
            ),
            LanguageMetadata(
                code="ta",
                name="Tamil",
                native_name="தமிழ்",
                script="Tamil",
                script_family="Dravidian (Southern)",
                unicode_start=0x0B80,
                unicode_end=0x0BFF,
                is_indic=True,
                tokenizer_strategy="sentencepiece",
                enabled=True,
            ),
            LanguageMetadata(
                code="te",
                name="Telugu",
                native_name="తెలుగు",
                script="Telugu",
                script_family="Dravidian (South-Central)",
                unicode_start=0x0C00,
                unicode_end=0x0C7F,
                is_indic=True,
                tokenizer_strategy="sentencepiece",
                enabled=True,
            ),
            LanguageMetadata(
                code="ml",
                name="Malayalam",
                native_name="മലയാളം",
                script="Malayalam",
                script_family="Dravidian (Southern)",
                unicode_start=0x0D00,
                unicode_end=0x0D7F,
                is_indic=True,
                tokenizer_strategy="sentencepiece",
                enabled=True,
            ),
            LanguageMetadata(
                code="mr",
                name="Marathi",
                native_name="मराठी",
                script="Devanagari",
                script_family="Indo-Aryan (Southern)",
                unicode_start=0x0900,
                unicode_end=0x097F,
                is_indic=True,
                tokenizer_strategy="sentencepiece",
                enabled=True,
            ),
            LanguageMetadata(
                code="bn",
                name="Bengali",
                native_name="বাংলা",
                script="Bengali",
                script_family="Indo-Aryan (Eastern)",
                unicode_start=0x0980,
                unicode_end=0x09FF,
                is_indic=True,
                tokenizer_strategy="sentencepiece",
                enabled=True,
            ),
        ]

        for lang in initial_languages:
            self.register(lang)

    def register(self, metadata: LanguageMetadata) -> None:
        """Registers or updates a language metadata entry in the registry."""
        self._registry[metadata.code.lower().strip()] = metadata

    def get(self, code: str) -> Optional[LanguageMetadata]:
        """Retrieves language metadata by ISO code (e.g. 'kn', 'hi')."""
        if not code:
            return None
        return self._registry.get(code.lower().strip())

    def is_supported(self, code: str) -> bool:
        """Checks if a language code is registered and enabled."""
        lang = self.get(code)
        return lang is not None and lang.enabled

    def all(self) -> List[LanguageMetadata]:
        """Returns all registered language definitions."""
        return list(self._registry.values())

    def enabled(self) -> List[LanguageMetadata]:
        """Returns only enabled language definitions."""
        return [lang for lang in self._registry.values() if lang.enabled]

    def get_by_script(self, script_name: str) -> List[LanguageMetadata]:
        """Returns all registered languages that utilize the specified script."""
        script_norm = script_name.lower().strip()
        return [lang for lang in self._registry.values() if lang.script.lower() == script_norm]

    def get_indic_languages(self) -> List[LanguageMetadata]:
        """Returns all registered languages belonging to the Indic family."""
        return [lang for lang in self._registry.values() if lang.is_indic and lang.enabled]

language_registry = LanguageRegistry()
