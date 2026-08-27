import time
from typing import List, Optional
from pydantic import BaseModel, Field

from nlp.language_registry import language_registry
from nlp.detectors.script_identifier import unicode_script_identifier
from nlp.disambiguation.language_disambiguator import language_disambiguator
from nlp.normalizers.base_normalizer import base_normalizer
from nlp.normalizers.indic_normalizer import indic_normalizer
from nlp.normalizers.kannada_normalizer import kannada_normalizer
from nlp.tokenizers.tokenizer_factory import tokenizer_factory

MAX_TEXT_INPUT_LIMIT = 100_000


class PipelineResult(BaseModel):
    """
    Unified end-to-end result model for the NEXORA Multilingual NLP Pipeline.
    Combines script detection, disambiguation, language normalization,
    and subword tokenization with processing latency telemetry.
    """
    original_text: str = Field(..., description="Original un-normalized user input text")
    normalized_text: str = Field(..., description="Canonical Unicode normalized text")
    language: str = Field(..., description="Predicted ISO 639-1 language code (e.g., 'kn', 'hi', 'en')")
    language_candidates: List[str] = Field(default_factory=list, description="Ranked or considered candidate language codes")
    script: str = Field(..., description="Primary detected script")
    scripts: List[str] = Field(default_factory=list, description="All detected scripts present in input")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Derived classification confidence score [0.0, 1.0]")
    mixed_language: bool = Field(False, description="Whether multiple scripts/languages are present")
    romanized: bool = Field(False, description="Whether input is Romanized Indic transliterated text")
    ambiguous: bool = Field(False, description="Whether linguistic evidence is insufficient or conflicting")
    tokens: List[str] = Field(default_factory=list, description="Token string units")
    token_ids: List[int] = Field(default_factory=list, description="Encoded integer token IDs")
    token_count: int = Field(0, description="Total number of extracted tokens")
    processing_time_ms: float = Field(0.0, description="Total pipeline execution latency in milliseconds")


class MultilingualNLPPipeline:
    """
    Unified Multilingual NLP Pipeline for NEXORA AI.
    Integrates Language Registry, Unicode Script Detection, Statistical Disambiguation,
    Canonical Normalization, and Indic Subword Tokenization.
    """

    def __init__(self, max_length: int = MAX_TEXT_INPUT_LIMIT):
        self.max_length = max_length

    def validate_input(self, text: str) -> None:
        """Defensively validates text length against memory bounds."""
        if text and len(text) > self.max_length:
            raise ValueError(
                f"Input text exceeds maximum allowed pipeline limit of {self.max_length} characters (received {len(text)})"
            )

    def analyze(self, text: str) -> PipelineResult:
        """
        Executes complete 7-stage deterministic NLP analysis for a single input text:
        1. Input validation & timing start
        2. Unicode script identification
        3. Statistical language disambiguation & candidate ranking
        4. Language-specific canonical normalization
        5. Language-aware subword tokenization
        6. Telemetry & latency calculation
        7. Final PipelineResult construction
        """
        start_time = time.perf_counter()

        if not text or not isinstance(text, str):
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 3)
            return PipelineResult(
                original_text="" if text is None else text,
                normalized_text="",
                language="unknown",
                language_candidates=[],
                script="Unknown",
                scripts=[],
                confidence=0.0,
                mixed_language=False,
                romanized=False,
                ambiguous=True,
                tokens=[],
                token_ids=[],
                token_count=0,
                processing_time_ms=elapsed_ms,
            )

        self.validate_input(text)

        # Stage 1: Script Identification
        script_res = unicode_script_identifier.identify(text)

        # Stage 2: Language Disambiguation & Candidate Selection
        disambig_res = language_disambiguator.disambiguate(text)
        predicted_lang = disambig_res.language

        # Stage 3: Language-Specific Normalization
        if predicted_lang == "kn":
            normalizer = kannada_normalizer
        elif predicted_lang in ("hi", "mr", "ta", "te", "ml", "bn"):
            normalizer = indic_normalizer
        else:
            normalizer = base_normalizer

        normalized_text = normalizer.normalize(text)

        # Stage 4: Language-Aware Tokenization
        tokenizer = tokenizer_factory.get(predicted_lang)
        token_res = tokenizer.tokenize_with_result(normalized_text)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 3)

        return PipelineResult(
            original_text=text,
            normalized_text=normalized_text,
            language=predicted_lang,
            language_candidates=disambig_res.candidates,
            script=disambig_res.script,
            scripts=script_res.scripts,
            confidence=disambig_res.confidence,
            mixed_language=disambig_res.mixed_language or script_res.mixed,
            romanized=disambig_res.romanized,
            ambiguous=disambig_res.ambiguous,
            tokens=token_res.tokens,
            token_ids=token_res.token_ids,
            token_count=token_res.token_count,
            processing_time_ms=elapsed_ms,
        )

    def analyze_batch(self, texts: List[str]) -> List[PipelineResult]:
        """Processes a batch of texts sequentially through the unified pipeline."""
        if not texts or not isinstance(texts, list):
            return []
        return [self.analyze(t) for t in texts]


multilingual_pipeline = MultilingualNLPPipeline()
