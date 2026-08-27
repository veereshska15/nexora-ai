from typing import Dict, List, Optional
from nlp.disambiguation.base_disambiguator import BaseDisambiguator, MAX_TEXT_INPUT_LIMIT
from nlp.disambiguation.models.disambiguation_result import DisambiguationResult
from nlp.disambiguation.ngram_model import NGramModel
from nlp.disambiguation.romanized_detector import RomanizedLanguageDetector
from nlp.detectors.script_identifier import unicode_script_identifier
from nlp.language_registry import language_registry

# Direct 1-to-1 mappings from distinct Unicode script to ISO language code
SCRIPT_TO_LANGUAGE_MAP: Dict[str, str] = {
    "Kannada": "kn",
    "Tamil": "ta",
    "Telugu": "te",
    "Malayalam": "ml",
    "Bengali": "bn",
}


class LanguageDisambiguator(BaseDisambiguator):
    """
    Primary Multilingual Language Disambiguation Engine for NEXORA AI.
    Combines deterministic Unicode script detection, character n-gram scoring,
    Devanagari Hindi/Marathi differentiation, and Romanized Indic heuristics.
    """

    def __init__(
        self,
        ngram_model: Optional[NGramModel] = None,
        romanized_detector: Optional[RomanizedLanguageDetector] = None,
        max_length: int = MAX_TEXT_INPUT_LIMIT,
    ):
        super().__init__(max_length=max_length)
        self.ngram_model = ngram_model or NGramModel()
        self.romanized_detector = romanized_detector or RomanizedLanguageDetector()

    def disambiguate(self, text: str) -> DisambiguationResult:
        """
        Executes end-to-end language disambiguation.
        Returns a structured, strongly typed DisambiguationResult.
        """
        if not text or not isinstance(text, str):
            return DisambiguationResult(
                language="unknown",
                language_name="Unknown",
                confidence=0.0,
                candidates=[],
                script="Unknown",
                evidence=["empty_or_null_input"],
                romanized=False,
                mixed_language=False,
                ambiguous=True,
            )

        self.validate_input(text)

        # 1. Inspect Unicode Scripts
        script_res = unicode_script_identifier.identify(text)
        primary_script = script_res.primary_script
        is_mixed = script_res.mixed
        evidence: List[str] = [f"primary_script_{primary_script.lower()}"]

        if is_mixed:
            evidence.append(f"mixed_scripts_{'_'.join(s.lower() for s in script_res.scripts)}")

        # 2. Case: Unknown / Symbol-only script
        if primary_script == "Unknown":
            return DisambiguationResult(
                language="unknown",
                language_name="Unknown",
                confidence=0.0,
                candidates=[],
                script="Unknown",
                evidence=evidence + ["insufficient_script_evidence"],
                romanized=False,
                mixed_language=is_mixed,
                ambiguous=True,
            )

        # 3. Case: Direct 1-to-1 Indic Scripts (Kannada, Tamil, Telugu, Malayalam, Bengali)
        if primary_script in SCRIPT_TO_LANGUAGE_MAP:
            lang_code = SCRIPT_TO_LANGUAGE_MAP[primary_script]
            meta = language_registry.get(lang_code)
            lang_name = meta.name if meta else primary_script
            evidence.append("deterministic_unicode_script_match")

            return DisambiguationResult(
                language=lang_code,
                language_name=lang_name,
                confidence=round(script_res.confidence, 2),
                candidates=[lang_code],
                script=primary_script,
                evidence=evidence,
                romanized=False,
                mixed_language=is_mixed,
                ambiguous=False,
            )

        # 4. Case: Devanagari Script (Hindi vs Marathi disambiguation)
        if primary_script == "Devanagari":
            candidates = ["hi", "mr"]
            scores = self.ngram_model.score_candidates(text, candidates)
            evidence.append("character_ngram_statistical_scoring")

            # Rank candidates by score
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            top_lang, top_score = ranked[0]
            second_lang, second_score = ranked[1]

            score_diff = top_score - second_score
            is_ambiguous = score_diff < 0.15 or len(text.strip()) <= 5

            if is_ambiguous:
                evidence.append("devanagari_insufficient_differentiation_ambiguous")

            meta = language_registry.get(top_lang)
            lang_name = meta.name if meta else top_lang

            return DisambiguationResult(
                language=top_lang,
                language_name=lang_name,
                confidence=round(top_score, 2),
                candidates=[top_lang, second_lang],
                script="Devanagari",
                evidence=evidence,
                romanized=False,
                mixed_language=is_mixed,
                ambiguous=is_ambiguous,
            )

        # 5. Case: Latin Script (English vs Romanized Indic)
        if primary_script == "Latin":
            pred_lang, conf, is_rom, rom_ev = self.romanized_detector.detect_romanized(text)
            evidence.extend(rom_ev)
            chosen_lang = pred_lang or "en"
            meta = language_registry.get(chosen_lang)
            lang_name = meta.name if meta else chosen_lang

            is_ambiguous = conf < 0.60

            return DisambiguationResult(
                language=chosen_lang,
                language_name=lang_name,
                confidence=conf,
                candidates=[chosen_lang],
                script="Latin",
                evidence=evidence,
                romanized=is_rom,
                mixed_language=is_mixed,
                ambiguous=is_ambiguous,
            )

        # Fallback default
        return DisambiguationResult(
            language="en",
            language_name="English",
            confidence=0.50,
            candidates=["en"],
            script=primary_script,
            evidence=evidence + ["fallback_resolution"],
            romanized=False,
            mixed_language=is_mixed,
            ambiguous=True,
        )


language_disambiguator = LanguageDisambiguator()
