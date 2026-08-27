import re
from typing import Dict, List, Optional, Set, Tuple

ROMANIZED_KANNADA_PATTERNS: Set[str] = {
    "namaskara", "namaskaram", "hegiddira", "hegidira", "hegidiya", "nanna",
    "hesaru", "dhanyavada", "dhanyavadagalu", "illa", "enu", "yaake", "yake",
    "hege", "beku", "kannada", "kannadadalli", "gothilla", "gottilla",
    "chennagiddira", "neevu", "naanu", "illi", "alli", "banni", "hogu",
    "maadi", "kelasa", "oota", "ayitha", "houdu",
}

ROMANIZED_HINDI_PATTERNS: Set[str] = {
    "namaste", "kaise", "kya", "nahi", "nahin", "mera", "meri", "naam",
    "shukriya", "dhanyawad", "aap", "tum", "bhai", "karo", "hoga", "hogi",
    "achha", "theek", "kripya", "kaha", "idhar", "udhar", "bahut",
}

ENGLISH_COMMON_PATTERNS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "and", "or", "how", "what", "where",
    "who", "when", "why", "this", "that", "with", "from", "have", "has", "had",
    "to", "for", "of", "in", "on", "at", "by", "as", "be", "it", "platform", "system",
    "artificial", "intelligence", "today", "good", "welcome", "please", "doing",
    "fine", "hello", "hi", "world", "ai", "multimodal", "fast", "language", "model",
}


class RomanizedLanguageDetector:
    """
    Heuristic and statistical detector for Romanized Indic languages in Latin script.
    Differentiates between standard English and transliterated Kannada or Hindi.
    """

    def __init__(self):
        self._word_pattern = re.compile(r"[a-zA-Z]+")

    def detect_romanized(self, text: str) -> Tuple[Optional[str], float, bool, List[str]]:
        """
        Analyzes Latin-script text for Romanized Indic signals.
        Returns:
            (predicted_lang_code, confidence, is_romanized, evidence_list)
        """
        if not text:
            return None, 0.0, False, ["empty_input"]

        words = [w.lower() for w in self._word_pattern.findall(text)]
        if not words:
            return None, 0.0, False, ["no_latin_words"]

        total_words = len(words)
        kn_matches = sum(1 for w in words if w in ROMANIZED_KANNADA_PATTERNS)
        hi_matches = sum(1 for w in words if w in ROMANIZED_HINDI_PATTERNS)
        en_matches = sum(1 for w in words if w in ENGLISH_COMMON_PATTERNS)

        evidence: List[str] = []

        # 1. Strong Romanized Kannada signal
        if kn_matches > 0 and kn_matches >= hi_matches and kn_matches >= en_matches:
            confidence = min(0.60 + (kn_matches / total_words) * 0.35, 0.95)
            evidence.append("romanized_kannada_vocabulary_match")
            evidence.append(f"matched_{kn_matches}_kannada_signals")
            return "kn", round(confidence, 2), True, evidence

        # 2. Strong Romanized Hindi signal
        if hi_matches > 0 and hi_matches >= kn_matches and hi_matches >= en_matches:
            confidence = min(0.60 + (hi_matches / total_words) * 0.35, 0.95)
            evidence.append("romanized_hindi_vocabulary_match")
            evidence.append(f"matched_{hi_matches}_hindi_signals")
            return "hi", round(confidence, 2), True, evidence

        # 3. Standard English signal
        if en_matches > 0:
            confidence = min(0.70 + (en_matches / total_words) * 0.25, 0.95)
            evidence.append("standard_english_vocabulary_match")
            return "en", round(confidence, 2), False, evidence

        # 4. Low/Neutral evidence on generic Latin text
        evidence.append("generic_latin_script_insufficient_lexical_evidence")
        return "en", 0.50, False, evidence
