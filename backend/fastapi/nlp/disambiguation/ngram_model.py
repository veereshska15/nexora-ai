import math
from typing import Dict, List, Set, Tuple

# Distinctive character n-grams and vocabulary markers for Devanagari languages
# NOTE: These represent development-grade statistical profiles.
DEV_LANGUAGE_PROFILES: Dict[str, Dict[str, int]] = {
    "hi": {
        # Hindi common words & distinctive character n-grams
        "है": 100, "हूँ": 80, "था": 80, "थी": 80, "में": 90, "के": 90, "की": 85,
        "और": 90, "का": 80, "से": 80, "को": 75, "पर": 75, "यह": 60, "वह": 60,
        "लिए": 60, "करना": 50, "किया": 50, "रहे": 50, "रहा": 50, "सकते": 50,
        "हैन": 40, "नही": 40, "नहीं": 60, "मेरा": 50, "आपका": 50, "भारत": 10,
        "स्वागत": 40, "खुश": 40, "सब": 30, "हम": 40, "भाषा": 10,
    },
    "mr": {
        # Marathi common words & distinctive character n-grams
        "ळ": 100, "आहे": 100, "नाही": 90, "च्या": 95, "मध्ये": 90, "आणि": 90,
        "करणे": 80, "केले": 80, "होते": 80, "होती": 75, "माझे": 75, "तुमचे": 75,
        "आहेत": 75, "म्हणून": 60, "पण": 60, "फार": 50, "महाराष्ट्र": 60,
        "झाले": 60, "करतात": 60, "त्यांना": 60, "हे": 50, "तर": 50,
        "सुंदर": 30, "शिकण्यासारखे": 50, "बरेच": 40, "काही": 40, "भाषा": 10,
        "ळा": 50, "ळी": 50, "ळू": 50, "ळे": 50, "यां": 40,
    },
}


class NGramModel:
    """
    Character N-Gram Statistical Language Profiler.
    Extracts character unigrams, bigrams, and trigrams to calculate relative
    likelihood scores between candidate languages sharing the same script.
    """

    def __init__(self, n_range: Tuple[int, int] = (1, 3)):
        self.min_n, self.max_n = n_range

    def extract_ngrams(self, text: str) -> List[str]:
        """Extracts character n-grams within the specified order range."""
        if not text:
            return []
        ngrams: List[str] = []
        cleaned = "".join(text.split())
        length = len(cleaned)
        for n in range(self.min_n, self.max_n + 1):
            for i in range(length - n + 1):
                ngrams.append(cleaned[i : i + n])
        return ngrams

    def score_candidates(
        self, text: str, candidates: List[str]
    ) -> Dict[str, float]:
        """
        Calculates normalized likelihood scores across candidate languages using
        accumulated profile signal weights and Laplace-smoothed softmax.
        """
        if not text or not candidates:
            return {c: 1.0 / max(len(candidates), 1) for c in candidates}

        if len(candidates) == 1:
            return {candidates[0]: 1.0}

        words = text.split()
        text_ngrams = self.extract_ngrams(text)

        raw_scores: Dict[str, float] = {c: 1.0 for c in candidates}

        for cand in candidates:
            profile = DEV_LANGUAGE_PROFILES.get(cand, {})
            # Word-level matches
            for w in words:
                if w in profile:
                    raw_scores[cand] += profile[w]

            # Subword & character n-gram matches (especially unique characters like 'ळ')
            for ng in text_ngrams:
                if ng in profile:
                    raw_scores[cand] += profile[ng] * 0.5

        total_raw = sum(raw_scores.values())
        if total_raw == 0:
            return {c: 1.0 / len(candidates) for c in candidates}

        # Normalize to probability distribution [0.0, 1.0]
        return {c: round(score / total_raw, 4) for c, score in raw_scores.items()}
