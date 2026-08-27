import unicodedata
from typing import Dict, List, Tuple
from nlp.models.language_metadata import ScriptDetectionResult

# Script Unicode code point ranges (start, end)
SCRIPT_RANGES: Dict[str, List[Tuple[int, int]]] = {
    "Kannada": [(0x0C80, 0x0CFF)],
    "Devanagari": [(0x0900, 0x097F)],
    "Tamil": [(0x0B80, 0x0BFF)],
    "Telugu": [(0x0C00, 0x0C7F)],
    "Malayalam": [(0x0D00, 0x0D7F)],
    "Bengali": [(0x0980, 0x09FF)],
    "Latin": [
        (0x0041, 0x005A),  # Basic Latin uppercase A-Z
        (0x0061, 0x007A),  # Basic Latin lowercase a-z
        (0x00C0, 0x00FF),  # Latin-1 Supplement letters
        (0x0100, 0x017F),  # Latin Extended-A
        (0x0180, 0x024F),  # Latin Extended-B
    ],
}

INDIC_SCRIPTS = {"Kannada", "Devanagari", "Tamil", "Telugu", "Malayalam", "Bengali"}

SCRIPT_TO_CANDIDATES: Dict[str, List[str]] = {
    "Kannada": ["kn"],
    "Devanagari": ["hi", "mr"],
    "Tamil": ["ta"],
    "Telugu": ["te"],
    "Malayalam": ["ml"],
    "Bengali": ["bn"],
    "Latin": ["en"],
    "Unknown": [],
}

MAX_ANALYSIS_CHAR_LIMIT = 100_000

class UnicodeScriptIdentifier:
    """
    Deterministic, high-performance Unicode Script Identifier for Indic and Latin scripts.
    Executes code-point range categorization in O(N) time with sub-millisecond latency.
    """

    def __init__(self, max_char_limit: int = MAX_ANALYSIS_CHAR_LIMIT):
        self.max_char_limit = max_char_limit

    def _get_char_script(self, char: str) -> str | None:
        """Determines the script of a single character by code-point range."""
        cp = ord(char)

        # Ignore ASCII whitespace, numbers, and basic punctuation
        if char.isspace() or (0x0020 <= cp <= 0x0040) or (0x005B <= cp <= 0x0060) or (0x007B <= cp <= 0x007E):
            return None

        # Ignore zero-width characters (ZWNJ, ZWJ, BOM) and Unicode hyphens/punctuation
        if cp in (0x200C, 0x200D, 0xFEFF) or (0x2010 <= cp <= 0x2027) or (0x2030 <= cp <= 0x205E):
            return None

        # Check against defined script ranges
        for script_name, ranges in SCRIPT_RANGES.items():
            for start, end in ranges:
                if start <= cp <= end:
                    return script_name

        # Ignore general symbols, punctuation, and numeric categories
        cat = unicodedata.category(char)
        if cat.startswith("P") or cat.startswith("S") or cat.startswith("N") or cat.startswith("C"):
            return None

        return "Unknown"

    def identify_script(self, text: str) -> ScriptDetectionResult:
        """
        Analyzes the input text and returns structured script identification metadata.
        Handles mixed-script text, ambiguous multi-language scripts (Devanagari),
        and enforces defensive bounds on input length.
        """
        if not text or not isinstance(text, str):
            return ScriptDetectionResult(
                primary_script="Unknown",
                language_candidates=[],
                confidence=0.0,
                is_indic=False,
                scripts=[],
                script_distribution={},
                mixed=False,
                total_characters=0,
            )

        if len(text) > self.max_char_limit:
            raise ValueError(
                f"Input text exceeds maximum allowed length of {self.max_char_limit} characters (received {len(text)})"
            )

        script_counts: Dict[str, int] = {}
        total_counted = 0

        for char in text:
            script = self._get_char_script(char)
            if script:
                script_counts[script] = script_counts.get(script, 0) + 1
                total_counted += 1

        if total_counted == 0:
            return ScriptDetectionResult(
                primary_script="Unknown",
                language_candidates=[],
                confidence=0.0,
                is_indic=False,
                scripts=[],
                script_distribution={},
                mixed=False,
                total_characters=0,
            )

        # Sort scripts by character frequency descending
        sorted_scripts = sorted(script_counts.items(), key=lambda item: item[1], reverse=True)
        primary_script, primary_count = sorted_scripts[0]

        distribution = {
            script: round(count / total_counted, 4) for script, count in sorted_scripts
        }
        confidence = distribution[primary_script]
        distinct_scripts = [script for script, _ in sorted_scripts if script != "Unknown"]
        if not distinct_scripts:
            distinct_scripts = ["Unknown"]

        is_mixed = len(distinct_scripts) > 1
        candidates = SCRIPT_TO_CANDIDATES.get(primary_script, [])

        return ScriptDetectionResult(
            primary_script=primary_script,
            language_candidates=candidates,
            confidence=confidence,
            is_indic=primary_script in INDIC_SCRIPTS,
            scripts=distinct_scripts,
            script_distribution=distribution,
            mixed=is_mixed,
            total_characters=total_counted,
        )

    # Alias for convenience
    identify = identify_script

unicode_script_identifier = UnicodeScriptIdentifier()
