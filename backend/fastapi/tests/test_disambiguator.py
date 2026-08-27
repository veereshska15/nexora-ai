import pytest
from nlp.disambiguation.language_disambiguator import (
    LanguageDisambiguator,
    language_disambiguator,
)
from nlp.disambiguation.ngram_model import NGramModel
from nlp.disambiguation.romanized_detector import RomanizedLanguageDetector
from nlp.disambiguation.models.disambiguation_result import DisambiguationResult

# ==============================================================================
# 1. DIRECT SCRIPT MAPPING TESTS ACROSS INDIC & ENGLISH
# ==============================================================================

def test_kannada_unicode():
    res = language_disambiguator.disambiguate("ಕನ್ನಡ ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆ ಮತ್ತು ನರಮಂಡಲ ಜಾಲ")
    assert res.language == "kn"
    assert res.language_name == "Kannada"
    assert res.script == "Kannada"
    assert res.confidence >= 0.90
    assert res.candidates == ["kn"]
    assert res.romanized is False
    assert "deterministic_unicode_script_match" in res.evidence

def test_hindi_unicode():
    res = language_disambiguator.disambiguate("हिन्दी में आपका स्वागत है। हम सब खुश हैं।")
    assert res.language == "hi"
    assert res.script == "Devanagari"
    assert "hi" in res.candidates
    assert "mr" in res.candidates
    assert res.romanized is False

def test_marathi_unicode():
    # Contains distinct Marathi markers (आहे, आणि, मध्ये, ळ)
    res = language_disambiguator.disambiguate("मराठी भाषा अतिशय सुंदर आहे आणि यामध्ये बरेच काही शिकण्यासारखे आहे.")
    assert res.language == "mr"
    assert res.script == "Devanagari"
    assert "mr" in res.candidates
    assert "hi" in res.candidates

def test_tamil_unicode():
    res = language_disambiguator.disambiguate("தமிழ் இயற்கை மொழி செயலாக்கம்")
    assert res.language == "ta"
    assert res.script == "Tamil"
    assert res.confidence >= 0.90

def test_telugu_unicode():
    res = language_disambiguator.disambiguate("తెలుగు సహజ భాషా ప్రాసెసింగ్")
    assert res.language == "te"
    assert res.script == "Telugu"
    assert res.confidence >= 0.90

def test_malayalam_unicode():
    res = language_disambiguator.disambiguate("മലയാളം പ്രകൃതിഭാഷാ സംസ്കരണം")
    assert res.language == "ml"
    assert res.script == "Malayalam"
    assert res.confidence >= 0.90

def test_bengali_unicode():
    res = language_disambiguator.disambiguate("বাংলা প্রাকৃতিক ভাষা প্রক্রিয়াকরণ")
    assert res.language == "bn"
    assert res.script == "Bengali"
    assert res.confidence >= 0.90

def test_english_latin():
    res = language_disambiguator.disambiguate("NEXORA AI is a cutting-edge multimodal artificial intelligence platform.")
    assert res.language == "en"
    assert res.script == "Latin"
    assert res.romanized is False
    assert res.confidence >= 0.80

# ==============================================================================
# 2. DEVANAGARI AMBIGUITY & STATISTICAL CANDIDATES
# ==============================================================================

def test_hindi_vs_marathi_ambiguous_short_text():
    # Very short single word without distinctive Hindi/Marathi inflection
    res = language_disambiguator.disambiguate("भारत")
    assert res.script == "Devanagari"
    assert "hi" in res.candidates
    assert "mr" in res.candidates
    # Should safely flag ambiguity
    assert res.ambiguous is True

# ==============================================================================
# 3. ROMANIZED INDIC VS STANDARD ENGLISH
# ==============================================================================

def test_romanized_kannada_heuristic():
    res = language_disambiguator.disambiguate("namaskara nanna hesaru Veeresh, hegiddira neevu?")
    assert res.language == "kn"
    assert res.script == "Latin"
    assert res.romanized is True
    assert res.confidence >= 0.60
    assert any("romanized_kannada" in ev for ev in res.evidence)

def test_standard_english_not_misclassified():
    res = language_disambiguator.disambiguate("How are you doing today? Welcome to our new intelligence system.")
    assert res.language == "en"
    assert res.script == "Latin"
    assert res.romanized is False

# ==============================================================================
# 4. MIXED-SCRIPT & CODE-SWITCHING HANDLING
# ==============================================================================

def test_mixed_kannada_and_english():
    res = language_disambiguator.disambiguate("ನಮಸ್ಕಾರ Hello world! Welcome to Bengaluru.")
    assert res.mixed_language is True
    assert res.confidence > 0.0

def test_mixed_latin_and_indic():
    res = language_disambiguator.disambiguate("Hello ಕನ್ನಡ ಮತ್ತು Hindi हिन्दी AI platform")
    assert res.mixed_language is True

# ==============================================================================
# 5. EDGE CASES & DEFENSIVE BOUNDS
# ==============================================================================

def test_empty_and_null_text():
    res1 = language_disambiguator.disambiguate("")
    assert res1.language == "unknown"
    assert res1.confidence == 0.0
    assert res1.ambiguous is True

    res2 = language_disambiguator.disambiguate(None)
    assert res2.language == "unknown"
    assert res2.confidence == 0.0
    assert res2.ambiguous is True

def test_unknown_symbolic_text():
    res = language_disambiguator.disambiguate("!@#$%^&*() 1234567890")
    assert res.language == "unknown"
    assert res.confidence == 0.0
    assert res.ambiguous is True

def test_low_confidence_handling():
    # Ambiguous generic single Latin word with no strong language signal
    res = language_disambiguator.disambiguate("xyz abc")
    assert res.script == "Latin"
    assert res.confidence <= 0.60

def test_ambiguous_handling():
    res = language_disambiguator.disambiguate("xyz")
    assert res.ambiguous is True

def test_confidence_range():
    samples = [
        "ಕನ್ನಡ",
        "हिन्दी भाषा है",
        "This is an English sentence.",
        "namaskara",
        "12345",
    ]
    for s in samples:
        res = language_disambiguator.disambiguate(s)
        assert 0.0 <= res.confidence <= 1.0

def test_evidence_generation():
    res = language_disambiguator.disambiguate("ಕನ್ನಡ ಸಾಹಿತ್ಯ")
    assert isinstance(res.evidence, list)
    assert len(res.evidence) > 0
    assert all(isinstance(e, str) for e in res.evidence)

def test_long_input_protection():
    huge_text = "ಕನ್ನಡ " * 20000  # > 100,000 chars
    with pytest.raises(ValueError) as exc:
        language_disambiguator.disambiguate(huge_text)
    assert "exceeds maximum allowed length" in str(exc.value)

def test_deterministic_results():
    text = "ನಮಸ್ಕಾರ ಮತ್ತು ಶುಭೋದಯ"
    res1 = language_disambiguator.disambiguate(text)
    res2 = language_disambiguator.disambiguate(text)
    assert res1.language == res2.language
    assert res1.confidence == res2.confidence
    assert res1.evidence == res2.evidence

def test_custom_dependency_injection():
    custom_ngram = NGramModel(n_range=(1, 2))
    custom_rom = RomanizedLanguageDetector()
    custom_disambiguator = LanguageDisambiguator(
        ngram_model=custom_ngram,
        romanized_detector=custom_rom,
    )
    res = custom_disambiguator.disambiguate("ಕನ್ನಡ")
    assert res.language == "kn"
    assert res.script == "Kannada"
