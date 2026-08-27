import pytest
from nlp.language_registry import language_registry
from nlp.detectors.script_identifier import unicode_script_identifier, UnicodeScriptIdentifier
from nlp.models.language_metadata import LanguageMetadata

# ==============================================================================
# 1. SCRIPT IDENTIFIER TESTS
# ==============================================================================

def test_english_script_detection():
    text = "Artificial Intelligence and Neural Networks in NEXORA."
    res = unicode_script_identifier.identify_script(text)
    assert res.primary_script == "Latin"
    assert "en" in res.language_candidates
    assert res.is_indic is False
    assert res.confidence == 1.0
    assert res.mixed is False

def test_kannada_script_detection():
    text = "ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆ ಮತ್ತು ನರಮಂಡಲ ಜಾಲಗಳು"
    res = unicode_script_identifier.identify_script(text)
    assert res.primary_script == "Kannada"
    assert res.language_candidates == ["kn"]
    assert res.is_indic is True
    assert res.confidence == 1.0
    assert res.mixed is False

def test_hindi_devanagari_script_detection():
    text = "कृत्रिम बुद्धिमत्ता और न्यूरल नेटवर्क"
    res = unicode_script_identifier.identify_script(text)
    assert res.primary_script == "Devanagari"
    assert set(res.language_candidates) == {"hi", "mr"}
    assert res.is_indic is True
    assert res.confidence == 1.0

def test_tamil_script_detection():
    text = "செயற்கை நுண்ணறிவு மற்றும் நரம்பியல் நெட்வொர்க்குகள்"
    res = unicode_script_identifier.identify_script(text)
    assert res.primary_script == "Tamil"
    assert res.language_candidates == ["ta"]
    assert res.is_indic is True
    assert res.confidence == 1.0

def test_telugu_script_detection():
    text = "కృత్రిమ మేధస్సు మరియు న్యూరల్ నెట్‌వర్క్‌లు"
    res = unicode_script_identifier.identify_script(text)
    assert res.primary_script == "Telugu"
    assert res.language_candidates == ["te"]
    assert res.is_indic is True
    assert res.confidence == 1.0

def test_malayalam_script_detection():
    text = "കൃത്രിമ ബുദ്ധി intelligence"
    res = unicode_script_identifier.identify_script(text)
    assert "Malayalam" in res.scripts
    assert "ml" in res.language_candidates or res.primary_script == "Malayalam"
    assert res.is_indic is True

def test_marathi_devanagari_candidate_handling():
    text = "कृत्रिम बुद्धिमत्ता आणि संगणकीय प्रणाली"
    res = unicode_script_identifier.identify_script(text)
    assert res.primary_script == "Devanagari"
    # Devanagari should accurately present both Hindi and Marathi as candidate languages
    assert "mr" in res.language_candidates
    assert "hi" in res.language_candidates

def test_bengali_script_detection():
    text = "কৃত্রিম বুদ্ধিমত্তা এবং নিউরাল নেটওয়ার্ক"
    res = unicode_script_identifier.identify_script(text)
    assert res.primary_script == "Bengali"
    assert res.language_candidates == ["bn"]
    assert res.is_indic is True
    assert res.confidence == 1.0

def test_mixed_kannada_and_english_script_detection():
    text = "ನಮಸ್ಕಾರ NEXORA AI Welcome to Cyber-Nature"
    res = unicode_script_identifier.identify_script(text)
    assert res.mixed is True
    assert "Kannada" in res.scripts
    assert "Latin" in res.scripts
    assert len(res.scripts) == 2
    assert res.script_distribution["Kannada"] > 0.0
    assert res.script_distribution["Latin"] > 0.0

def test_empty_and_whitespace_input():
    res_empty = unicode_script_identifier.identify_script("")
    assert res_empty.primary_script == "Unknown"
    assert res_empty.confidence == 0.0
    assert res_empty.total_characters == 0

    res_spaces = unicode_script_identifier.identify_script("   \n\t  ")
    assert res_spaces.primary_script == "Unknown"
    assert res_spaces.confidence == 0.0

def test_unsupported_or_symbol_only_script():
    text = "😀 🚀 12345 !!! ??? @@@ ###"
    res = unicode_script_identifier.identify_script(text)
    assert res.primary_script == "Unknown"
    assert res.confidence == 0.0
    assert res.total_characters == 0

def test_very_long_input_protection():
    custom_identifier = UnicodeScriptIdentifier(max_char_limit=500)
    safe_text = "ಕನ್ನಡ " * 50
    assert custom_identifier.identify_script(safe_text).primary_script == "Kannada"

    huge_text = "ಕನ್ನಡ " * 200  # Exceeds 500 chars limit
    with pytest.raises(ValueError) as exc:
        custom_identifier.identify_script(huge_text)
    assert "exceeds maximum allowed length" in str(exc.value)

def test_romanized_kannada_identifies_as_latin():
    # As specified: Romanized Indic text should be identified as Latin with 'en' candidate for this phase
    text = "namaskara hegiddira nimage shubhadina"
    res = unicode_script_identifier.identify_script(text)
    assert res.primary_script == "Latin"
    assert res.is_indic is False
    assert "en" in res.language_candidates

# ==============================================================================
# 2. LANGUAGE REGISTRY TESTS
# ==============================================================================

def test_registry_lookup_by_code():
    kn = language_registry.get("kn")
    assert kn is not None
    assert kn.name == "Kannada"
    assert kn.native_name == "ಕನ್ನಡ"
    assert kn.script == "Kannada"
    assert kn.is_indic is True
    assert kn.unicode_start == 0x0C80
    assert kn.unicode_end == 0x0CFF

    hi = language_registry.get("HI")  # Case-insensitive
    assert hi is not None
    assert hi.name == "Hindi"
    assert hi.script == "Devanagari"

def test_registry_unsupported_language_code():
    assert language_registry.get("xyz") is None
    assert language_registry.is_supported("xyz") is False
    assert language_registry.get("") is None

def test_registry_enabled_and_indic_filtering():
    all_langs = language_registry.all()
    assert len(all_langs) >= 8

    indic_langs = language_registry.get_indic_languages()
    assert len(indic_langs) == 7  # kn, hi, ta, te, ml, mr, bn
    codes = [l.code for l in indic_langs]
    assert "kn" in codes
    assert "hi" in codes
    assert "en" not in codes

    devanagari_langs = language_registry.get_by_script("Devanagari")
    assert len(devanagari_langs) == 2
    dev_codes = [l.code for l in devanagari_langs]
    assert "hi" in dev_codes
    assert "mr" in dev_codes
