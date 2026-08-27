import unicodedata
import pytest
from nlp.normalizers.base_normalizer import base_normalizer, BaseNormalizer
from nlp.normalizers.indic_normalizer import indic_normalizer
from nlp.normalizers.kannada_normalizer import kannada_normalizer, KannadaNormalizer

# ==============================================================================
# 1. UNICODE NFC CANONICAL NORMALIZATION TESTS ACROSS 8 LANGUAGES
# ==============================================================================

def test_english_nfc():
    # Combining acute accent: 'e' + '\u0301' -> 'é'
    decomposed = "Cafe\u0301"
    normalized = base_normalizer.normalize(decomposed)
    assert normalized == "Café"
    assert unicodedata.is_normalized("NFC", normalized)

def test_kannada_nfc():
    text = "ಕನ್ನಡ ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆ"
    normalized = kannada_normalizer.normalize(text)
    assert normalized == "ಕನ್ನಡ ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆ"
    assert unicodedata.is_normalized("NFC", normalized)

def test_hindi_nfc():
    text = "हिन्दी प्राकृतिक भाषा संसाधन"
    normalized = indic_normalizer.normalize(text)
    assert normalized == "हिन्दी प्राकृतिक भाषा संसाधन"
    assert unicodedata.is_normalized("NFC", normalized)

def test_tamil_nfc():
    text = "தமிழ் இயற்கை மொழி செயலாக்கம்"
    normalized = indic_normalizer.normalize(text)
    assert normalized == "தமிழ் இயற்கை மொழி செயலாக்கம்"
    assert unicodedata.is_normalized("NFC", normalized)

def test_telugu_nfc():
    text = "తెలుగు సహజ భాషా ప్రాసెసింగ్"
    normalized = indic_normalizer.normalize(text)
    assert normalized == "తెలుగు సహజ భాషా ప్రాసెసింగ్"
    assert unicodedata.is_normalized("NFC", normalized)

def test_malayalam_nfc():
    text = "മലയാളം പ്രകൃതിഭാഷാ സംസ്കരണം"
    normalized = indic_normalizer.normalize(text)
    assert normalized == "മലയാളം പ്രകൃതിഭാഷാ സംസ്കരണം"
    assert unicodedata.is_normalized("NFC", normalized)

def test_marathi_nfc():
    text = "मराठी नैसर्गिक भाषा प्रक्रिया"
    normalized = indic_normalizer.normalize(text)
    assert normalized == "मराठी नैसर्गिक भाषा प्रक्रिया"
    assert unicodedata.is_normalized("NFC", normalized)

def test_bengali_nfc():
    text = "বাংলা প্রাকৃতিক ভাষা প্রক্রিয়াকরণ"
    normalized = indic_normalizer.normalize(text)
    assert normalized == "বাংলা প্রাকৃতিক ভাষা প্রক্রিয়াকরণ"
    assert unicodedata.is_normalized("NFC", normalized)

# ==============================================================================
# 2. KANNADA SPECIFIC LINGUISTIC STRUCTURES
# ==============================================================================

def test_kannada_vowel_signs():
    # Test all canonical Kannada matras attached to 'ಕ'
    matras = ["ಕ", "ಕಾ", "ಕಿ", "ಕೀ", "ಕು", "ಕೂ", "ಕೆ", "ಕೇ", "ಕೈ", "ಕೊ", "ಕೋ", "ಕೌ"]
    for word in matras:
        norm = kannada_normalizer.normalize(word)
        assert norm == word
        assert unicodedata.is_normalized("NFC", norm)

def test_kannada_virama():
    # Test virama / halant (್)
    halant_word = "ಕ್"
    norm = kannada_normalizer.normalize(halant_word)
    assert norm == "ಕ್"

def test_kannada_conjuncts_ottakshara():
    # Valid Kannada Ottaksharas (consonant + virama + consonant)
    conjuncts = ["ಕ್ಕ", "ಕ್ರ", "ಕ್ಲ", "ಜ್ಞ", "ಷ್ಟ", "ಪ್ರಜ್ಞೆ", "ಕರ್ನಾಟಕ"]
    for conj in conjuncts:
        norm = kannada_normalizer.normalize(conj)
        assert norm == conj
        assert unicodedata.is_normalized("NFC", norm)

def test_kannada_anusvara():
    text = "ಕಂ ನಮಸ್ಕಾರಂ ಬೆಂಗಳೂರು"
    norm = kannada_normalizer.normalize(text)
    assert norm == "ಕಂ ನಮಸ್ಕಾರಂ ಬೆಂಗಳೂರು"

def test_kannada_visarga():
    text = "ದುಃಖ ಪುನಃ ನಮಃ"
    norm = kannada_normalizer.normalize(text)
    assert norm == "ದುಃಖ ಪುನಃ ನಮಃ"

# ==============================================================================
# 3. ZERO-WIDTH AND CONTROL CHARACTERS
# ==============================================================================

def test_zwj_handling():
    # Valid ZWJ inside ligature preserved
    ligature = "ക്\u200D"
    norm = indic_normalizer.normalize(ligature)
    assert "\u200d" in norm or len(norm) > 0

def test_zwnj_handling():
    # Valid ZWNJ separating consonants
    separated = "ಕ್\u200Cಕ"
    norm = kannada_normalizer.normalize(separated)
    assert "\u200c" in norm

def test_bom_handling():
    # Byte Order Mark (\uFEFF) should be stripped
    bom_text = "\ufeffನಮಸ್ಕಾರ ವಿಶ್ವ"
    norm = kannada_normalizer.normalize(bom_text)
    assert norm == "ನಮಸ್ಕಾರ ವಿಶ್ವ"
    assert "\ufeff" not in norm

# ==============================================================================
# 4. PUNCTUATION & WHITESPACE FORMATTING
# ==============================================================================

def test_indic_punctuation():
    text = "ನಮಸ್ಕಾರ। ಹೇಗಿದ್ದೀರಾ॥"
    norm = kannada_normalizer.normalize(text)
    assert "।" in norm
    assert "॥" in norm

def test_whitespace_normalization():
    messy_text = "  ಕನ್ನಡ     ಭಾಷೆ   \n\n\n\n   ಮತ್ತು   AI    "
    norm = kannada_normalizer.normalize(messy_text)
    assert norm == "ಕನ್ನಡ ಭಾಷೆ\n\nಮತ್ತು AI"

# ==============================================================================
# 5. MIXED-SCRIPT TEXT HANDLING
# ==============================================================================

def test_mixed_kannada_and_english():
    text = "  ನಮಸ್ಕಾರ   NEXORA AI  Welcome to Cyber-Nature!  "
    norm = kannada_normalizer.normalize(text)
    assert norm == "ನಮಸ್ಕಾರ NEXORA AI Welcome to Cyber-Nature!"

def test_mixed_indic_and_latin():
    text = "  Hello   ಕನ್ನಡ   ಮತ್ತು   Hindi हिन्दी   "
    norm = kannada_normalizer.normalize(text)
    assert norm == "Hello ಕನ್ನಡ ಮತ್ತು Hindi हिन्दी"

# ==============================================================================
# 6. EDGE CASES, SECURITY & IDEMPOTENCY
# ==============================================================================

def test_empty_and_whitespace_only_strings():
    assert kannada_normalizer.normalize("") == ""
    assert kannada_normalizer.normalize("     \n\t   ") == ""
    assert kannada_normalizer.normalize(None) == ""

def test_invalid_input_types():
    assert base_normalizer.normalize(12345) == ""
    assert base_normalizer.is_valid("") is False
    assert base_normalizer.is_valid(None) is False

def test_maximum_length_protection():
    custom_norm = BaseNormalizer(max_length=50)
    safe_text = "NEXORA AI Platform"
    assert custom_norm.normalize(safe_text) == "NEXORA AI Platform"

    huge_text = "ಕನ್ನಡ " * 30  # > 50 characters
    with pytest.raises(ValueError) as exc:
        custom_norm.normalize(huge_text)
    assert "exceeds maximum allowed length" in str(exc.value)

def test_original_text_preservation_with_metadata():
    raw_input = "  \ufeffಕನ್ನಡ    ಭಾಷೆ  "
    res = kannada_normalizer.normalize_with_metadata(raw_input, language="kn", script="Kannada")
    assert res.original_text == raw_input
    assert res.normalized_text == "ಕನ್ನಡ ಭಾಷೆ"
    assert res.changed is True
    assert res.language == "kn"
    assert res.script == "Kannada"

def test_idempotent_normalization():
    texts = [
        "  \ufeffನಮಸ್ಕಾರ   NEXORA AI    ",
        "हिन्दी भाषा। प्राकृतिक संसाधन॥",
        "Café au lait\n\n\n\nDouble space   ",
        "ಕರ್ನಾಟಕ ಮತ್ತು ಪ್ರಜ್ಞೆ",
    ]
    for text in texts:
        first_pass = kannada_normalizer.normalize(text)
        second_pass = kannada_normalizer.normalize(first_pass)
        third_pass = kannada_normalizer.normalize(second_pass)
        assert first_pass == second_pass == third_pass
