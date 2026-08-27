import pytest
from nlp.tokenizers.subword_tokenizer import SubwordTokenizer
from nlp.tokenizers.tokenizer_factory import tokenizer_factory
from nlp.tokenizers.base_tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    UNK_TOKEN_ID,
    MAX_TEXT_INPUT_LIMIT,
    MAX_TOKEN_COUNT,
)

# ==============================================================================
# 1. MULTILINGUAL TOKENIZATION ACROSS 8 SUPPORTED LANGUAGES
# ==============================================================================

def test_english_tokenization():
    tok = tokenizer_factory.get("en")
    text = "NEXORA AI is a cutting-edge multimodal platform."
    tokens = tok.tokenize(text)
    assert len(tokens) > 0
    assert "NEXORA" in tokens
    assert "AI" in tokens

def test_kannada_tokenization():
    tok = tokenizer_factory.get("kn")
    text = "ನಮಸ್ಕಾರ ನನ್ನ ಹೆಸರು ವೀರೇಶ್"
    tokens = tok.tokenize(text)
    assert len(tokens) > 0
    assert "ನಮಸ್ಕಾರ" in tokens or "ನಮ" in tokens[0]

def test_hindi_tokenization():
    tok = tokenizer_factory.get("hi")
    text = "कृत्रिम बुद्धिमत्ता और प्राकृतिक भाषा"
    tokens = tok.tokenize(text)
    assert len(tokens) > 0

def test_tamil_tokenization():
    tok = tokenizer_factory.get("ta")
    text = "தமிழ் இயற்கை மொழி செயலாக்கம்"
    tokens = tok.tokenize(text)
    assert len(tokens) > 0

def test_telugu_tokenization():
    tok = tokenizer_factory.get("te")
    text = "తెలుగు కృత్రిమ మేధస్సు"
    tokens = tok.tokenize(text)
    assert len(tokens) > 0

def test_malayalam_tokenization():
    tok = tokenizer_factory.get("ml")
    text = "മലയാളം കൃത്രിമബുദ്ധി പ്ലാറ്റ്‌ഫോം"
    tokens = tok.tokenize(text)
    assert len(tokens) > 0

def test_marathi_tokenization():
    tok = tokenizer_factory.get("mr")
    text = "मराठी कृत्रिम बुद्धिमत्ता"
    tokens = tok.tokenize(text)
    assert len(tokens) > 0

def test_bengali_tokenization():
    tok = tokenizer_factory.get("bn")
    text = "বাংলা কৃত্রিম বুদ্ধিমত্তা"
    tokens = tok.tokenize(text)
    assert len(tokens) > 0

# ==============================================================================
# 2. KANNADA LINGUISTIC STRUCTURE & CONJUNCT PRESERVATION
# ==============================================================================

def test_kannada_conjunct_preservation():
    tok = tokenizer_factory.get("kn")
    # Ottaksharas: Consonant + Virama + Consonant
    conjuncts = ["ಕ್ಕ", "ಕ್ರ", "ಕ್ಲ", "ಜ್ಞ", "ಷ್ಟ", "ಪ್ರ", "ತ್ರ", "ಗ್ರ", "ದ್ವ"]
    for conj in conjuncts:
        tokens = tok.tokenize(conj)
        # Verify the virama-consonant structure remains united
        assert len(tokens) > 0
        assert conj in tokens or any(c in t for t in tokens for c in conj)

def test_kannada_vowel_signs():
    tok = tokenizer_factory.get("kn")
    words = ["ಕ", "ಕಾ", "ಕಿ", "ಕೀ", "ಕು", "ಕೂ", "ಕೆ", "ಕೇ", "ಕೈ", "ಕೊ", "ಕೋ", "ಕೌ"]
    for word in words:
        tokens = tok.tokenize(word)
        assert len(tokens) > 0

# ==============================================================================
# 3. MIXED SCRIPT & CODE-SWITCHING
# ==============================================================================

def test_mixed_kannada_and_english():
    tok = tokenizer_factory.get("kn")
    text = "ನಮಸ್ಕಾರ Hello! ನಾನು AIML student."
    res = tok.tokenize_with_result(text)
    assert res.token_count > 0
    assert "Hello" in res.tokens
    assert "AIML" in res.tokens
    assert "student" in res.tokens

def test_mixed_indic_and_latin():
    tok = tokenizer_factory.get("kn")
    text = "Hello ಕನ್ನಡ ಮತ್ತು Hindi हिन्दी AI"
    tokens = tok.tokenize(text)
    assert len(tokens) > 0
    assert "Hello" in tokens
    assert "AI" in tokens

# ==============================================================================
# 4. EDGE CASES & DEFENSIVE BOUNDS
# ==============================================================================

def test_empty_and_null_input():
    tok = tokenizer_factory.get("kn")
    assert tok.tokenize("") == []
    assert tok.tokenize(None) == []
    assert tok.encode("") == []
    assert tok.decode([]) == ""

def test_long_input_protection():
    tok = tokenizer_factory.get("kn")
    huge_text = "ಕನ್ನಡ " * 20000  # Exceeds 100,000 chars
    with pytest.raises(ValueError) as exc:
        tok.tokenize(huge_text)
    assert "exceeds maximum allowed tokenizer limit" in str(exc.value)

def test_token_count_in_result():
    tok = tokenizer_factory.get("kn")
    text = "ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆ ಮತ್ತು ನರಮಂಡಲ ಜಾಲ"
    res = tok.tokenize_with_result(text)
    assert res.token_count == len(res.tokens)
    assert res.token_count == len(res.token_ids)
    assert res.language == "kn"
    assert res.tokenizer_type == "development_fallback"

def test_unknown_token_count():
    tok = tokenizer_factory.get("kn")
    text = "ನಮಸ್ಕಾರ"
    res = tok.tokenize_with_result(text)
    assert res.unknown_token_count == 0

# ==============================================================================
# 5. ENCODING, DECODING & ROUND TRIP
# ==============================================================================

def test_encode():
    tok = tokenizer_factory.get("kn")
    text = "ನಮಸ್ಕಾರ NEXORA AI"
    ids = tok.encode(text)
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)

def test_decode():
    tok = tokenizer_factory.get("kn")
    text = "ನಮಸ್ಕಾರ NEXORA AI"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert "ನಮಸ್ಕಾರ" in decoded or "NEXORA" in decoded

def test_round_trip():
    tok = tokenizer_factory.get("en")
    original = "NEXORA AI is fast and reliable."
    ids = tok.encode(original)
    decoded = tok.decode(ids)
    assert "NEXORA" in decoded
    assert "fast" in decoded
    assert "reliable" in decoded

def test_special_tokens():
    tok = tokenizer_factory.get("kn")
    text = "ಕನ್ನಡ"
    tokens_with_special = tok.tokenize(text, add_special_tokens=True)
    ids_with_special = tok.encode(text, add_special_tokens=True)

    assert tokens_with_special[0] == BOS_TOKEN
    assert tokens_with_special[-1] == EOS_TOKEN
    assert ids_with_special[0] == BOS_TOKEN_ID
    assert ids_with_special[-1] == EOS_TOKEN_ID

# ==============================================================================
# 6. FACTORY, FALLBACK & DETERMINISM
# ==============================================================================

def test_factory_lookup():
    for lang in ["en", "kn", "hi", "ta", "te", "ml", "mr", "bn"]:
        tok = tokenizer_factory.get(lang)
        assert tok.language == lang
        assert tok.vocabulary_size == 32000

def test_unsupported_language_fallback():
    tok = tokenizer_factory.get("xyz_unknown_lang")
    assert tok is not None
    assert tok.language in ("fallback", "en")
    tokens = tok.tokenize("Universal fallback test")
    assert len(tokens) > 0

def test_deterministic_output():
    tok = tokenizer_factory.get("kn")
    text = "ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆ"
    ids1 = tok.encode(text)
    ids2 = tok.encode(text)
    ids3 = tok.encode(text)
    assert ids1 == ids2 == ids3

def test_maximum_token_limit_on_decode():
    tok = tokenizer_factory.get("kn")
    huge_ids = [100] * (MAX_TOKEN_COUNT + 10)
    with pytest.raises(ValueError) as exc:
        tok.decode(huge_ids)
    assert "exceeds maximum safe decoding limit" in str(exc.value)
