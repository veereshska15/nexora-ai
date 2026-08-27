import pytest
from fastapi.testclient import TestClient
from nlp.services.multilingual_pipeline import multilingual_pipeline, MultilingualNLPPipeline
from schemas.nlp import NLPAnalyzeResponse

# ==============================================================================
# 1. LANGUAGE & SCRIPT PIPELINE TESTS ACROSS 8 LANGUAGES
# ==============================================================================

def test_pipeline_kannada():
    text = "ನಮಸ್ಕಾರ, ಹೇಗಿದ್ದೀರಾ?"
    res = multilingual_pipeline.analyze(text)
    assert res.language == "kn"
    assert res.script == "Kannada"
    assert "kn" in res.language_candidates
    assert res.confidence >= 0.80
    assert len(res.tokens) > 0
    assert res.token_count == len(res.tokens)
    assert res.processing_time_ms >= 0.0

def test_pipeline_english():
    text = "NEXORA AI is a cutting-edge multimodal platform."
    res = multilingual_pipeline.analyze(text)
    assert res.language == "en"
    assert res.script == "Latin"
    assert res.romanized is False
    assert len(res.tokens) > 0
    assert res.confidence >= 0.80

def test_pipeline_hindi():
    text = "हिन्दी में आपका स्वागत है। हम सब बहुत खुश हैं।"
    res = multilingual_pipeline.analyze(text)
    assert res.language == "hi"
    assert res.script == "Devanagari"
    assert "hi" in res.language_candidates
    assert len(res.tokens) > 0

def test_pipeline_marathi():
    text = "मराठी भाषा अतिशय सुंदर आहे आणि यामध्ये बरेच काही शिकण्यासारखे आहे."
    res = multilingual_pipeline.analyze(text)
    assert res.language == "mr"
    assert res.script == "Devanagari"
    assert "mr" in res.language_candidates
    assert len(res.tokens) > 0

def test_pipeline_tamil():
    text = "தமிழ் இயற்கை மொழி செயலாக்கம்"
    res = multilingual_pipeline.analyze(text)
    assert res.language == "ta"
    assert res.script == "Tamil"
    assert res.confidence >= 0.80
    assert len(res.tokens) > 0

def test_pipeline_telugu():
    text = "తెలుగు సహజ భాషా ప్రాసెసింగ్"
    res = multilingual_pipeline.analyze(text)
    assert res.language == "te"
    assert res.script == "Telugu"
    assert res.confidence >= 0.80
    assert len(res.tokens) > 0

def test_pipeline_malayalam():
    text = "മലയാളം പ്രകൃതിഭാഷാ സംസ്കരണം"
    res = multilingual_pipeline.analyze(text)
    assert res.language == "ml"
    assert res.script == "Malayalam"
    assert res.confidence >= 0.80
    assert len(res.tokens) > 0

def test_pipeline_bengali():
    text = "বাংলা প্রাকৃতিক ভাষা প্রক্রিয়াকরণ"
    res = multilingual_pipeline.analyze(text)
    assert res.language == "bn"
    assert res.script == "Bengali"
    assert res.confidence >= 0.80
    assert len(res.tokens) > 0

# ==============================================================================
# 2. ROMANIZED & MIXED TEXT HANDLING
# ==============================================================================

def test_pipeline_romanized_kannada():
    text = "namaskara nanna hesaru Veeresh, hegiddira neevu?"
    res = multilingual_pipeline.analyze(text)
    assert res.language == "kn"
    assert res.script == "Latin"
    assert res.romanized is True
    assert len(res.tokens) > 0

def test_pipeline_mixed_kannada_english():
    text = "ನಮಸ್ಕಾರ Hello world! Welcome to Bengaluru."
    res = multilingual_pipeline.analyze(text)
    assert res.mixed_language is True
    assert len(res.tokens) > 0

# ==============================================================================
# 3. EDGE CASES & DEFENSIVE BOUNDS
# ==============================================================================

def test_pipeline_empty_input():
    res1 = multilingual_pipeline.analyze("")
    assert res1.language == "unknown"
    assert res1.tokens == []
    assert res1.token_count == 0
    assert res1.ambiguous is True

    res2 = multilingual_pipeline.analyze(None)
    assert res2.language == "unknown"
    assert res2.tokens == []
    assert res2.token_count == 0

def test_pipeline_very_long_input_protection():
    huge_text = "ಕನ್ನಡ " * 25000  # > 100,000 characters
    with pytest.raises(ValueError) as exc:
        multilingual_pipeline.analyze(huge_text)
    assert "exceeds maximum allowed pipeline limit" in str(exc.value)

# ==============================================================================
# 4. TOKEN & NORMALIZATION VERIFICATION
# ==============================================================================

def test_pipeline_token_output():
    text = "ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆ ಮತ್ತು ನರಮಂಡಲ ಜಾಲ"
    res = multilingual_pipeline.analyze(text)
    assert len(res.tokens) == res.token_count
    assert len(res.token_ids) == res.token_count
    assert all(isinstance(t, str) for t in res.tokens)
    assert all(isinstance(tid, int) for tid in res.token_ids)

def test_pipeline_normalization():
    messy_text = "  \ufeffಕನ್ನಡ    ಭಾಷೆ   \n\n\n\n  ಮತ್ತು   AI  "
    res = multilingual_pipeline.analyze(messy_text)
    assert res.normalized_text == "ಕನ್ನಡ ಭಾಷೆ\n\nಮತ್ತು AI"
    assert "\ufeff" not in res.normalized_text

def test_pipeline_confidence_range():
    samples = [
        "ಕನ್ನಡ ಭಾಷೆ",
        "हिन्दी भाषा",
        "This is an English sentence.",
        "namaskara",
        "12345",
    ]
    for s in samples:
        res = multilingual_pipeline.analyze(s)
        assert 0.0 <= res.confidence <= 1.0

def test_pipeline_batch_processing():
    texts = [
        "ನಮಸ್ಕಾರ, ಹೇಗಿದ್ದೀರಾ?",
        "NEXORA AI platform",
        "हिन्दी में स्वागत है",
    ]
    results = multilingual_pipeline.analyze_batch(texts)
    assert len(results) == 3
    assert results[0].language == "kn"
    assert results[1].language == "en"
    assert results[2].language == "hi"

# ==============================================================================
# 5. FASTAPI REST ENDPOINT VERIFICATION
# ==============================================================================

def test_api_nlp_analyze_endpoint(client: TestClient):
    payload = {"text": "ನಮಸ್ಕಾರ, ಹೇಗಿದ್ದೀರಾ?"}
    response = client.post("/api/v1/nlp/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["language"] == "kn"
    assert data["script"] == "Kannada"
    assert data["token_count"] > 0
    assert "tokens" in data
    assert "token_ids" in data
    assert data["processing_time_ms"] >= 0.0

def test_api_nlp_analyze_batch_endpoint(client: TestClient):
    payload = {
        "texts": [
            "ನಮಸ್ಕಾರ ವಿಶ್ವ",
            "Artificial Intelligence",
            "मराठी भाषा आहे",
        ]
    }
    response = client.post("/api/v1/nlp/analyze/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_texts"] == 3
    assert len(data["results"]) == 3
    assert data["results"][0]["language"] == "kn"
    assert data["results"][1]["language"] == "en"
    assert data["results"][2]["language"] == "mr"
