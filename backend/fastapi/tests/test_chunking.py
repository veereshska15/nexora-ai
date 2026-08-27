import pytest
from fastapi.testclient import TestClient

from documents.chunking.chunking_service import chunking_service
from documents.chunking.recursive_chunker import CharacterChunker, RecursiveChunker
from documents.chunking.token_aware_chunker import TokenAwareChunker


# ==============================================================================
# 1. CORE CHUNKING STRATEGY TESTS
# ==============================================================================

def test_character_chunking():
    text = "NEXORA AI is a cutting-edge multimodal intelligence platform designed for enterprise scale."
    chunker = CharacterChunker()
    chunks = chunker.split(
        text=text,
        document_id="doc_101",
        document_name="intro.txt",
        chunk_size=30,
        chunk_overlap=5,
    )

    assert len(chunks) > 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].character_count <= 30
    assert chunks[0].document_id == "doc_101"
    assert chunks[0].document_name == "intro.txt"


def test_recursive_chunking():
    paragraphs = [
        "Paragraph 1: NEXORA AI provides real-time neural streaming.",
        "Paragraph 2: Multilingual NLP handles Kannada and Indic scripts natively.",
        "Paragraph 3: Enterprise auth provides role-based access control.",
    ]
    text = "\n\n".join(paragraphs)
    chunker = RecursiveChunker()
    chunks = chunker.split(
        text=text,
        document_id="doc_102",
        document_name="multi.txt",
        chunk_size=80,
        chunk_overlap=10,
    )

    assert len(chunks) >= 2
    assert all(c.character_count <= 100 for c in chunks)
    assert chunks[0].content.startswith("Paragraph 1")


def test_token_aware_chunking():
    text = "Artificial intelligence and neural network embeddings enable semantically rich vector retrieval."
    chunker = TokenAwareChunker()
    chunks = chunker.split(
        text=text,
        document_id="doc_103",
        document_name="tokens.txt",
        chunk_size=6,
        chunk_overlap=2,
    )

    assert len(chunks) >= 2
    assert all(c.token_count <= 6 for c in chunks)
    assert all(c.token_count > 0 for c in chunks)


def test_chunk_overlap():
    text = "Alpha Beta Gamma Delta Epsilon Zeta Eta Theta Iota Kappa Lambda Mu"
    res = chunking_service.chunk_text(
        text=text,
        strategy="character",
        chunk_size=25,
        chunk_overlap=10,
    )

    assert res.total_chunks > 1
    # Check that chunk 1 contains characters from the end of chunk 0
    chunk0_end = res.chunks[0].content[-10:]
    assert any(word in res.chunks[1].content for word in chunk0_end.split() if len(word) > 2)


def test_chunk_ordering():
    text = "Section 1 content. Section 2 content. Section 3 content. Section 4 content."
    res = chunking_service.chunk_text(
        text=text,
        strategy="recursive",
        chunk_size=30,
        chunk_overlap=5,
    )

    for i, c in enumerate(res.chunks):
        assert c.chunk_index == i
        assert c.start_offset <= c.end_offset
        assert c.character_count == len(c.content)


# ==============================================================================
# 2. VALIDATION & ERROR HANDLING TESTS
# ==============================================================================

def test_empty_text_rejection():
    with pytest.raises(ValueError) as exc:
        chunking_service.chunk_text(text="", strategy="recursive")
    assert "empty" in str(exc.value).lower()

    with pytest.raises(ValueError) as exc:
        chunking_service.chunk_text(text="   \n\t  ", strategy="character")
    assert "empty" in str(exc.value).lower()


def test_invalid_chunk_size():
    with pytest.raises(ValueError) as exc:
        chunking_service.chunk_text(text="Sample text", chunk_size=0)
    assert "greater than 0" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        chunking_service.chunk_text(text="Sample text", chunk_size=60000)
    assert "exceeds maximum allowed limit" in str(exc.value)


def test_invalid_overlap():
    with pytest.raises(ValueError) as exc:
        chunking_service.chunk_text(text="Sample text", chunk_size=100, chunk_overlap=-5)
    assert "non-negative" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        chunking_service.chunk_text(text="Sample text", chunk_size=100, chunk_overlap=100)
    assert "smaller than chunk_size" in str(exc.value)


# ==============================================================================
# 3. INDIC & MULTILINGUAL SCRIPT TESTS
# ==============================================================================

def test_kannada_text():
    kannada_text = (
        "ಕನ್ನಡ ಭಾಷೆಯು ಭಾರತದ ಅತ್ಯಂತ ಪ್ರಾಚೀನ ಮತ್ತು ಸುಂದರ ಭಾಷೆಗಳಲ್ಲಿ ಒಂದಾಗಿದೆ.\n\n"
        "ನೆಕ್ಸೋರಾ ಎಐ ಕನ್ನಡ ಭಾಷೆಗೆ ಪ್ರಥಮ ಆದ್ಯತೆ ನೀಡುತ್ತದೆ ಮತ್ತು ನೈಸರ್ಗಿಕ ಸಂಸ್ಕರಣೆ ಮಾಡುತ್ತದೆ."
    )
    res = chunking_service.chunk_text(
        text=kannada_text,
        strategy="recursive",
        chunk_size=70,
        chunk_overlap=10,
    )

    assert res.total_chunks >= 2
    assert "ಕನ್ನಡ" in res.chunks[0].content
    assert res.chunks[0].token_count > 0
    assert res.chunks[0].metadata.get("language") == "kn"


def test_kannada_ottakshara_preservation():
    # Conjuncts: ಬುದ್ಧಿಮತ್ತೆ, ನರಮಂಡಲ, ವಿಶ್ಲೇಷಣೆ, ದೃಷ್ಟಿಕೋನ
    text = "ಬುದ್ಧಿಮತ್ತೆ ನರಮಂಡಲ ವಿಶ್ಲೇಷಣೆ ದೃಷ್ಟಿಕೋನ ರಾಷ್ಟ್ರ ವ್ಯಾಪ್ತಿ"
    chunker = CharacterChunker()
    chunks = chunker.split(
        text=text,
        document_id="doc_kn",
        document_name="kn.txt",
        chunk_size=15,
        chunk_overlap=0,
    )

    # Verify no chunk ends with a dangling virama
    for c in chunks:
        assert not c.content.endswith("\u0CCD")


def test_mixed_kannada_english_text():
    text = "ನಮಸ್ಕಾರ NEXORA AI! Welcome to our multilingual artificial intelligence platform."
    res = chunking_service.chunk_text(
        text=text,
        strategy="recursive",
        chunk_size=40,
        chunk_overlap=5,
    )

    assert res.total_chunks >= 2
    assert any("ನಮಸ್ಕಾರ" in c.content for c in res.chunks)
    assert any("Welcome" in c.content for c in res.chunks)


def test_hindi_text():
    hindi_text = (
        "हिन्दी भारत की प्रमुख भाषा है। यह देवनागरी लिपि में लिखी जाती है।\n\n"
        "नेक्सोरा एआई हिन्दी भाषा में प्राकृतिक भाषा प्रसंस्करण की सुविधा देता है।"
    )
    res = chunking_service.chunk_text(
        text=hindi_text,
        strategy="recursive",
        chunk_size=65,
        chunk_overlap=10,
    )

    assert res.total_chunks >= 2
    assert any("हिन्दी" in c.content for c in res.chunks)


def test_tamil_text():
    tamil_text = "தமிழ் திராவிட மொழிக் குடும்பத்தின் முதன்மையான மொழியாகும்.\n\nநெக்ஸோரா ஏஐ தமிழ் மொழியை ஆதரிக்கிறது."
    res = chunking_service.chunk_text(
        text=tamil_text,
        strategy="recursive",
        chunk_size=60,
        chunk_overlap=5,
    )

    assert res.total_chunks >= 2
    assert "தமிழ்" in res.chunks[0].content


def test_telugu_text():
    telugu_text = "తెలుగు భాష భారతదేశంలోని ప్రముఖ ద్రావిడ భాషలలో ఒకటి.\n\nనెక్సోరా ఏఐ తెలుగు భాషలో నాణ్యమైన ఫలితాలను అందిస్తుంది."
    res = chunking_service.chunk_text(
        text=telugu_text,
        strategy="recursive",
        chunk_size=60,
        chunk_overlap=5,
    )

    assert res.total_chunks >= 2
    assert "తెలుగు" in res.chunks[0].content


# ==============================================================================
# 4. TELEMETRY, METADATA & LARGE TEXT TESTS
# ==============================================================================

def test_token_count_generation():
    text = "Quantum computing and topological quantum field theory."
    res = chunking_service.chunk_text(
        text=text,
        strategy="token",
        chunk_size=5,
        chunk_overlap=1,
    )

    for c in res.chunks:
        assert c.token_count > 0
        assert isinstance(c.token_count, int)


def test_metadata_preservation():
    text = "Confidential financial analysis report 2026."
    custom_meta = {"author": "Veeresh", "department": "AI Research", "confidential": True}
    res = chunking_service.chunk_text(
        text=text,
        strategy="recursive",
        metadata=custom_meta,
    )

    assert res.chunks[0].metadata["author"] == "Veeresh"
    assert res.chunks[0].metadata["department"] == "AI Research"
    assert res.chunks[0].metadata["confidential"] is True


def test_large_text():
    paragraphs = [f"Paragraph {i}: This is automated enterprise knowledge base text block {i}." for i in range(50)]
    large_text = "\n\n".join(paragraphs)

    res = chunking_service.chunk_text(
        text=large_text,
        document_name="large_doc.txt",
        strategy="recursive",
        chunk_size=300,
        chunk_overlap=50,
    )

    assert res.total_chunks > 5
    assert res.processing_time_ms >= 0.0
    for i, c in enumerate(res.chunks):
        assert c.chunk_index == i
        assert c.character_count > 0


# ==============================================================================
# 5. REST API ENDPOINT INTEGRATION TESTS
# ==============================================================================

def test_api_chunk_endpoint(client: TestClient):
    payload = {
        "text": "NEXORA AI is a cutting-edge multimodal platform.\n\nIt features subword tokenization and Indic NLP.",
        "document_name": "overview.txt",
        "document_id": "doc_api_01",
        "strategy": "recursive",
        "chunk_size": 60,
        "chunk_overlap": 10,
        "metadata": {"source": "api_test"},
    }

    response = client.post("/api/v1/documents/chunk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["document_name"] == "overview.txt"
    assert data["strategy"] == "recursive"
    assert data["total_chunks"] >= 2
    assert len(data["chunks"]) == data["total_chunks"]
    assert data["chunks"][0]["metadata"]["source"] == "api_test"
    assert data["processing_time_ms"] >= 0.0
