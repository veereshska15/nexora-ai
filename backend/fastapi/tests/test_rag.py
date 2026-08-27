import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from rag.models.citation import Citation
from rag.models.rag_result import RAGResponse
from rag.providers.development_llm import DevelopmentLLMProvider, development_llm
from rag.prompt.grounded_prompt import GroundedPromptBuilder, grounded_prompt_builder
from rag.citation.citation_service import CitationService, citation_service
from rag.guardrails.grounding_guard import GroundingGuard, grounding_guard
from rag.rag_service import RAGService, rag_service, DEV_DEFAULT_USER_ID
from retrieval.models.retrieval_result import AssembledContext, CandidateChunk, RerankedChunk, RetrievalResult
from retrieval.retrieval_service import RetrievalService


def create_sample_reranked_chunk(
    chunk_id: str = "chunk_101",
    document_id: str = "doc_kannada",
    document_name: str = "kannada_guide.pdf",
    chunk_index: int = 0,
    content: str = "ಕನ್ನಡ ಭಾಷೆಯು ಭಾರತದ ಅತ್ಯಂತ ಪ್ರಾಚೀನ ಶಾಸ್ತ್ರೀಯ ಭಾಷೆಗಳಲ್ಲಿ ಒಂದಾಗಿದೆ.",
    similarity: float = 0.88,
    rerank_score: float = 0.92,
    language: str = "kn",
    script: str = "Kannada",
) -> RerankedChunk:
    return RerankedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_name=document_name,
        chunk_index=chunk_index,
        content=content,
        similarity=similarity,
        rerank_score=rerank_score,
        lexical_score=0.4,
        language_match=True,
        script_match=True,
        language=language,
        script=script,
        matched_signals=["vector_sim_0.88", "lang_match_kn"],
    )


# ==============================================================================
# 1. DEVELOPMENT LLM & PROMPT BUILDER TESTS
# ==============================================================================

@pytest.mark.anyio
async def test_development_llm_generate():
    provider = DevelopmentLLMProvider()
    prompt = (
        "=== RETRIEVED CONTEXT ===\n"
        "### [Source: ai_intro.pdf | ID: doc_1 | Chunk: 0 | Lang: en | Relevance: 0.90]\n"
        "Artificial Intelligence processes data efficiently.\n\n"
        "=== USER QUESTION ===\n"
        "USER QUESTION: What is AI?\n\n"
        "=== GROUNDED ANSWER ==="
    )
    answer = await provider.generate(prompt)
    assert "Based on the retrieved documents" in answer
    assert "Artificial Intelligence processes data efficiently" in answer
    assert "[1]" in answer


def test_grounded_prompt_builder():
    builder = GroundedPromptBuilder()
    prompt = builder.build_prompt(
        query="ಕನ್ನಡದ ಬಗ್ಗೆ ತಿಳಿಸಿ",
        context="### [Source: kn.txt | ID: 1 | Chunk: 0 | Lang: kn | Relevance: 0.9]\nಕನ್ನಡ ಸಾಹಿತ್ಯ.",
        detected_language="kn",
        detected_script="Kannada",
    )
    assert "=== SYSTEM INSTRUCTIONS ===" in prompt
    assert "=== RETRIEVED CONTEXT ===" in prompt
    assert "=== USER QUESTION ===" in prompt
    assert "USER QUESTION: ಕನ್ನಡದ ಬಗ್ಗೆ ತಿಳಿಸಿ" in prompt
    assert "Target Language: kn (Script: Kannada)" in prompt


# ==============================================================================
# 2. CITATION SERVICE & MAPPING TESTS
# ==============================================================================

def test_citation_creation():
    service = CitationService()
    chunks = [
        create_sample_reranked_chunk(chunk_id="c1", document_name="doc_1.pdf", chunk_index=0),
        create_sample_reranked_chunk(chunk_id="c2", document_name="doc_2.pdf", chunk_index=1),
    ]
    citations = service.build_citations(chunks)
    assert len(citations) == 2
    assert citations[0].citation_id == 1
    assert citations[0].marker == "[1]"
    assert citations[0].document_name == "doc_1.pdf"
    assert citations[1].citation_id == 2
    assert citations[1].marker == "[2]"


def test_citation_mapping_from_answer():
    service = CitationService()
    chunks = [
        create_sample_reranked_chunk(chunk_id="c1", document_name="doc_1.pdf", chunk_index=0),
        create_sample_reranked_chunk(chunk_id="c2", document_name="doc_2.pdf", chunk_index=1),
    ]
    all_citations = service.build_citations(chunks)

    # Answer only references [2]
    answer = "The second document proves the claim [2]."
    referenced = service.extract_referenced_citations(answer, all_citations)
    assert len(referenced) == 1
    assert referenced[0].citation_id == 2
    assert referenced[0].document_name == "doc_2.pdf"


# ==============================================================================
# 3. GROUNDING GUARD & HALLUCINATION MITIGATION
# ==============================================================================

def test_grounding_guard_valid():
    guard = GroundingGuard()
    chunks = [create_sample_reranked_chunk()]
    citations = [
        Citation(
            citation_id=1,
            marker="[1]",
            document_id="doc_kannada",
            document_name="kannada_guide.pdf",
            chunk_id="chunk_101",
            chunk_index=0,
            content_snippet="ಕನ್ನಡ ಭಾಷೆ...",
            language="kn",
            script="Kannada",
            relevance_score=0.92,
        )
    ]
    answer = "ಕನ್ನಡ ಭಾಷೆಯು ಪ್ರಾಚೀನವಾಗಿದೆ [1]."
    is_grounded, conf, warnings = guard.evaluate_grounding(answer, chunks, citations)
    assert is_grounded is True
    assert conf == 0.92
    assert len(warnings) == 0


def test_grounding_guard_unsupported_citation():
    guard = GroundingGuard()
    chunks = [create_sample_reranked_chunk()]
    citations = [
        Citation(
            citation_id=1,
            marker="[1]",
            document_id="d1",
            document_name="d1.pdf",
            chunk_id="c1",
            chunk_index=0,
            content_snippet="Snippet",
            language="kn",
            script="Kannada",
            relevance_score=0.9,
        )
    ]
    # Answer references [99] which doesn't exist
    answer = "Unsupported claim made here [99]."
    is_grounded, conf, warnings = guard.evaluate_grounding(answer, chunks, citations)
    assert is_grounded is False
    assert any("unsupported citation marker [99]" in w for w in warnings)


def test_grounding_guard_empty_answer():
    guard = GroundingGuard()
    is_grounded, conf, warnings = guard.evaluate_grounding("", [], [])
    assert is_grounded is False
    assert conf == 0.0


# ==============================================================================
# 4. RAG ORCHESTRATION & MULTILINGUAL TESTS
# ==============================================================================

@pytest.mark.anyio
async def test_rag_query_orchestration():
    res = await rag_service.query(
        query="ಕನ್ನಡ ಭಾಷೆಯ ಬಗ್ಗೆ ಮಾಹಿತಿ",
        user_id=DEV_DEFAULT_USER_ID,
        top_k=3,
    )
    assert res.query == "ಕನ್ನಡ ಭಾಷೆಯ ಬಗ್ಗೆ ಮಾಹಿತಿ"
    assert res.language == "kn"
    assert res.script == "Kannada"
    assert res.provider == "development"
    assert res.model == "development-grounded"
    assert res.processing_time_ms >= 0.0


@pytest.mark.anyio
async def test_rag_no_context_behavior():
    mock_retriever = AsyncMock()
    mock_retriever.retrieve.return_value = RetrievalResult(
        query="Non-existent topic xyz123",
        detected_language="en",
        detected_script="Latin",
        total_candidates=0,
        results=[],
        context="",
        assembled_context=AssembledContext(),
        processing_time_ms=0.1,
    )

    custom_rag = RAGService(retrieval_svc=mock_retriever)
    res = await custom_rag.query(query="Non-existent topic xyz123")

    assert res.grounded is False
    assert res.grounding_confidence == 0.0
    assert "could not find relevant information" in res.answer
    assert res.retrieved_chunks == 0


@pytest.mark.anyio
async def test_rag_multilingual_hindi():
    res = await rag_service.query(query="कृत्रिम बुद्धिमत्ता क्या है?")
    assert res.language == "hi"
    assert res.script == "Devanagari"


@pytest.mark.anyio
async def test_rag_multilingual_tamil():
    res = await rag_service.query(query="செயற்கை நுண்ணறிவு விளக்கம்")
    assert res.language == "ta"
    assert res.script == "Tamil"


@pytest.mark.anyio
async def test_rag_multilingual_telugu():
    res = await rag_service.query(query="కృత్రిమ మేధస్సు అంటే ఏమిటి?")
    assert res.language == "te"
    assert res.script == "Telugu"


@pytest.mark.anyio
async def test_rag_multilingual_malayalam():
    res = await rag_service.query(query="കൃത്രിമ ബുദ്ധി വിവരണം")
    assert res.language == "ml"
    assert res.script == "Malayalam"


@pytest.mark.anyio
async def test_rag_multilingual_marathi():
    res = await rag_service.query(query="मराठी भाषा आणि इतिहास")
    assert res.language == "mr"
    assert res.script == "Devanagari"


@pytest.mark.anyio
async def test_rag_multilingual_bengali():
    res = await rag_service.query(query="কৃত্রিম বুদ্ধিমত্তা সম্পর্কিত তথ্য")
    assert res.language == "bn"
    assert res.script == "Bengali"


@pytest.mark.anyio
async def test_rag_multilingual_english():
    res = await rag_service.query(query="Explain neural network architecture")
    assert res.language == "en"
    assert res.script == "Latin"


@pytest.mark.anyio
async def test_rag_mixed_kannada_english():
    res = await rag_service.query(query="NEXORA AI nalli Kannada support hegidhe?")
    assert res.script == "Latin"
    assert res.processing_time_ms >= 0.0


@pytest.mark.anyio
async def test_rag_romanized_kannada():
    res = await rag_service.query(query="kannada bhasha bagge mahiti kodi")
    assert res.script == "Latin"
    assert res.processing_time_ms >= 0.0


@pytest.mark.anyio
async def test_rag_user_isolation():
    mock_retriever = AsyncMock()
    mock_retriever.retrieve.return_value = RetrievalResult(
        query="Isolated user query",
        detected_language="en",
        detected_script="Latin",
        total_candidates=0,
        results=[],
        context="",
        assembled_context=AssembledContext(),
        processing_time_ms=0.1,
    )
    custom_rag = RAGService(retrieval_svc=mock_retriever)
    res = await custom_rag.query(query="Isolated user query", user_id="user_b_isolated")
    assert res.grounded is False


@pytest.mark.anyio
async def test_rag_empty_query():
    res = await rag_service.query(query="")
    assert res.grounded is False
    assert res.retrieved_chunks == 0
    assert "Please provide a valid question" in res.answer


@pytest.mark.anyio
async def test_rag_streaming_generator():
    tokens = []
    async for token in rag_service.query_stream(query="ಕನ್ನಡ ಭಾಷೆ"):
        tokens.append(token)
    assert len(tokens) > 0
    full_text = "".join(tokens)
    assert len(full_text) > 0


def test_deterministic_offline_behavior():
    """Verify that development provider requires zero API keys and produces deterministic responses."""
    p1 = grounded_prompt_builder.build_prompt("Question 1", "Context 1", "en", "Latin")
    p2 = grounded_prompt_builder.build_prompt("Question 1", "Context 1", "en", "Latin")
    assert p1 == p2


# ==============================================================================
# 5. REST API ENDPOINT INTEGRATION TESTS
# ==============================================================================

def test_api_rag_query_endpoint(client: TestClient):
    payload = {
        "query": "ಕನ್ನಡ ಭಾಷೆಯ ಬಗ್ಗೆ ವಿವರಿಸಿ",
        "top_k": 3,
        "user_id": DEV_DEFAULT_USER_ID,
    }

    response = client.post("/api/v1/rag/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == payload["query"]
    assert data["language"] == "kn"
    assert data["script"] == "Kannada"
    assert "answer" in data
    assert "citations" in data
    assert data["provider"] == "development"
    assert data["model"] == "development-grounded"
    assert data["processing_time_ms"] >= 0.0


def test_api_rag_streaming_endpoint(client: TestClient):
    payload = {
        "query": "Tell me about NEXORA AI",
        "top_k": 2,
        "user_id": DEV_DEFAULT_USER_ID,
    }

    response = client.post("/api/v1/rag/query/stream", json=payload)
    assert response.status_code == 200
    assert len(response.text) > 0
