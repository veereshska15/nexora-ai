import time
from typing import Any, AsyncGenerator, Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import logger
from rag.models.citation import Citation
from rag.models.rag_result import RAGResponse
from rag.providers.base_llm import BaseLLMProvider
from rag.providers.development_llm import development_llm
from rag.prompt.grounded_prompt import grounded_prompt_builder, GroundedPromptBuilder
from rag.citation.citation_service import citation_service, CitationService
from rag.guardrails.grounding_guard import grounding_guard, GroundingGuard
from retrieval.retrieval_service import retrieval_service, RetrievalService

DEV_DEFAULT_USER_ID = "00000000-0000-0000-0000-000000000001"


class RAGService:
    """
    End-to-End Multilingual Grounded RAG Query Orchestration Service.
    Coordinates query NLP analysis, hybrid vector retrieval, multi-signal reranking,
    prompt assembly, grounded LLM synthesis, citation extraction, and safety guardrails.
    """

    def __init__(
        self,
        retrieval_svc: Optional[RetrievalService] = None,
        llm: Optional[BaseLLMProvider] = None,
        prompt_builder: Optional[GroundedPromptBuilder] = None,
        citation_svc: Optional[CitationService] = None,
        guard: Optional[GroundingGuard] = None,
    ):
        self.retrieval_service = retrieval_svc or retrieval_service
        self.llm = llm or development_llm
        self.prompt_builder = prompt_builder or grounded_prompt_builder
        self.citation_service = citation_svc or citation_service
        self.grounding_guard = guard or grounding_guard

    async def query(
        self,
        query: str,
        user_id: Optional[str] = None,
        session: Optional[AsyncSession] = None,
        top_k: int = 5,
        candidate_k: int = 20,
        minimum_similarity: float = 0.0,
        max_context_chars: int = 4000,
        document_id: Optional[str] = None,
        document_name: Optional[str] = None,
        language: Optional[str] = None,
    ) -> RAGResponse:
        """
        Executes end-to-end grounded RAG generation for a user query.
        """
        start_time = time.perf_counter()
        effective_user_id = user_id or DEV_DEFAULT_USER_ID

        # 1. Handle empty / invalid query edge cases
        if not query or not query.strip():
            return RAGResponse(
                query=query or "",
                answer="Please provide a valid question to search.",
                language="en",
                script="Latin",
                grounded=False,
                grounding_confidence=0.0,
                citations=[],
                retrieved_chunks=0,
                provider=self.llm.provider_name,
                model=self.llm.model_name,
                processing_time_ms=0.0,
                warnings=["Empty query provided."],
            )

        # 2. Retrieve candidates & assemble context via RetrievalService
        retrieval_res = await self.retrieval_service.retrieve(
            query=query,
            user_id=effective_user_id,
            session=session,
            top_k=top_k,
            candidate_k=candidate_k,
            minimum_similarity=minimum_similarity,
            max_context_chars=max_context_chars,
            document_id=document_id,
            document_name=document_name,
            language=language,
        )

        detected_lang = retrieval_res.detected_language
        detected_script = retrieval_res.detected_script
        retrieved_chunks = retrieval_res.results
        context_text = retrieval_res.context

        # 3. Build Citations from retrieved chunks
        all_citations = self.citation_service.build_citations(retrieved_chunks)

        # 4. If no relevant chunks retrieved: return grounded unavailability response
        if not retrieved_chunks:
            answer = "I could not find relevant information in the documents available to me."
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 3)
            return RAGResponse(
                query=query,
                answer=answer,
                language=detected_lang,
                script=detected_script,
                grounded=False,
                grounding_confidence=0.0,
                citations=[],
                retrieved_chunks=0,
                provider=self.llm.provider_name,
                model=self.llm.model_name,
                processing_time_ms=elapsed_ms,
                warnings=["No relevant document context found."],
            )

        # 5. Build strict grounded prompt
        grounded_prompt = self.prompt_builder.build_prompt(
            query=query,
            context=context_text,
            detected_language=detected_lang,
            detected_script=detected_script,
        )

        # 6. Generate answer via LLM Provider
        answer = await self.llm.generate(prompt=grounded_prompt)

        # 7. Extract referenced citations
        referenced_citations = self.citation_service.extract_referenced_citations(answer, all_citations)

        # 8. Evaluate grounding integrity with GroundingGuard
        is_grounded, confidence, warnings = self.grounding_guard.evaluate_grounding(
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            citations=all_citations,
        )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 3)

        return RAGResponse(
            query=query,
            answer=answer,
            language=detected_lang,
            script=detected_script,
            grounded=is_grounded,
            grounding_confidence=confidence,
            citations=referenced_citations,
            retrieved_chunks=len(retrieved_chunks),
            provider=self.llm.provider_name,
            model=self.llm.model_name,
            processing_time_ms=elapsed_ms,
            warnings=warnings,
        )

    async def query_stream(
        self,
        query: str,
        user_id: Optional[str] = None,
        session: Optional[AsyncSession] = None,
        top_k: int = 5,
    ) -> AsyncGenerator[str, None]:
        """
        Streams generated RAG tokens asynchronously.
        """
        effective_user_id = user_id or DEV_DEFAULT_USER_ID

        retrieval_res = await self.retrieval_service.retrieve(
            query=query,
            user_id=effective_user_id,
            session=session,
            top_k=top_k,
        )

        if not retrieval_res.results:
            yield "I could not find relevant information in the documents available to me."
            return

        grounded_prompt = self.prompt_builder.build_prompt(
            query=query,
            context=retrieval_res.context,
            detected_language=retrieval_res.detected_language,
            detected_script=retrieval_res.detected_script,
        )

        async for chunk in self.llm.generate_stream(prompt=grounded_prompt):
            yield chunk


rag_service = RAGService()
