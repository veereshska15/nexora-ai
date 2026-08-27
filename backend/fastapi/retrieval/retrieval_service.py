import time
from typing import Any, Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import logger
from embeddings.embedding_service import embedding_service, EmbeddingService
from nlp.services.multilingual_pipeline import multilingual_pipeline, MultilingualNLPPipeline
from retrieval.models.retrieval_result import AssembledContext, CandidateChunk, RerankedChunk, RetrievalResult
from retrieval.hybrid_retriever import hybrid_retriever, HybridRetriever
from retrieval.reranker import development_reranker, BaseReranker
from retrieval.context_assembler import context_assembler, ContextAssembler

DEV_DEFAULT_USER_ID = "00000000-0000-0000-0000-000000000001"


class RetrievalService:
    """
    End-to-End Multilingual Hybrid Retrieval, Reranking, and Context Assembly Service.
    Integrates NLP query analysis, 1536d dense embedding generation, Qdrant/pgvector hybrid candidate search,
    multi-signal reranking, and structured RAG context assembly with user isolation.
    """

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        reranker: Optional[BaseReranker] = None,
        assembler: Optional[ContextAssembler] = None,
        embedding_svc: Optional[EmbeddingService] = None,
        nlp_svc: Optional[MultilingualNLPPipeline] = None,
    ):
        self.retriever = retriever or hybrid_retriever
        self.reranker = reranker or development_reranker
        self.assembler = assembler or context_assembler
        self.embedding_service = embedding_svc or embedding_service
        self.nlp_pipeline = nlp_svc or multilingual_pipeline

    async def retrieve(
        self,
        query: str,
        user_id: Optional[str] = None,
        session: Optional[AsyncSession] = None,
        top_k: int = 5,
        candidate_k: int = 20,
        minimum_similarity: float = 0.0,
        max_context_chars: int = 4000,
        max_context_chunks: int = 5,
        document_id: Optional[str] = None,
        document_name: Optional[str] = None,
        language: Optional[str] = None,
        script: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Executes the complete retrieval pipeline:
        1. Validates query and handles edge cases safely
        2. Performs NLP analysis for language/script detection
        3. Embeds query into 1536d vector
        4. Retrieves candidates from Qdrant and pgvector
        5. Reranks candidates using multi-signal scoring
        6. Assembles token/char-bounded RAG context
        """
        start_time = time.perf_counter()
        effective_user_id = user_id or DEV_DEFAULT_USER_ID

        # Edge case: Empty, whitespace, or invalid query
        if not query or not query.strip():
            return RetrievalResult(
                query=query or "",
                detected_language="en",
                detected_script="Latin",
                total_candidates=0,
                results=[],
                context="",
                assembled_context=AssembledContext(
                    context_text="",
                    total_chunks=0,
                    total_characters=0,
                    sources=[],
                    truncated=False,
                ),
                processing_time_ms=0.0,
            )

        clean_query = query.strip()

        # 1. NLP Pipeline Analysis
        try:
            nlp_res = self.nlp_pipeline.analyze(clean_query)
            detected_lang = nlp_res.language or "en"
            detected_script = nlp_res.script or "Latin"
        except Exception as e:
            logger.warning("NLP query analysis fallback", error=str(e))
            detected_lang = "en"
            detected_script = "Latin"

        # 2. Embedding Generation (1536d normalized vector)
        emb_res = self.embedding_service.embed_single(clean_query)
        query_vector = emb_res.vector

        # 3. Hybrid Candidate Retrieval (Qdrant + PostgreSQL)
        effective_candidate_k = max(top_k, candidate_k)
        candidates = await self.retriever.retrieve_candidates(
            query_vector=query_vector,
            user_id=effective_user_id,
            session=session,
            candidate_k=effective_candidate_k,
            minimum_similarity=minimum_similarity,
            document_id=document_id,
            document_name=document_name,
            language=language,
            script=script,
        )

        total_candidates = len(candidates)

        # 4. Multi-Signal Deterministic Reranking
        reranked_chunks = self.reranker.rerank(
            query=clean_query,
            candidates=candidates,
            query_language=detected_lang,
            query_script=detected_script,
            top_k=top_k,
        )

        # 5. Context Assembly
        assembled_context = self.assembler.assemble_context(
            chunks=reranked_chunks,
            max_chars=max_context_chars,
            max_chunks=max_context_chunks,
        )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 3)

        return RetrievalResult(
            query=clean_query,
            detected_language=detected_lang,
            detected_script=detected_script,
            total_candidates=total_candidates,
            results=reranked_chunks,
            context=assembled_context.context_text,
            assembled_context=assembled_context,
            processing_time_ms=elapsed_ms,
        )

    async def retrieve_context(
        self,
        query: str,
        user_id: Optional[str] = None,
        session: Optional[AsyncSession] = None,
        top_k: int = 5,
        max_context_chars: int = 4000,
    ) -> AssembledContext:
        """
        Convenience method to retrieve and assemble only the formatted RAG context.
        """
        res = await self.retrieve(
            query=query,
            user_id=user_id,
            session=session,
            top_k=top_k,
            max_context_chars=max_context_chars,
        )
        return res.assembled_context


retrieval_service = RetrievalService()
