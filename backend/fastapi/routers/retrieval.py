from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db_session
from schemas.retrieval import (
    RetrievalSearchRequest,
    RetrievalSearchResponse,
    RetrievalContextRequest,
    RetrievalContextResponse,
    RerankedChunkSchema,
)
from retrieval.retrieval_service import retrieval_service

router = APIRouter(prefix="/retrieval", tags=["Hybrid Retrieval & Reranking"])


@router.post(
    "/search",
    response_model=RetrievalSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Hybrid multi-store retrieval with deterministic multi-signal reranking and context assembly",
)
async def search_and_rerank(
    request: RetrievalSearchRequest,
    db: AsyncSession = Depends(get_db_session),
) -> RetrievalSearchResponse:
    """
    Executes hybrid retrieval across Qdrant and PostgreSQL pgvector, fuses candidate scores,
    applies multi-signal reranking (vector similarity + lexical overlap + language/script matching),
    and formats assembled RAG context with user isolation.
    """
    try:
        res = await retrieval_service.retrieve(
            query=request.query,
            user_id=request.user_id,
            session=db,
            top_k=request.top_k,
            candidate_k=request.candidate_k,
            minimum_similarity=request.minimum_similarity,
            max_context_chars=request.max_context_chars,
            document_id=request.document_id,
            document_name=request.document_name,
            language=request.language,
            script=request.script,
        )

        return RetrievalSearchResponse(
            query=res.query,
            language=res.detected_language,
            script=res.detected_script,
            total_candidates=res.total_candidates,
            results=[RerankedChunkSchema(**r.model_dump()) for r in res.results],
            context=res.context,
            processing_time_ms=res.processing_time_ms,
        )

    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hybrid retrieval error: {str(err)}",
        )


@router.post(
    "/context",
    response_model=RetrievalContextResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve assembled RAG context for a search query",
)
async def retrieve_rag_context(
    request: RetrievalContextRequest,
    db: AsyncSession = Depends(get_db_session),
) -> RetrievalContextResponse:
    """
    Directly returns the structured markdown context assembled from the most relevant chunks.
    """
    try:
        res = await retrieval_service.retrieve(
            query=request.query,
            user_id=request.user_id,
            session=db,
            top_k=request.top_k,
            max_context_chars=request.max_context_chars,
        )

        return RetrievalContextResponse(
            query=res.query,
            context=res.assembled_context.context_text,
            total_chunks=res.assembled_context.total_chunks,
            total_characters=res.assembled_context.total_characters,
            truncated=res.assembled_context.truncated,
            processing_time_ms=res.processing_time_ms,
        )

    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Context retrieval error: {str(err)}",
        )
