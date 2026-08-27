from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db_session
from schemas.rag import RAGQueryRequest, RAGQueryResponse, CitationSchema
from rag.rag_service import rag_service

router = APIRouter(prefix="/rag", tags=["Grounded RAG Generation"])


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="End-to-end multilingual grounded RAG query orchestration",
)
async def query_rag(
    request: RAGQueryRequest,
    db: AsyncSession = Depends(get_db_session),
) -> RAGQueryResponse:
    """
    Executes full RAG workflow:
    NLP analysis -> Dense embedding -> Hybrid retrieval -> Multi-signal reranking ->
    Prompt assembly -> Grounded LLM generation -> Citation extraction & Grounding Guard.
    """
    try:
        res = await rag_service.query(
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
        )

        return RAGQueryResponse(
            query=res.query,
            answer=res.answer,
            language=res.language,
            script=res.script,
            grounded=res.grounded,
            grounding_confidence=res.grounding_confidence,
            citations=[CitationSchema(**c.model_dump()) for c in res.citations],
            retrieved_chunks=res.retrieved_chunks,
            provider=res.provider,
            model=res.model,
            processing_time_ms=res.processing_time_ms,
            warnings=res.warnings,
        )

    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG query orchestration error: {str(err)}",
        )


@router.post(
    "/query/stream",
    status_code=status.HTTP_200_OK,
    summary="Streaming grounded RAG token generation",
)
async def query_rag_stream(
    request: RAGQueryRequest,
    db: AsyncSession = Depends(get_db_session),
) -> StreamingResponse:
    """
    Streams grounded response tokens asynchronously.
    """
    async def token_generator():
        try:
            async for token in rag_service.query_stream(
                query=request.query,
                user_id=request.user_id,
                session=db,
                top_k=request.top_k,
            ):
                yield token
        except Exception as err:
            yield f"\n[Streaming error: {str(err)}]"

    return StreamingResponse(token_generator(), media_type="text/plain")
