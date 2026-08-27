from fastapi import APIRouter, HTTPException, status
from schemas.embeddings import (
    EmbeddingRequest,
    EmbeddingResponse,
    BatchEmbeddingRequest,
    BatchEmbeddingResponse,
)
from embeddings.embedding_service import embedding_service

router = APIRouter(prefix="/embeddings", tags=["Vector Embeddings"])


@router.post(
    "/embed",
    response_model=EmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate 1536d normalized vector embedding for single text input",
)
async def generate_single_embedding(request: EmbeddingRequest) -> EmbeddingResponse:
    """
    Generates a deterministic 1536-dimensional normalized vector embedding
    for the provided text input without external API keys or network latency.
    """
    try:
        result = embedding_service.embed_single(text=request.text, provider=request.provider)
        return EmbeddingResponse(
            dimension=result.dimension,
            provider=result.provider,
            model=result.model,
            vector=result.vector,
            processing_time_ms=result.processing_time_ms,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation error: {str(err)}",
        )


@router.post(
    "/embed/batch",
    response_model=BatchEmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate 1536d normalized vector embeddings for batch text inputs",
)
async def generate_batch_embeddings(request: BatchEmbeddingRequest) -> BatchEmbeddingResponse:
    """
    Generates deterministic 1536-dimensional normalized vector embeddings
    for a batch list of text inputs.
    """
    try:
        results = embedding_service.embed_batch(texts=request.texts, provider=request.provider)
        return BatchEmbeddingResponse(
            total_embeddings=len(results),
            dimension=results[0].dimension if results else 1536,
            provider=results[0].provider if results else "development_deterministic",
            model=results[0].model if results else "development-1536",
            embeddings=[r.vector for r in results],
            processing_time_ms=round(sum(r.processing_time_ms for r in results), 3),
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch embedding generation error: {str(err)}",
        )
