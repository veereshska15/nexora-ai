import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db_session
from schemas.vector import (
    VectorSearchRequest,
    VectorSearchResponse,
    VectorIngestRequest,
    VectorIngestResponse,
)
from services.vector_search_service import vector_search_service

router = APIRouter(prefix="/vector", tags=["Vector Search & Embeddings"])

@router.post("/search", response_model=VectorSearchResponse)
async def search_vectors(
    request: VectorSearchRequest,
    db: AsyncSession = Depends(get_db_session),
) -> VectorSearchResponse:
    """
    Performs cosine similarity approximate nearest neighbor search using pgvector and HNSW index.
    Restricts search results strictly to the authenticated user scope.
    """
    return await vector_search_service.search_similar_chunks(
        request=request,
        session=db,
    )

@router.post("/ingest", response_model=VectorIngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_vector_chunk(
    request: VectorIngestRequest,
    db: AsyncSession = Depends(get_db_session),
) -> VectorIngestResponse:
    """
    Development endpoint for inserting a pre-computed 1536-dimensional embedding vector and text chunk.
    """
    return await vector_search_service.ingest_chunk(
        request=request,
        session=db,
    )

@router.delete("/{chunk_id}", response_model=dict)
async def delete_vector_chunk(
    chunk_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """
    Deletes a specific document chunk by UUID with user scope verification.
    """
    deleted = await vector_search_service.delete_chunk(
        chunk_id=chunk_id,
        session=db,
    )
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document chunk not found or unauthorized",
        )
    return {"message": "Document chunk deleted successfully", "id": str(chunk_id)}
