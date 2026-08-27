from fastapi import APIRouter, status
from schemas.qdrant import (
    QdrantSearchRequest,
    QdrantSearchResponse,
    QdrantUpsertRequest,
    QdrantUpsertResponse,
    QdrantCollectionInfo,
)
from services.qdrant_service import qdrant_service
from services.qdrant_collection_manager import qdrant_collection_manager

router = APIRouter(prefix="/qdrant", tags=["Qdrant Vector Database"])

@router.post("/search", response_model=QdrantSearchResponse)
async def search_qdrant_points(request: QdrantSearchRequest) -> QdrantSearchResponse:
    """
    Executes high-performance approximate nearest neighbor search on Qdrant collections.
    Enforces multi-tenant user isolation by scoping query payloads to the authenticated user.
    """
    return await qdrant_service.search_points(request=request)

@router.post("/upsert", response_model=QdrantUpsertResponse, status_code=status.HTTP_201_CREATED)
async def upsert_qdrant_points(request: QdrantUpsertRequest) -> QdrantUpsertResponse:
    """
    Batch upserts 1536-dimensional vector points and metadata payloads to a Qdrant collection.
    """
    return await qdrant_service.upsert_points(request=request)

@router.get("/collections", response_model=list[QdrantCollectionInfo])
async def list_qdrant_collections() -> list[QdrantCollectionInfo]:
    """
    Lists active Qdrant vector collections and configuration metadata.
    """
    return await qdrant_collection_manager.list_collections()
