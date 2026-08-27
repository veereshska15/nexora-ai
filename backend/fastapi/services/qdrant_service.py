import uuid
from qdrant_client.http import models
from core.qdrant import qdrant_manager
from core.logging import logger
from schemas.qdrant import (
    QdrantUpsertRequest,
    QdrantUpsertResponse,
    QdrantSearchRequest,
    QdrantSearchResponse,
    QdrantSearchResult,
    DEFAULT_COLLECTION_NAME,
)
from services.qdrant_collection_manager import qdrant_collection_manager

DEV_DEFAULT_USER_ID = "00000000-0000-0000-0000-000000000001"

class QdrantVectorService:

    async def upsert_points(
        self,
        request: QdrantUpsertRequest,
        user_id: str | None = None,
    ) -> QdrantUpsertResponse:
        effective_user_id = user_id or DEV_DEFAULT_USER_ID

        await qdrant_collection_manager.ensure_collection(request.collection_name)

        if not qdrant_manager.client:
            logger.info("Qdrant client offline; mock upsert performed", count=len(request.points))
            return QdrantUpsertResponse(
                upserted_count=len(request.points),
                collection_name=request.collection_name,
                status="completed_mock",
            )

        points = []
        for p in request.points:
            payload = p.payload.model_dump()
            payload["user_id"] = effective_user_id  # Enforce authenticated ownership
            points.append(
                models.PointStruct(
                    id=p.id or str(uuid.uuid4()),
                    vector=p.vector,
                    payload=payload,
                )
            )

        try:
            await qdrant_manager.client.upsert(
                collection_name=request.collection_name,
                points=points,
            )
            logger.info("Upserted points to Qdrant", collection=request.collection_name, count=len(points))
            return QdrantUpsertResponse(
                upserted_count=len(points),
                collection_name=request.collection_name,
                status="completed",
            )
        except Exception as e:
            logger.warning("Qdrant upsert fallback", error=str(e))
            return QdrantUpsertResponse(
                upserted_count=len(points),
                collection_name=request.collection_name,
                status="fallback_mode",
            )

    async def search_points(
        self,
        request: QdrantSearchRequest,
        user_id: str | None = None,
    ) -> QdrantSearchResponse:
        effective_user_id = user_id or DEV_DEFAULT_USER_ID

        await qdrant_collection_manager.ensure_collection(request.collection_name)

        if not qdrant_manager.client:
            logger.info("Qdrant client offline; mock search returned", top_k=request.top_k)
            return QdrantSearchResponse(
                results=[
                    QdrantSearchResult(
                        id=str(uuid.uuid4()),
                        score=0.92,
                        payload={
                            "user_id": effective_user_id,
                            "document_id": "knowledge_architecture.pdf",
                            "chunk_index": 0,
                            "content": "Qdrant provides high-scale Approximate Nearest Neighbor vector search for multi-modal embeddings.",
                            "source": "knowledge_base",
                        },
                    )
                ],
                total_results=1,
                collection_name=request.collection_name,
                dimensions=len(request.query_vector),
            )

        # Build multi-tenant filter: user_id MUST match authenticated user
        filter_conditions = [
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=effective_user_id),
            )
        ]

        if request.document_id:
            filter_conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=request.document_id),
                )
            )

        qdrant_filter = models.Filter(must=filter_conditions)

        try:
            hits = await qdrant_manager.client.search(
                collection_name=request.collection_name,
                query_vector=request.query_vector,
                query_filter=qdrant_filter,
                limit=request.top_k,
            )

            results: list[QdrantSearchResult] = []
            for hit in hits:
                results.append(
                    QdrantSearchResult(
                        id=str(hit.id),
                        score=round(float(hit.score), 4),
                        payload=hit.payload or {},
                    )
                )

            return QdrantSearchResponse(
                results=results,
                total_results=len(results),
                collection_name=request.collection_name,
                dimensions=len(request.query_vector),
            )
        except Exception as e:
            logger.warning("Qdrant search query fallback", error=str(e))
            return QdrantSearchResponse(
                results=[],
                total_results=0,
                collection_name=request.collection_name,
                dimensions=len(request.query_vector),
            )

qdrant_service = QdrantVectorService()
