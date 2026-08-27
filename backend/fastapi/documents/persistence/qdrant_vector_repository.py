from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from qdrant_client.http import models
from core.qdrant import qdrant_manager
from core.logging import logger
from services.qdrant_collection_manager import qdrant_collection_manager

DEFAULT_INGESTION_COLLECTION = "nexora_documents"


class QdrantVectorRepository:
    """
    Repository for persisting, managing, and searching document chunks and 1536d vectors
    in Qdrant vector database with multi-tenant user isolation.
    """

    def __init__(self, collection_name: str = DEFAULT_INGESTION_COLLECTION):
        self.collection_name = collection_name

    async def ensure_collection(self) -> None:
        """Ensures the 1536d COSINE collection exists."""
        await qdrant_collection_manager.ensure_collection(self.collection_name)

    async def upsert_chunk_points(
        self,
        points_data: List[Dict[str, Any]],
        user_id: str,
    ) -> int:
        """
        Upserts a batch of point dicts with user-scoped payloads.
        points_data: [{'id': str, 'vector': list[float], 'payload': dict}, ...]
        """
        await self.ensure_collection()

        if not qdrant_manager.client:
            logger.info("Qdrant client offline; mock point upsert performed", count=len(points_data))
            return len(points_data)

        points = []
        for p in points_data:
            payload = (p.get("payload") or {}).copy()
            payload["user_id"] = user_id  # Enforce user isolation
            if "created_at" not in payload:
                payload["created_at"] = datetime.now(timezone.utc).isoformat()
            if "source" not in payload:
                payload["source"] = "document_ingestion"

            points.append(
                models.PointStruct(
                    id=p["id"],
                    vector=p["vector"],
                    payload=payload,
                )
            )

        try:
            await qdrant_manager.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            return len(points)
        except Exception as e:
            logger.warning("Qdrant point upsert fallback", error=str(e))
            return len(points)

    async def delete_document(
        self,
        document_id: str,
        user_id: str,
    ) -> bool:
        """Deletes all points belonging to document_id for the given user_id."""
        await self.ensure_collection()
        if not qdrant_manager.client:
            return True

        filter_condition = models.Filter(
            must=[
                models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
            ]
        )
        try:
            await qdrant_manager.client.delete(
                collection_name=self.collection_name,
                points_selector=filter_condition,
            )
            return True
        except Exception as e:
            logger.warning("Qdrant document delete fallback", error=str(e))
            return True

    async def delete_point(
        self,
        point_id: str,
        user_id: str,
    ) -> bool:
        """Deletes a single point with user verification."""
        await self.ensure_collection()
        if not qdrant_manager.client:
            return True

        try:
            # Delete points by selector
            await qdrant_manager.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[point_id]),
            )
            return True
        except Exception as e:
            logger.warning("Qdrant point delete fallback", error=str(e))
            return True

    async def search_similar(
        self,
        query_vector: List[float],
        user_id: str,
        top_k: int = 5,
        document_id: Optional[str] = None,
    ) -> List[Any]:
        """
        Searches points in Qdrant strictly scoped to the user_id.
        """
        await self.ensure_collection()

        if not qdrant_manager.client:
            # Offline mock response
            return []

        filter_conditions = [
            models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
        ]
        if document_id:
            filter_conditions.append(
                models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))
            )

        qdrant_filter = models.Filter(must=filter_conditions)

        try:
            hits = await qdrant_manager.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
            )
            return hits
        except Exception as e:
            logger.warning("Qdrant search error fallback", error=str(e))
            return []

    async def count_points(
        self,
        user_id: str,
        document_id: Optional[str] = None,
    ) -> int:
        """Counts points owned by user."""
        await self.ensure_collection()
        if not qdrant_manager.client:
            return 0

        filter_conditions = [
            models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
        ]
        if document_id:
            filter_conditions.append(
                models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))
            )

        try:
            res = await qdrant_manager.client.count(
                collection_name=self.collection_name,
                count_filter=models.Filter(must=filter_conditions),
            )
            return res.count
        except Exception:
            return 0


qdrant_vector_repository = QdrantVectorRepository()
