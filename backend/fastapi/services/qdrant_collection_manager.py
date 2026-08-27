from qdrant_client.http import models
from core.qdrant import qdrant_manager
from core.logging import logger
from schemas.qdrant import QdrantCollectionInfo, DEFAULT_COLLECTION_NAME, QDRANT_VECTOR_DIMENSION

class QdrantCollectionManager:

    async def collection_exists(self, collection_name: str = DEFAULT_COLLECTION_NAME) -> bool:
        if not qdrant_manager.client:
            return False
        try:
            collections = await qdrant_manager.client.get_collections()
            return any(c.name == collection_name for c in collections.collections)
        except Exception as e:
            logger.warning("Error checking Qdrant collection existence", error=str(e))
            return False

    async def ensure_collection(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        vector_size: int = QDRANT_VECTOR_DIMENSION,
        distance: models.Distance = models.Distance.COSINE,
    ) -> bool:
        """
        Idempotently creates the Qdrant vector collection if it does not already exist.
        """
        if not qdrant_manager.client:
            logger.info("Qdrant client offline; mock collection ensured", collection=collection_name)
            return True

        try:
            exists = await self.collection_exists(collection_name)
            if not exists:
                await qdrant_manager.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=distance,
                    ),
                )
                logger.info("Created Qdrant collection successfully", collection=collection_name, size=vector_size)
            return True
        except Exception as e:
            logger.warning("Failed ensuring Qdrant collection", collection=collection_name, error=str(e))
            return False

    async def list_collections(self) -> list[QdrantCollectionInfo]:
        if not qdrant_manager.client:
            return [
                QdrantCollectionInfo(
                    name=DEFAULT_COLLECTION_NAME,
                    status="mock_active",
                    vectors_count=42,
                    vector_size=QDRANT_VECTOR_DIMENSION,
                    distance="Cosine",
                )
            ]

        try:
            res = await qdrant_manager.client.get_collections()
            infos = []
            for col in res.collections:
                infos.append(
                    QdrantCollectionInfo(
                        name=col.name,
                        status="active",
                        vectors_count=0,
                        vector_size=QDRANT_VECTOR_DIMENSION,
                        distance="Cosine",
                    )
                )
            return infos
        except Exception as e:
            logger.warning("Error listing Qdrant collections", error=str(e))
            return [
                QdrantCollectionInfo(
                    name=DEFAULT_COLLECTION_NAME,
                    status="offline_fallback",
                    vectors_count=0,
                    vector_size=QDRANT_VECTOR_DIMENSION,
                    distance="Cosine",
                )
            ]

qdrant_collection_manager = QdrantCollectionManager()
