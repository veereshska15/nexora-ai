from qdrant_client import AsyncQdrantClient
from core.config import settings
from core.logging import logger

class QdrantClientManager:
    def __init__(self):
        self.client: AsyncQdrantClient | None = None

    async def connect(self):
        try:
            url = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"
            self.client = AsyncQdrantClient(
                url=url,
                api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None,
                timeout=5.0,
            )
            # Verify connectivity
            await self.client.get_collections()
            logger.info("Connected to Qdrant vector database", url=url)
        except Exception as e:
            logger.warning("Qdrant connection failed (running in offline fallback mode)", error=str(e))

    async def disconnect(self):
        if self.client:
            await self.client.close()
            logger.info("Closed Qdrant client connection")

    async def is_healthy(self) -> bool:
        if not self.client:
            return False
        try:
            await self.client.get_collections()
            return True
        except Exception:
            return False

qdrant_manager = QdrantClientManager()
