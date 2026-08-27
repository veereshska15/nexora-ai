from redis.asyncio import Redis
from .config import settings
from .logging import logger

class RedisManager:
    def __init__(self):
        self.redis: Redis | None = None

    async def connect(self):
        try:
            self.redis = Redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis.ping()
            logger.info("Connected to Redis server successfully")
        except Exception as e:
            logger.warning("Redis connection failed (will run in fallback mode if offline)", error=str(e))

    async def disconnect(self):
        if self.redis:
            await self.redis.close()
            logger.info("Closed Redis connection")

    async def is_healthy(self) -> bool:
        if not self.redis:
            return False
        try:
            return await self.redis.ping()
        except Exception:
            return False

redis_manager = RedisManager()
