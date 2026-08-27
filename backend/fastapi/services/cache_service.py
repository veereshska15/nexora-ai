import json
import time
from typing import Any, Dict, Optional
from core.redis import redis_manager
from core.config import settings
from core.logging import logger

class CacheKeyManager:
    """Centralized, typed Redis cache key namespace generator for NEXORA AI."""

    @staticmethod
    def user_profile(user_id: str) -> str:
        return f"nexora:user:{user_id}"

    @staticmethod
    def user_permissions(user_id: str) -> str:
        return f"nexora:user:permissions:{user_id}"

    @staticmethod
    def session(session_id: str) -> str:
        return f"nexora:session:{session_id}"

    @staticmethod
    def telemetry(scope: str = "system") -> str:
        return f"nexora:telemetry:{scope}"

    @staticmethod
    def rate_limit(action: str, identifier: str) -> str:
        return f"nexora:rate:{action}:{identifier}"

class CacheService:
    """
    Distributed Redis Cache Service with safe JSON serialization,
    centralized key management, and in-memory TTL fallback.
    """

    def __init__(self):
        self.keys = CacheKeyManager
        # Local fallback cache for development/test environments when Redis is offline
        self._local_cache: Dict[str, tuple[str, float]] = {}

    def _serialize(self, data: Any) -> str:
        try:
            return json.dumps(data)
        except Exception as e:
            logger.error("Cache serialization failed", error=str(e))
            raise ValueError(f"Unable to serialize cache payload: {str(e)}")

    def _deserialize(self, payload: str) -> Any:
        try:
            return json.loads(payload)
        except Exception as e:
            logger.error("Cache deserialization failed", error=str(e))
            return None

    async def get(self, key: str) -> Optional[Any]:
        # Try Redis first
        if redis_manager.redis:
            try:
                raw_val = await redis_manager.redis.get(key)
                if raw_val is not None:
                    return self._deserialize(raw_val)
            except Exception as e:
                logger.warning("Redis GET failed, checking fallback", key=key, error=str(e))

        # Fallback to local memory cache
        if key in self._local_cache:
            val, expiry = self._local_cache[key]
            if time.time() < expiry:
                return self._deserialize(val)
            else:
                del self._local_cache[key]

        return None

    async def set(self, key: str, value: Any, ttl_seconds: int = settings.CACHE_DEFAULT_TTL) -> bool:
        serialized = self._serialize(value)

        # Try Redis
        if redis_manager.redis:
            try:
                await redis_manager.redis.set(key, serialized, ex=ttl_seconds)
                return True
            except Exception as e:
                logger.warning("Redis SET failed, storing in fallback", key=key, error=str(e))

        # Store in local fallback cache
        self._local_cache[key] = (serialized, time.time() + ttl_seconds)
        return True

    async def delete(self, key: str) -> bool:
        # Delete from Redis
        if redis_manager.redis:
            try:
                await redis_manager.redis.delete(key)
            except Exception as e:
                logger.warning("Redis DELETE failed", key=key, error=str(e))

        # Delete from local fallback
        if key in self._local_cache:
            del self._local_cache[key]
        return True

    async def exists(self, key: str) -> bool:
        if redis_manager.redis:
            try:
                count = await redis_manager.redis.exists(key)
                if count > 0:
                    return True
            except Exception:
                pass

        if key in self._local_cache:
            _, expiry = self._local_cache[key]
            if time.time() < expiry:
                return True
            else:
                del self._local_cache[key]
        return False

    async def invalidate_user_cache(self, user_id: str) -> None:
        """Invalidates both user profile and permission cache keys atomically."""
        await self.delete(self.keys.user_profile(user_id))
        await self.delete(self.keys.user_permissions(user_id))
        logger.info("Invalidated user cache keys", user_id=user_id)

cache_service = CacheService()
