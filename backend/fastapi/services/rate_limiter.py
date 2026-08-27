import time
from typing import Dict, Tuple
from fastapi import HTTPException, Request, status
from core.redis import redis_manager
from core.config import settings
from core.logging import logger
from services.cache_service import CacheKeyManager

class RateLimiterService:
    """
    Distributed Redis Rate Limiting Service using Atomic INCR and TTL Windows.
    Includes in-memory fixed-window fallback for local/test environments.
    """

    def __init__(self):
        # Fallback in-memory tracking: {key: (count, window_start_time)}
        self._local_counters: Dict[str, Tuple[int, float]] = {}

    async def check_rate_limit(
        self,
        action: str,
        identifier: str,
        limit: int,
        window_seconds: int = 60,
    ) -> Tuple[bool, int, int]:
        """
        Evaluates whether a client request is within the allowed rate limit threshold.
        Returns: (is_allowed, remaining_requests, retry_after_seconds)
        """
        key = CacheKeyManager.rate_limit(action, identifier)

        # 1. Try Redis Atomic INCR
        if redis_manager.redis:
            try:
                # Increment counter atomically
                current_count = await redis_manager.redis.incr(key)

                # If this is the first hit in the window, set expiration
                if current_count == 1:
                    await redis_manager.redis.expire(key, window_seconds)

                ttl = await redis_manager.redis.ttl(key)
                ttl_remaining = max(1, ttl if ttl > 0 else window_seconds)

                if current_count > limit:
                    logger.warning(
                        "Rate limit exceeded",
                        action=action,
                        identifier=identifier,
                        count=current_count,
                        limit=limit,
                    )
                    return False, 0, ttl_remaining

                remaining = max(0, limit - current_count)
                return True, remaining, ttl_remaining
            except Exception as e:
                logger.warning("Redis rate limiter error, using fallback", key=key, error=str(e))

        # 2. Local In-Memory Fallback
        now = time.time()
        if key in self._local_counters:
            count, start_time = self._local_counters[key]
            elapsed = now - start_time

            if elapsed < window_seconds:
                new_count = count + 1
                self._local_counters[key] = (new_count, start_time)
                ttl_remaining = max(1, int(window_seconds - elapsed))

                if new_count > limit:
                    return False, 0, ttl_remaining
                return True, max(0, limit - new_count), ttl_remaining
            else:
                # Window expired, reset
                self._local_counters[key] = (1, now)
                return True, limit - 1, window_seconds
        else:
            self._local_counters[key] = (1, now)
            return True, limit - 1, window_seconds

rate_limiter_service = RateLimiterService()

class RateLimitDependency:
    """FastAPI route dependency enforcing endpoint-specific distributed rate limits."""

    def __init__(self, action: str, limit: int, window_seconds: int = 60):
        self.action = action
        self.limit = limit
        self.window_seconds = window_seconds

    async def __call__(self, request: Request):
        # Extract client IP or authenticated user ID
        client_ip = request.client.host if request.client else "127.0.0.1"
        auth_header = request.headers.get("Authorization")
        identifier = client_ip if not auth_header else f"auth_{hash(auth_header)}"

        is_allowed, remaining, retry_after = await rate_limiter_service.check_rate_limit(
            action=self.action,
            identifier=identifier,
            limit=self.limit,
            window_seconds=self.window_seconds,
        )

        if not is_allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": f"Too many requests for action '{self.action}'. Please retry after {retry_after} seconds.",
                    "retry_after": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )
