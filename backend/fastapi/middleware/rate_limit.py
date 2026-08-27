from services.rate_limiter import RateLimitDependency
from core.config import settings

# Pre-configured rate limiting dependencies
rate_limit_general_api = RateLimitDependency(
    action="general_api",
    limit=settings.RATE_LIMIT_API_PER_MIN,
    window_seconds=60,
)

rate_limit_vector_search = RateLimitDependency(
    action="vector_search",
    limit=settings.RATE_LIMIT_VECTOR_PER_MIN,
    window_seconds=60,
)

rate_limit_auth_login = RateLimitDependency(
    action="login",
    limit=settings.RATE_LIMIT_LOGIN_PER_MIN,
    window_seconds=60,
)
