import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from core.logging import logger

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        start_time = time.time()
        response = await call_next(request)
        process_time_ms = int((time.time() - start_time) * 1000)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(process_time_ms)

        logger.info(
            "HTTP Request Processed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=process_time_ms
        )

        return response
