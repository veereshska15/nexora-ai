from fastapi import Request, status
from fastapi.responses import JSONResponse
from core.logging import logger

async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error("Unhandled Global Exception", request_id=request_id, error=str(exc), path=request.url.path)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later.",
            "request_id": request_id,
            "details": str(exc) if request.app.debug else None
        }
    )
