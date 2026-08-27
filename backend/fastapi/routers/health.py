from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db_session
from schemas.health import HealthCheckResponse, ReadinessResponse
from services.health_service import HealthService

router = APIRouter(prefix="/health", tags=["Health & Readiness"])

@router.get("", response_model=HealthCheckResponse)
@router.get("/live", response_model=HealthCheckResponse)
async def health_liveness():
    """Liveness probe indicating application process is running."""
    return HealthCheckResponse(
        status="healthy",
        service="NEXORA AI Engine",
        environment="development",
        version="1.0.0"
    )

@router.get("/ready", response_model=ReadinessResponse)
async def health_readiness(db: AsyncSession = Depends(get_db_session)):
    """Readiness probe checking PostgreSQL, Redis, and Vector DB connections."""
    return await HealthService.check_readiness(db_session=db)
