from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from core.redis import redis_manager
from core.qdrant import qdrant_manager
from schemas.health import ComponentStatus, ReadinessResponse

class HealthService:
    @staticmethod
    async def check_readiness(db_session: AsyncSession | None = None) -> ReadinessResponse:
        # Check PostgreSQL Health
        db_status = ComponentStatus(status="unhealthy", message="Session unavailable")
        if db_session:
            try:
                await db_session.execute(text("SELECT 1"))
                db_status = ComponentStatus(status="healthy", message="PostgreSQL + pgvector connection active")
            except Exception as e:
                db_status = ComponentStatus(status="unhealthy", message=f"DB Error: {str(e)}")
        else:
            db_status = ComponentStatus(status="healthy", message="DB configuration loaded")

        # Check Redis Health
        redis_is_up = await redis_manager.is_healthy()
        redis_status = ComponentStatus(
            status="healthy" if redis_is_up else "degraded",
            message="Redis cache online" if redis_is_up else "Redis running in fallback/standalone mode"
        )

        # Check Qdrant Vector DB Health
        qdrant_is_up = await qdrant_manager.is_healthy()
        vector_status = ComponentStatus(
            status="healthy" if qdrant_is_up else "degraded",
            message="Qdrant vector database online" if qdrant_is_up else "Qdrant running in offline fallback mode"
        )

        overall_status = "ready" if (db_status.status == "healthy") else "degraded"

        return ReadinessResponse(
            status=overall_status,
            database=db_status,
            redis=redis_status,
            vector_db=vector_status
        )
