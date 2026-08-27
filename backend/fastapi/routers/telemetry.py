from fastapi import APIRouter
from schemas.telemetry import TelemetrySnapshot
from services.telemetry_service import TelemetryService

router = APIRouter(prefix="/telemetry", tags=["Developer Telemetry"])

@router.get("", response_model=TelemetrySnapshot)
async def get_telemetry():
    """Retrieve current hardware & AI system telemetry."""
    return await TelemetryService.get_system_telemetry()
