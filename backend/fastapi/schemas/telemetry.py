from pydantic import BaseModel, Field

class TelemetrySnapshot(BaseModel):
    cpu_percent: float = Field(..., example=42.5)
    memory_percent: float = Field(..., example=61.2)
    gpu_percent: float | None = Field(73.8, example=73.8)
    latency_ms: int = Field(..., example=41)
    active_connections: int = Field(..., example=1)
    active_rag_documents: int = Field(..., example=428)
    active_mcp_tools: int = Field(..., example=17)
    security_threats: int = Field(..., example=0)
    is_mock_data: bool = Field(True, description="Indicates development/mock telemetry vs production hardware telemetry")
