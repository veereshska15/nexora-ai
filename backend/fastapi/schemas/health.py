from pydantic import BaseModel, Field

class ComponentStatus(BaseModel):
    status: str = Field(..., example="healthy")
    message: str | None = Field(None, example="Connected successfully")

class HealthCheckResponse(BaseModel):
    status: str = Field(..., example="healthy")
    service: str = Field("NEXORA AI Engine", example="NEXORA AI Engine")
    environment: str = Field("development", example="development")
    version: str = Field("1.0.0", example="1.0.0")

class ReadinessResponse(BaseModel):
    status: str = Field(..., example="ready")
    database: ComponentStatus
    redis: ComponentStatus
    vector_db: ComponentStatus
