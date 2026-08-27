import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Application Config
    NEXORA_ENV: str = "development"
    NEXORA_LOG_LEVEL: str = "INFO"
    NEXORA_SECRET_KEY: str = "nexora-super-secret-jwt-signing-key-change-in-production-min-32-chars!"

    # PostgreSQL Database
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "nexora_db"
    POSTGRES_USER: str = "nexora_admin"
    POSTGRES_PASSWORD: str = "nexora_secure_password_2026"

    # Redis Cache & Queue
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = "nexora_redis_password_2026"
    REDIS_DB: int = 0

    # Qdrant Vector DB
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: str = "nexora_qdrant_api_key_2026"

    # Redis Cache & TTL Settings (Seconds)
    CACHE_DEFAULT_TTL: int = 300           # 5 minutes
    CACHE_USER_PROFILE_TTL: int = 600      # 10 minutes
    CACHE_TELEMETRY_TTL: int = 5           # 5 seconds

    # Distributed Rate Limiting (Requests / Window)
    RATE_LIMIT_LOGIN_PER_MIN: int = 5
    RATE_LIMIT_REGISTER_PER_MIN: int = 5
    RATE_LIMIT_REFRESH_PER_MIN: int = 20
    RATE_LIMIT_API_PER_MIN: int = 60
    RATE_LIMIT_VECTOR_PER_MIN: int = 30

    # Document Ingestion Settings
    MAX_UPLOAD_SIZE_BYTES: int = 20 * 1024 * 1024  # 20 MB
    ALLOWED_DOCUMENT_EXTENSIONS: List[str] = [".txt", ".pdf", ".docx", ".csv"]

    # CORS Origins
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://localhost:5173",
        "http://localhost:8000",
        "*"
    ]

    @property
    def postgres_async_url(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def redis_url(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
