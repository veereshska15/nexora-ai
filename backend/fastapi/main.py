from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import settings
from core.logging import setup_logging, logger
from core.redis import redis_manager
from core.qdrant import qdrant_manager
from middleware.request_id import RequestIDMiddleware
from middleware.error_handler import global_exception_handler
from routers import health, system, chat, session, telemetry, ws_router, vector, qdrant_router, nlp, documents, embeddings, retrieval, rag

# Setup structured logging
setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup sequence
    logger.info("Starting NEXORA AI FastAPI Gateway", environment=settings.NEXORA_ENV)
    await redis_manager.connect()
    await qdrant_manager.connect()
    yield
    # Shutdown sequence
    logger.info("Shutting down NEXORA AI FastAPI Gateway")
    await qdrant_manager.disconnect()
    await redis_manager.disconnect()

app = FastAPI(
    title="NEXORA AI Engine",
    description="Secure Real-Time Multimodal, Multilingual AI Microservice Platform",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Exception Handler
app.add_exception_handler(Exception, global_exception_handler)

# Custom Middlewares
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Versioned API v1 Routers
api_v1_prefix = "/api/v1"
app.include_router(health.router, prefix=api_v1_prefix)
app.include_router(system.router, prefix=api_v1_prefix)
app.include_router(chat.router, prefix=api_v1_prefix)
app.include_router(session.router, prefix=api_v1_prefix)
app.include_router(telemetry.router, prefix=api_v1_prefix)
app.include_router(ws_router.router, prefix=api_v1_prefix)
app.include_router(vector.router, prefix=api_v1_prefix)
app.include_router(qdrant_router.router, prefix=api_v1_prefix)
app.include_router(nlp.router, prefix=api_v1_prefix)
app.include_router(documents.router, prefix=api_v1_prefix)
app.include_router(embeddings.router, prefix=api_v1_prefix)
app.include_router(retrieval.router, prefix=api_v1_prefix)
app.include_router(rag.router, prefix=api_v1_prefix)

# Unversioned Root Health Alias
@app.get("/health")
async def root_health():
    return {"status": "healthy", "service": "NEXORA AI Engine"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
