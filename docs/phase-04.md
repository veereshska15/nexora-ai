# NEXORA AI — Phase 04: FastAPI Core & Real-Time Engine Architecture

## 1. Overview

Phase 04 establishes the production-grade Python FastAPI microservice backend for **NEXORA AI**.

It implements a modular, async-first architecture providing REST APIs under `/api/v1/`, real-time bidirectional WebSocket streaming under `/api/v1/ws/chat`, SQLAlchemy 2.x async ORM database models for PostgreSQL + `pgvector`, an async Redis connection manager, Pydantic v2 settings & schemas, structured JSON logging, and correlation ID middleware.

```
Flutter Client (Mobile / Web / Desktop)
       │
       ├─── HTTP/2 REST API (/api/v1/health, /system, /chat, /telemetry) ───┐
       │                                                                      ▼
       └─── WSS Bidirectional Streaming (/api/v1/ws/chat) ─────────► [ FastAPI Gateway ]
                                                                             │
                                              ┌──────────────────────────────┴──────────────────────────────┐
                                              ▼                                                             ▼
                                [ Async SQLAlchemy 2.x ]                                         [ Async Redis Client ]
                                              │                                                             │
                                              ▼                                                             ▼
                                    PostgreSQL + pgvector                                          Redis Session Store
```

---

## 2. Directory Architecture

```
backend/fastapi/
├── core/
│   ├── config.py             # Strongly typed Pydantic Settings (.env configuration)
│   ├── database.py           # SQLAlchemy 2.x async engine & session factory
│   ├── logging.py            # Structured JSON structlog logger
│   └── redis.py              # Async Redis client manager
├── middleware/
│   ├── error_handler.py      # Global unhandled exception JSON handler
│   └── request_id.py         # Correlation ID (X-Request-ID) & latency timing middleware
├── models/                   # SQLAlchemy 2.x async ORM Entities
│   ├── base.py
│   ├── user.py               # UserModel
│   ├── session.py            # SessionModel
│   ├── conversation.py       # ConversationModel
│   └── message.py            # MessageModel
├── repositories/
│   └── conversation_repository.py
├── routers/                  # Versioned API Endpoints (/api/v1/)
│   ├── chat.py               # REST Chat (/api/v1/chat)
│   ├── health.py             # Liveness (/health/live) & Readiness (/health/ready)
│   ├── session.py            # Session State (/api/v1/session)
│   ├── system.py             # System Info (/api/v1/system)
│   ├── telemetry.py          # Telemetry Metrics (/api/v1/telemetry)
│   └── ws_router.py          # WebSocket Streaming (/api/v1/ws/chat)
├── schemas/                  # Pydantic v2 DTOs
│   ├── chat.py               # ChatMessageRequest / Response
│   ├── health.py             # HealthCheckResponse / ReadinessResponse
│   ├── telemetry.py          # TelemetrySnapshot
│   └── ws_events.py          # WSClientEvent / WSServerEvent / AIStateEnum
├── services/
│   ├── chat_service.py       # Development mock chat response engine
│   ├── health_service.py     # PostgreSQL, Redis & Vector DB health inspector
│   └── telemetry_service.py  # System hardware telemetry generator
├── tests/                    # PyTest Unit & Integration Test Suite
│   ├── conftest.py
│   ├── test_chat.py
│   ├── test_health.py
│   ├── test_telemetry.py
│   └── test_websocket.py
├── websocket/
│   ├── connection_manager.py # Active WebSocket connection registry & broadcast manager
│   └── ws_chat_handler.py    # WebSocket event parser & Riverpod AIState streamer
└── main.py                   # FastAPI app initialization, middleware & router mounting
```

---

## 3. WebSocket Event Matrix & Riverpod AIState Mapping

The WebSocket server (`/api/v1/ws/chat`) translates client interactions directly into Riverpod `AIState` events consumable by the Flutter client:

| Event Type | Direction | Payload Description | Riverpod `AIState` |
| :--- | :--- | :--- | :--- |
| `connection_init` | Client $\rightarrow$ Server | Initial handshake initialization | `idle` |
| `connection_ready` | Server $\rightarrow$ Client | Acknowledges socket connection | `idle` |
| `user_message` | Client $\rightarrow$ Server | User text prompt payload | Triggers `thinking` $\rightarrow$ `processing` $\rightarrow$ `speaking` |
| `voice_start` | Client $\rightarrow$ Server | VAD speech start detection | `listening` |
| `voice_end` | Client $\rightarrow$ Server | VAD speech stop signal | Triggers `thinking` $\rightarrow$ `speaking` $\rightarrow$ `idle` |
| `ai_state` | Server $\rightarrow$ Client | Explicit AI State push notification | `idle` / `listening` / `thinking` / `processing` / `speaking` / `success` / `error` |
| `token` | Server $\rightarrow$ Client | Real-time streaming response chunk | `speaking` |
| `message_complete` | Server $\rightarrow$ Client | Final response completion signal | `success` $\rightarrow$ `idle` |
| `ping` / `pong` | Both | Socket heartbeat keeping connection alive | N/A |

---

## 4. API Endpoints Reference

- `GET /api/v1/health` & `GET /api/v1/health/live`: Returns liveness status (`status: "healthy"`).
- `GET /api/v1/health/ready`: Inspects live PostgreSQL and Redis connections (`status: "ready"`).
- `GET /api/v1/system`: Returns active platform capabilities matrix.
- `POST /api/v1/chat/message`: Posts a REST chat prompt and receives a formatted `ChatMessageResponse`.
- `GET /api/v1/telemetry`: Serves hardware and AI system telemetry snapshot for Developer Mode.
- `WSS /api/v1/ws/chat`: Real-time bidirectional WebSocket event stream.

---

## 5. Security & Logging Foundation

1. **Correlation ID Tracking**: `RequestIDMiddleware` generates a unique `X-Request-ID` UUID for every HTTP request, enabling end-to-end trace correlation across backend logs.
2. **Structured JSON Logs**: Uses `structlog` outputting ISO timestamps, request IDs, endpoints, latency, and status codes.
3. **CORS Configuration**: Restricted to configured origins (`CORS_ORIGINS`).
4. **Secret Protection**: All passwords and API keys are loaded via Pydantic `BaseSettings` from `.env`.

---

## 6. How to Run & Test

1. Activate virtual environment:
   ```bash
   cd backend/fastapi
   .\venv\Scripts\Activate.ps1
   ```
2. Start FastAPI Server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```
3. Interactive API Docs: Open [`http://localhost:8000/docs`](http://localhost:8000/docs) in browser.
4. Run PyTest Test Suite:
   ```bash
   pytest
   ```
