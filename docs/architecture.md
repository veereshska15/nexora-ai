# NEXORA AI — Technical Architecture & Microservices Blueprint

## System Overview

NEXORA AI utilizes a dual-backend, microservice-oriented design connecting a Flutter cross-platform front-end to high-speed Python (FastAPI) AI engines and Java (Spring Boot) business management services.

```
+-----------------------------------------------------------------------------+
|                          FLUTTER CLIENT APPLICATION                         |
|  [ Nature Mode: 3D Neural Forest & Avatar ]   [ Developer Mode Telemetry ]  |
+--------------------------------------┬────────────────────────────────------+
                                       │ HTTPS / WebSockets / WebRTC
                                       ▼
+-----------------------------------------------------------------------------+
|                       SECURITY GATEWAY & ROUTE GUARD                        |
|   Prompt Injection Filter  |  DLP Payload Inspection  |  Token Auth         |
+──────────────────────────────────────┬──────────────────────────────────────+
                                       │
           ┌───────────────────────────┴───────────────────────────┐
           ▼                                                       ▼
+---------------------------------------+  +----------------------------------+
|     FASTAPI AI MICROSERVICE ENGINE    |  |  SPRING BOOT ENTERPRISE SERVICE  |
| - LangGraph Stateful Orchestration    |  | - Identity & JWT Token Management|
| - LangChain Document Processing & RAG |  | - Payment & Subscription Billing |
| - FastMCP Tool Execution Registry     |  | - Role-Based Access Control (RBAC)|
| - Custom PyTorch 3D-CNN Inference     |  | - Audit Trail & Compliance Logs  |
| - Whisper VAD STT & Audio TTS Engine  |  +────────────────┬─────────────────+
+──────────────────┬────────────────────+                   │
                   │                                        │
                   ▼                                        ▼
+-----------------------------------------------------------------------------+
|                         PERSISTENCE & DATA STORAGE                          |
|  PostgreSQL (pgvector)  │  Redis Cache  │  Qdrant Vector DB  │  S3 Storage    |
+-----------------------------------------------------------------------------+
```

## Data & Communication Protocols

1. **REST APIs (HTTP/2)**: Used for stateless operations (Authentication, Payment checkout, Subscription management, Analytics dashboard).
2. **WebSockets (WSS)**: Used for low-latency bidirectional real-time communication (Live AI Chat, Voice activity streams, 3D Avatar state synchronization).
3. **gRPC / Internal HTTP**: Used for fast service-to-service communication between Spring Boot business controllers and Python AI inference nodes.
4. **FastMCP Protocol**: Standardized JSON-RPC tool calling protocol allowing LLMs to interact with internal database handlers, location APIs, and external systems securely.
