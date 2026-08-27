# NEXORA AI — Master Monorepo

> **A Secure Real-Time Multimodal, Multilingual, Location-Aware, Cloud-Native Generative AI Platform**

[![Architecture](https://img.shields.io/badge/Architecture-Cloud--Native%20Microservices-blue)](docs/architecture.md)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](backend/fastapi/)
[![Java](https://img.shields.io/badge/Java-Spring%20Boot%203-red)](backend/springboot/)
[![Flutter](https://img.shields.io/badge/Flutter-Cross--Platform-cyan)](mobile/flutter_app/)
[![License](https://img.shields.io/badge/License-MIT-purple)](#)

---

## 🏛️ Monorepo Directory Structure

```
nexora-ai/
├── .github/workflows/          # Automated CI/CD build & evaluation pipelines
├── ai/                         # Python AI Microservices Core Engine
│   ├── agents/                 # Multi-agent orchestrators & delegation networks
│   ├── genai/                  # LLM integrations, prompt templates & output parsers
│   ├── langchain/              # LangChain components, memory stores & chains
│   ├── langgraph/              # LangGraph stateful graph workflows, nodes & checkpointers
│   ├── mcp/                    # FastMCP tool handlers (Low/Med/High risk)
│   ├── rag/                    # RAG document loaders, text chunkers & dense vector retrievers
│   ├── speech/                 # VAD audio streaming, STT (Whisper) & TTS (Coqui/Piper)
│   └── vision/                 # OpenCV, OCR, object detection & 3D-CNN gesture inference
├── backend/
│   ├── fastapi/                # FastAPI Gateway & AI Streaming WebSockets
│   └── springboot/             # Enterprise Java Spring Boot business services
├── database/
│   ├── migrations/             # Database schema migration scripts (Flyway / Liquibase)
│   └── schemas/                # SQL definitions for PostgreSQL, pgvector & Redis schemas
├── docs/                       # Architecture diagrams, API specs & IEEE research paper
├── infrastructure/
│   ├── docker/                 # Container Dockerfiles & multi-stage build configs
│   ├── kubernetes/             # K8s manifests, deployments, services & Helm charts
│   └── terraform/              # Infrastructure-as-code for AWS/GCP resources
├── ml/                         # Research ML & 3D Deep Learning Core
│   ├── datasets/               # Preprocessing & augmentation pipelines
│   ├── evaluation/             # Confusion matrices, accuracy & F1 score evaluation scripts
│   ├── models/                 # Custom PyTorch 3D-CNN & XGBoost models
│   └── training/               # Model training scripts with MLflow tracking
├── mcp/                        # FastMCP Protocol Tool Implementations
│   ├── analytics/              # Usage & token analytics tool handlers
│   ├── database/               # Safe SQL execution tool handlers
│   ├── location/               # Geofencing & Google Places tool handlers
│   ├── payment/                # Subscription status & checkout tool handlers
│   └── rag/                    # Vector index query tool handlers
├── mlops/                      # MLOps Infrastructure & Governance
│   ├── mlflow/                 # Model registry & artifact logging configuration
│   ├── monitoring/             # Data drift & model performance drift detectors
│   └── pipelines/              # Automated retraining workflows
├── mobile/
│   └── flutter_app/            # Flutter cross-platform mobile/desktop application
├── security/                   # Defense-in-depth Security Infrastructure
│   ├── audit/                  # Security audit event logging
│   ├── guards/                 # Prompt injection & Output Data Loss Prevention guardrails
│   └── policies/               # Fine-grained RBAC permission matrix
├── tests/                      # PyTest, JUnit 5 & Flutter integration test suites
├── web/
│   └── admin_dashboard/        # Web administrative telemetry & analytics portal
├── docker-compose.yml          # Local containerized infrastructure orchestrator
└── README.md
```

---

## 🚀 Quick Start (Local Development)

### 1. Prerequisites
- **Docker Desktop** (with Compose v2+)
- **Python 3.11+** & **Virtual Environment**
- **Java OpenJDK 21** & **Maven 3.9+**
- **Flutter SDK 3.19+**

### 2. Environment Configuration
Copy the example configuration file and fill in your keys:
```bash
cp .env.example .env
```

### 3. Launch Local Infrastructure Stack
Start PostgreSQL (`pgvector`), Redis, Qdrant Vector DB, and MLflow tracking server:
```bash
docker-compose up -d postgres redis qdrant mlflow
```

### 4. Run AI Backend Service (FastAPI)
```bash
cd backend/fastapi
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 5. Run Enterprise Business Backend (Spring Boot)
```bash
cd backend/springboot
./mvnw spring-boot:run
```

### 6. Run Flutter App
```bash
cd mobile/flutter_app
flutter pub get
flutter run -d chrome  # Or Windows / Android / iOS target
```

---

## 🔒 Security Architecture Overview

NEXORA AI adheres to zero-trust principles:
1. **Input Guardrail**: All incoming user text, audio, and visual inputs pass through an automated Prompt Injection & Harm Filter before reaching LLM / Agent layers.
2. **Tenant Isolation**: RAG vector queries enforce strict user metadata filtering at the database layer (`user_id` / `tenant_id`).
3. **MCP Tool Authorization**: Medium- and High-risk MCP tools require explicit permission scopes; High-risk operations (payments, account modifications) enforce Human-in-the-Loop verification.

---

## 📚 License
Distributed under the MIT License. See `LICENSE` for details.
