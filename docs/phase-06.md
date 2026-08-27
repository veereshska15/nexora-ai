# NEXORA AI — Phase 06: Database Migrations, pgvector & Data Layer Architecture

## 1. Overview & Migration Architecture

Phase 06 establishes **Flyway** as the authoritative database migration and versioning engine for NEXORA AI, ensuring reproducible, deterministic database schema evolution across all development, staging, and production environments.

```
                          ┌────────────────────────────────────────────────────────┐
                          │               SPRING BOOT FLYWAY ENGINE                │
                          │   - Governs Schema Creation, Versioning & Evolution   │
                          │   - Enforces Idempotent, Immutable SQL Migrations     │
                          │   - Validates Hibernate Entity Mapping against DB     │
                          └───────────────────────────┬────────────────────────────┘
                                                      │
                            ┌─────────────────────────┴─────────────────────────┐
                            ▼                                                   ▼
                V1__init_core_schema.sql                           V2__init_enterprise_rbac.sql
                - PostgreSQL Extensions (vector, uuid)              - Permissions & Roles (RBAC)
                - Users, Conversations, Messages                   - User-Roles, Role-Permissions
                - Document Chunks (Dense Vectors)                  - User Sessions, Audit Logs
                                                      │
                                                      ▼
                                         V3__vector_hnsw_indexes.sql
                                         - HNSW Cosine Index on pgvector
```

---

## 2. Why Flyway Was Selected Over Hibernate Auto-DDL

1. **Production Determinism**: Hibernate `ddl-auto: update` can produce uncontrolled table mutations, drop constraints silently, or lead to schema drift between environments.
2. **Versioned Auditability**: Flyway maintains the `flyway_schema_history` table, tracking migration checksums, timestamps, execution state, and exact applied versions.
3. **Multi-Service Harmony**: Because both Spring Boot and Python FastAPI share PostgreSQL 16, a neutral, SQL-native migration runner prevents ORM conflicts between SQLAlchemy and Hibernate.

---

## 3. Migration Sequence Breakdown

### `V1__init_core_schema.sql` (Core Relational & AI Schema)
- Enables `vector` and `uuid-ossp` extensions.
- Creates `users` table with UUID primary keys and unique email index.
- Creates `conversations` and `messages` tables with cascading foreign keys for conversation history.
- Creates `document_chunks` table with `vector(1536)` embedding columns for dense vector storage.

### `V2__init_enterprise_rbac_schema.sql` (Enterprise Security & Audit)
- Creates `permissions`, `roles`, `role_permissions`, and `user_roles` tables for RBAC.
- Creates `user_sessions` table tracking hashed refresh tokens and session revocation flags.
- Creates `audit_logs` and `security_events` tables for immutable security event tracking.
- Seeds initial 7 system roles (`ROLE_USER` through `ROLE_SUPER_ADMIN`) and 14 fine-grained permissions.

### `V3__vector_hnsw_indexes.sql` (pgvector Indexing)
- Creates a dedicated Hierarchical Navigable Small World (HNSW) index on `document_chunks (embedding vector_cosine_ops)` for approximate nearest neighbor vector similarity search.

---

## 4. Hibernate Schema Validation

In [`backend/springboot/src/main/resources/application.yml`](file:///C:/Users/Administrator/.gemini/antigravity-ide/scratch/nexora-ai/backend/springboot/src/main/resources/application.yml), Hibernate DDL auto-generation is switched from `update` to `validate`:

```yaml
spring:
  jpa:
    hibernate:
      ddl-auto: validate
  flyway:
    enabled: true
    baseline-on-migrate: true
    baseline-version: 0
    locations: classpath:db/migration
    clean-disabled: true
    validate-on-migrate: true
```

---

## 5. Existing Database Baseline & Transition Strategy

- **Baseline on Migrate**: Setting `baseline-on-migrate: true` ensures that when migrating an existing database volume (previously populated by Docker entrypoint scripts), Flyway baselines the current schema at version 0 without throwing table collision errors.
- **`CREATE TABLE IF NOT EXISTS`**: All migration scripts use idempotent statements to guarantee safe execution across fresh and pre-existing databases.
- **Safety Rule**: `flyway.clean-disabled: true` strictly blocks destructive clean operations against production and development volumes.

---

## 6. Session Table Harmonization Decision

- **FastAPI / Spring Boot Discrepancy**: FastAPI previously declared a `SessionModel` mapped to `"sessions"`, whereas Spring Boot declared `UserSession` mapped to `"user_sessions"`.
- **Architectural Resolution**: Spring Boot is the single authoritative owner of session metadata in `"user_sessions"`. FastAPI's `SessionModel` is decoupled and scheduled for deprecation, ensuring zero runtime schema collision while keeping all FastAPI REST/WebSocket features functional.

---

## 7. Rollback & Forward Migration Strategy

Flyway Community Edition utilizes forward-only migrations. When changes are required, developers create sequential forward migrations (`V4__...`, `V5__...`) rather than mutating existing applied scripts.

---

## 8. pgvector Architecture & Vector Similarity Query Engine

### Retrieval Topology
```
DOCUMENT CHUNK ──► 1536-DIM EMBEDDING ──► POSTGRESQL 16 (pgvector)
                                                │
                                    ┌───────────┴───────────┐
                                    ▼                       ▼
                           HNSW COSINE INDEX       USER ISOLATION SCOPE
                       (vector_cosine_ops, <=>)    (WHERE user_id = :uid)
                                    │                       │
                                    └───────────┬───────────┘
                                                ▼
                                         TOP-K RETRIEVAL
                                   (similarity = 1.0 - distance)
```

### Key Concepts

1. **What Embeddings Are**: Dense numerical vectors (e.g. 1536 floats) capturing semantic meaning in a continuous latent space. Semantically related concepts cluster closely together.
2. **What Vector Databases Do**: Index high-dimensional points to perform fast Approximate Nearest Neighbor (ANN) search rather than scanning millions of rows sequentially.
3. **What pgvector Is**: An open-source PostgreSQL extension providing native `vector(N)` column types and vector distance operators (`<=>` cosine, `<->` L2 Euclidean, `<#>` inner product).
4. **What HNSW Is**: Hierarchical Navigable Small World graph indexing. It organizes vector nodes into multi-layer graphs, allowing logarithmic search complexity ($O(\log N)$) during top-k queries.
5. **Cosine Distance vs. Similarity Score**:
   - Cosine distance metric: $D_C(u, v) = 1 - \frac{u \cdot v}{\|u\| \|v\|} \in [0.0, 2.0]$.
   - Normalized similarity score: $S(u, v) = 1.0 - D_C(u, v) \in [0.0, 1.0]$ (clamped).
6. **User Isolation**: Every vector query strictly filters by `WHERE document_chunks.user_id = :authenticated_user_id`, preventing cross-tenant document data leaks.

---

## 9. Qdrant Architecture & Collection Management

### Dual Vector Storage Strategy

NEXORA AI leverages both **PostgreSQL + pgvector** and **Qdrant Vector Database**:

| Feature / Metric | PostgreSQL + pgvector | Qdrant Vector DB |
| :--- | :--- | :--- |
| **Primary Role** | Relational AI transactions, audit trail, structured documents | Standalone high-throughput vector index, multi-tenant collections |
| **Data Model** | Relational SQL Tables (`document_chunks`) | Vector Points with JSON Payload (`nexora_documents`) |
| **Multi-Tenancy** | Foreign Key SQL scoping (`WHERE user_id = :uid`) | Payload Filter Query (`FieldCondition(key="user_id", match=...)`) |
| **Distance Metric** | Cosine distance operator (`<=>`) | Native Cosine similarity score |
| **Best Used For** | ACID transactional consistency, relational joins | High-scale semantic retrieval, multimodal vector indexing |

### Qdrant Point & Payload Schema

```json
{
  "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "vector": [0.031, -0.012, ..., 0.045], // 1536 dimensions
  "payload": {
    "user_id": "00000000-0000-0000-0000-000000000001",
    "document_id": "system_design.pdf",
    "chunk_index": 0,
    "content": "NEXORA AI high-performance semantic retrieval engine.",
    "source": "manual_upload",
    "created_at": "2026-08-24T00:38:00Z"
  }
}
```

### Readiness Health Check Integration

The `/api/v1/health/ready` endpoint dynamically inspects all three data backends:
- **PostgreSQL**: `SELECT 1` ping verification
- **Redis**: Async ping check
- **Qdrant**: `get_collections()` connectivity check

---

## 10. Redis Architecture, Distributed Cache & Rate Limiting

### Why Redis is Used

Redis provides sub-millisecond in-memory caching and atomic operations for distributed workloads across multiple FastAPI and Spring Boot instances without putting load on PostgreSQL.

### Cache-Aside Architecture

```
Application Request
        │
        ▼
   Redis Cache ──────────── HIT ────────────► Fast Response (<2ms)
        │
      MISS
        │
        ▼
PostgreSQL Database
        │
        ▼
 Redis Cache SET (with TTL)
        │
        ▼
     Response
```

### Distributed Rate Limiting Flow

```
Client Request
      │
      ▼
Redis Atomic INCR (key: nexora:rate:{action}:{client_id})
      │
      ├──── Count <= Limit ────► Allowed (Proceed to handler)
      │
      └──── Count > Limit ─────► HTTP 429 Too Many Requests
                                (Headers: Retry-After: {seconds})
```

### Centralized Key Naming Convention

| Key Pattern | Description | Default TTL |
| :--- | :--- | :--- |
| `nexora:user:{user_id}` | Cached User Profile | 600s (10 min) |
| `nexora:user:permissions:{user_id}` | Cached RBAC Permissions | 600s (10 min) |
| `nexora:session:{session_id}` | Cached Active Session State | 120s (2 min) |
| `nexora:telemetry:{scope}` | Cached Cluster Metrics | 5s |
| `nexora:rate:{action}:{client_id}` | Distributed Rate Limit Window | 60s (1 min) |

### Redis Failure Strategy

- **Caching**: **Fail-Open with Fallback**. If Redis becomes unavailable, the system transparently executes database reads and preserves local in-memory fallback cache to ensure zero service disruption.
- **Authentication Rate Limiting**: Configured with strict fallback window counting to prevent brute-force attacks during transient network partitions.

---

## 11. Phase 06 Final Data Layer Verification Matrix

| Component | Status | Test / Validation Mechanism | Result |
| :--- | :--- | :--- | :--- |
| **PostgreSQL 16** | **PASS** | `pgvector/pgvector:pg16` schema, 1536-dim vector column, FK cascades | **VERIFIED** |
| **Flyway** | **PASS** | `V1__init_core`, `V2__init_enterprise_rbac`, `V3__vector_hnsw` | **VERIFIED** |
| **pgvector** | **PASS** | 1536-dim vector cosine distance (`<=>`) & similarity math | **VERIFIED** |
| **HNSW Index** | **PASS** | `document_chunks_embedding_hnsw_idx` on `(embedding vector_cosine_ops)` | **VERIFIED** |
| **FastAPI** | **PASS** | Async lifespan, CORS, middleware, Pydantic v2 schemas | **VERIFIED** |
| **Spring Boot** | **PASS** | JPA entities (`User`, `Role`, `UserSession`, `AuditLog`), Flyway validate | **VERIFIED** |
| **Qdrant Vector DB** | **PASS** | `nexora_documents` collection, 1536-dim, `Distance.COSINE`, payload search | **VERIFIED** |
| **Redis Cache** | **PASS** | Safe JSON serialization, `set`/`get`/`ttl`/`delete`, in-memory fallback | **VERIFIED** |
| **JWT Security** | **PASS** | HS256 stateless filter chain, subject extraction, role claims | **VERIFIED** |
| **RBAC Matrix** | **PASS** | 7 hierarchical roles (`ROLE_USER` to `ROLE_SUPER_ADMIN`), 14 permissions | **VERIFIED** |
| **User Sessions** | **PASS** | Hashed refresh tokens, session revocation check, IP & user-agent tracking | **VERIFIED** |
| **Security Audit** | **PASS** | Immutable `audit_logs` & `security_events` table models | **VERIFIED** |
| **REST APIs** | **PASS** | Versioned `/api/v1/` endpoints for chat, session, telemetry, vector, Qdrant | **VERIFIED** |
| **WebSocket** | **PASS** | Bidirectional state streaming (`ai_state`, `token`, `message_complete`) | **VERIFIED** |
| **Cache-Aside** | **PASS** | First request miss (`cached: false`), subsequent hit (`cached: true`) | **VERIFIED** |
| **Rate Limiting** | **PASS** | Atomic counter window, HTTP 429 status code, `Retry-After` header | **VERIFIED** |
| **User Isolation** | **PASS** | Enforced user ownership in Postgres (`user_id = :uid`) and Qdrant payloads | **VERIFIED** |
