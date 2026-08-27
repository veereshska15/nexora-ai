# NEXORA AI — Phase 05: Java Spring Boot Enterprise Core & Security Architecture

## 1. Overview

Phase 05 establishes the enterprise business, identity, role-based access control (RBAC), and security audit layer for **NEXORA AI** using **Java 21** and **Spring Boot 3.2.3**.

```
                          ┌────────────────────────────────────────────────────────┐
                          │                FLUTTER MULTIPLATFORM CLIENT            │
                          └───────────┬────────────────────────────────┬───────────┘
                                      │                                │
                 HTTPS REST (Auth / User / Audit)           WebSocket (AI Chat Stream)
                                      │                                │
                                      ▼                                ▼
                 ┌───────────────────────────┐           ┌───────────────────────────┐
                 │  JAVA SPRING BOOT 3 API   │           │   PYTHON FASTAPI ENGINE   │
                 │  - JWT Authentication     │           │  - LangGraph & AI DAG     │
                 │  - BCrypt Password Hash   │           │  - FastMCP Tool Registry  │
                 │  - RBAC & Permissions     │           │  - PyTorch 3D-CNN         │
                 │  - Audit Logging          │           │  - RAG Vector Index       │
                 └─────────────┬─────────────┘           └─────────────┬─────────────┘
                               │                                       │
                               └───────────────────┬───────────────────┘
                                                   ▼
                                 ┌───────────────────────────────────┐
                                 │       PostgreSQL 16 + pgvector    │
                                 └───────────────────────────────────┘
```

---

## 2. Directory Architecture & Package Structure

```
backend/springboot/src/main/java/com/nexora/
├── config/
│   ├── JwtConfig.java                 # Config properties loading secret & expiration
│   └── SecurityConfig.java              # Spring Security 6 filter chain & CORS
├── controller/
│   ├── AdminController.java           # /api/v1/admin (User management, role assignment)
│   ├── AuditController.java           # /api/v1/audit (Audit logs & security events)
│   ├── AuthController.java            # /api/v1/auth (Register, login, refresh, logout)
│   ├── SessionController.java         # /api/v1/sessions (Active sessions & revocation)
│   └── UserController.java            # /api/v1/users (User profile endpoints)
├── dto/
│   ├── request/                       # RegisterRequest, LoginRequest, RefreshTokenRequest
│   └── response/                      # ApiResponse, AuthResponse, UserResponse, SessionResponse
├── entity/
│   ├── AuditLog.java                  # Audit log entity
│   ├── Permission.java                # Fine-grained permission entity
│   ├── Role.java                      # RBAC Role entity
│   ├── RoleName.java                  # Role enum
│   ├── SecurityEvent.java             # Security threat event entity
│   ├── User.java                      # Core User entity
│   └── UserSession.java               # Refresh token session entity
├── exception/
│   ├── AuthException.java
│   ├── GlobalExceptionHandler.java    # @RestControllerAdvice returning JSON errors
│   └── ResourceNotFoundException.java
├── repository/
│   ├── AuditLogRepository.java
│   ├── PermissionRepository.java
│   ├── RoleRepository.java
│   ├── SecurityEventRepository.java
│   ├── SessionRepository.java
│   └── UserRepository.java
├── security/
│   ├── CustomUserDetailsService.java  # UserDetailsService implementation
│   ├── JwtAuthenticationEntryPoint.java# 401 Unauthorized JSON error handler
│   ├── JwtAuthenticationFilter.java  # OncePerRequestFilter extracting Bearer JWT
│   ├── JwtService.java                # Token signing & validation
│   └── UserPrincipal.java             # UserDetails mapping
├── service/
│   ├── AuditService.java              # Security & audit logging
│   ├── AuthService.java              # Auth workflows & BCrypt hashing
│   ├── SessionService.java            # Active session tracking & revocation
│   └── UserService.java               # Profile management & role updates
└── NexoraApplication.java             # Spring Boot main class
```

---

## 3. Role-Based Access Control (RBAC) & Permission Matrix

| Role | Access Level & Description | Key Permissions Granted |
| :--- | :--- | :--- |
| `ROLE_USER` | Standard registered user | `CHAT_READ`, `CHAT_WRITE`, `DOCUMENT_READ`, `VISION_USE`, `LOCATION_USE` |
| `ROLE_STUDENT` | Academic student plan | `CHAT_READ`, `CHAT_WRITE`, `DOCUMENT_READ`, `DOCUMENT_UPLOAD`, `ML_USE`, `VISION_USE` |
| `ROLE_PREMIUM_USER`| Pro subscriber plan | `CHAT_READ`, `CHAT_WRITE`, `DOCUMENT_READ`, `DOCUMENT_UPLOAD`, `VISION_USE`, `ML_USE`, `MCP_USE`, `LOCATION_USE`, `PAYMENT_READ` |
| `ROLE_MODERATOR` | Content & audit reviewer | `CHAT_READ`, `DOCUMENT_READ`, `AUDIT_READ` |
| `ROLE_DEVELOPER` | Developer telemetry mode | `CHAT_READ`, `CHAT_WRITE`, `MCP_USE`, `ANALYTICS_READ`, `ADMIN_READ` |
| `ROLE_ADMIN` | System administrator | Full administrative privileges (`USER_MANAGE`, `ADMIN_READ`, `ADMIN_WRITE`, `AUDIT_READ`) |
| `ROLE_SUPER_ADMIN` | Super administrator | Full root system access |

---

## 4. REST API Endpoint Mapping

| Method | Endpoint | Access Level | Description |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/auth/register` | Public | Register new user account |
| `POST` | `/api/v1/auth/login` | Public | Authenticate credentials & return JWT |
| `POST` | `/api/v1/auth/refresh` | Public | Rotate refresh token for new access token |
| `POST` | `/api/v1/auth/logout` | Protected | Invalidate current session |
| `GET` | `/api/v1/users/me` | Protected | Fetch current user profile & permissions |
| `GET` | `/api/v1/sessions/me` | Protected | View active user login sessions |
| `POST` | `/api/v1/sessions/revoke/{id}` | Protected | Revoke specific session |
| `GET` | `/api/v1/audit/logs` | `ADMIN` / `MODERATOR` | Inspect system audit trail |
| `GET` | `/api/v1/audit/security-events` | `ADMIN` | Inspect security threat events |
| `GET` | `/api/v1/admin/users` | `ADMIN` | List all system users |
| `PUT` | `/api/v1/admin/users/{id}/role` | `ADMIN` | Modify user role assignment |
| `GET` | `/actuator/health` | Public | Spring Boot Actuator health check |

---

## 5. Security & FastAPI Trust Model

FastAPI and Spring Boot share secret configuration via `NEXORA_SECRET_KEY`. When Spring Boot signs a JWT access token upon user login, FastAPI validates the token signature directly using HMAC-SHA512. This decouples AI inference from user management while guaranteeing zero-trust token verification across both microservices.

---

## 6. How to Run & Test

1. Navigate to Spring Boot directory:
   ```bash
   cd backend/springboot
   ```
2. Build and run with Maven:
   ```bash
   ./mvnw spring-boot:run
   ```
3. Run Unit Tests:
   ```bash
   ./mvnw test
   ```
