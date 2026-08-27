-- ==============================================================================
-- NEXORA AI — V2 ENTERPRISE RBAC, SESSIONS & AUDIT SCHEMA
-- ==============================================================================

CREATE TABLE IF NOT EXISTS permissions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) UNIQUE NOT NULL,
    description VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS roles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(50) UNIQUE NOT NULL,
    description VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS role_permissions (
    role_id UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    permission_id UUID NOT NULL REFERENCES permissions(id) ON DELETE CASCADE,
    PRIMARY KEY (role_id, permission_id)
);

CREATE TABLE IF NOT EXISTS user_roles (
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role_id UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    PRIMARY KEY (user_id, role_id)
);

CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    refresh_token_hash VARCHAR(255) NOT NULL,
    ip_address VARCHAR(45),
    user_agent VARCHAR(255),
    is_revoked BOOLEAN NOT NULL DEFAULT FALSE,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    event_type VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    ip_address VARCHAR(45),
    request_id VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS security_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    event_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    details TEXT NOT NULL,
    ip_address VARCHAR(45),
    request_id VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO permissions (name, description) VALUES
    ('CHAT_READ', 'Read chat histories and messages'),
    ('CHAT_WRITE', 'Send chat messages and generate responses'),
    ('DOCUMENT_READ', 'View uploaded RAG documents'),
    ('DOCUMENT_UPLOAD', 'Upload documents for RAG vector indexing'),
    ('VISION_USE', 'Use 2D/3D-CNN computer vision gesture suite'),
    ('ML_USE', 'Access predictive ML analytics'),
    ('MCP_USE', 'Execute FastMCP tools'),
    ('LOCATION_USE', 'Access location intelligence and maps'),
    ('ANALYTICS_READ', 'View user usage analytics'),
    ('PAYMENT_READ', 'View subscription status'),
    ('ADMIN_READ', 'Read administrative dashboard telemetry'),
    ('ADMIN_WRITE', 'Modify system settings'),
    ('USER_MANAGE', 'Manage users and assign roles'),
    ('AUDIT_READ', 'Read security audit logs')
ON CONFLICT (name) DO NOTHING;

INSERT INTO roles (name, description) VALUES
    ('ROLE_USER', 'Standard registered user'),
    ('ROLE_STUDENT', 'Academic student plan user'),
    ('ROLE_PREMIUM_USER', 'Pro subscriber with unlimited RAG/Vision access'),
    ('ROLE_MODERATOR', 'Community & content moderator'),
    ('ROLE_DEVELOPER', 'Developer mode API access'),
    ('ROLE_ADMIN', 'System administrator'),
    ('ROLE_SUPER_ADMIN', 'Super administrator with full access')
ON CONFLICT (name) DO NOTHING;
