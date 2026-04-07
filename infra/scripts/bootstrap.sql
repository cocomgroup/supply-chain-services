-- =============================================================================
-- AGT-SS Database Bootstrap
-- Run once against the Aurora cluster after first deploy.
-- Connection: psql -h <aurora-endpoint> -U agt_ss_admin -d agt_ss -f bootstrap.sql
-- =============================================================================

-- ── Application user ─────────────────────────────────────────────────────────
-- Password is managed by Secrets Manager; set it here from the secret value.
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'agt_ss_app') THEN
        CREATE ROLE agt_ss_app WITH LOGIN PASSWORD 'REPLACE_FROM_SECRETS_MANAGER';
    END IF;
END
$$;

-- ── Schema ────────────────────────────────────────────────────────────────────
CREATE SCHEMA IF NOT EXISTS public;
GRANT USAGE ON SCHEMA public TO agt_ss_app;
GRANT CREATE ON SCHEMA public TO agt_ss_app;

-- ── Extensions ────────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- =============================================================================
-- workflow_checkpoints
-- Primary persistence table for WorkflowState.
-- Upserted by checkpoints/aurora.py after every node execution.
-- =============================================================================
CREATE TABLE IF NOT EXISTS workflow_checkpoints (
    workflow_id     UUID         PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_name   TEXT         NOT NULL DEFAULT '',
    status          TEXT         NOT NULL DEFAULT 'pending',
    goal            TEXT         NOT NULL,
    category        TEXT,
    payload         JSONB        NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Partial indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_wf_status
    ON workflow_checkpoints (status)
    WHERE status NOT IN ('completed', 'failed');

CREATE INDEX IF NOT EXISTS idx_wf_awaiting_human
    ON workflow_checkpoints (updated_at DESC)
    WHERE status = 'awaiting_human';

CREATE INDEX IF NOT EXISTS idx_wf_updated_at
    ON workflow_checkpoints (updated_at DESC);

-- GIN index for JSON payload queries (e.g. payload->>'status', payload->'errors')
CREATE INDEX IF NOT EXISTS idx_wf_payload
    ON workflow_checkpoints USING GIN (payload jsonb_path_ops);

-- Auto-update updated_at on every write
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS workflow_checkpoints_updated_at ON workflow_checkpoints;
CREATE TRIGGER workflow_checkpoints_updated_at
    BEFORE UPDATE ON workflow_checkpoints
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- dead_letters
-- Append-only table for failed agent tool calls (MAX_RETRIES exhausted).
-- =============================================================================
CREATE TABLE IF NOT EXISTS dead_letters (
    id              BIGSERIAL    PRIMARY KEY,
    workflow_id     UUID         REFERENCES workflow_checkpoints (workflow_id) ON DELETE CASCADE,
    agent           TEXT         NOT NULL,
    tool_call       TEXT         NOT NULL,
    error           TEXT         NOT NULL,
    attempt         SMALLINT     NOT NULL DEFAULT 3,
    payload         JSONB        NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dl_workflow_id ON dead_letters (workflow_id);
CREATE INDEX IF NOT EXISTS idx_dl_agent       ON dead_letters (agent, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_dl_created_at  ON dead_letters (created_at DESC);

-- =============================================================================
-- approved_suppliers  (ASL — Approved Supplier List)
-- Populated and maintained by SupplierMarketAgent.
-- =============================================================================
CREATE TABLE IF NOT EXISTS approved_suppliers (
    supplier_id          TEXT         PRIMARY KEY,
    name                 TEXT         NOT NULL,
    hq_country           TEXT,
    category             TEXT,
    status               TEXT         NOT NULL DEFAULT 'pending_audit'
                         CHECK (status IN ('approved', 'conditional', 'rejected', 'pending_audit')),
    financial_health     TEXT         DEFAULT 'unknown'
                         CHECK (financial_health IN ('strong', 'stable', 'weak', 'unknown')),
    qualification_score  NUMERIC(5,2) CHECK (qualification_score BETWEEN 0 AND 100),
    sustainability_score NUMERIC(5,2) CHECK (sustainability_score BETWEEN 0 AND 100),
    coc_signed           BOOLEAN      NOT NULL DEFAULT FALSE,
    strengths            JSONB        NOT NULL DEFAULT '[]',
    risks                JSONB        NOT NULL DEFAULT '[]',
    recommended_audit    BOOLEAN      NOT NULL DEFAULT TRUE,
    metadata             JSONB        NOT NULL DEFAULT '{}',
    qualified_at         TIMESTAMPTZ,
    created_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_asl_status   ON approved_suppliers (status);
CREATE INDEX IF NOT EXISTS idx_asl_category ON approved_suppliers (category);

DROP TRIGGER IF EXISTS approved_suppliers_updated_at ON approved_suppliers;
CREATE TRIGGER approved_suppliers_updated_at
    BEFORE UPDATE ON approved_suppliers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- Permissions
-- =============================================================================
GRANT SELECT, INSERT, UPDATE, DELETE ON workflow_checkpoints TO agt_ss_app;
GRANT SELECT, INSERT                  ON dead_letters          TO agt_ss_app;
GRANT SELECT, INSERT, UPDATE, DELETE  ON approved_suppliers    TO agt_ss_app;
GRANT USAGE, SELECT ON SEQUENCE dead_letters_id_seq            TO agt_ss_app;

-- =============================================================================
-- Verification
-- =============================================================================
DO $$
DECLARE
    tbl TEXT;
BEGIN
    FOREACH tbl IN ARRAY ARRAY['workflow_checkpoints', 'dead_letters', 'approved_suppliers']
    LOOP
        IF NOT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = tbl
        ) THEN
            RAISE EXCEPTION 'Table not created: %', tbl;
        END IF;
    END LOOP;
    RAISE NOTICE 'Bootstrap complete — all tables created successfully.';
END
$$;
