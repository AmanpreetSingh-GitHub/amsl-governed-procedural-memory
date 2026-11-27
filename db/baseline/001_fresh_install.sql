-- Fresh AMSL install through migration 007. Use PostgreSQL 17 with pgvector installed.
-- Provision the agent_memory login first and run as that role with extension creation privileges,
-- or have an administrator install the vector extension in the target database before running this file.
BEGIN;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_catalog.pg_namespace
        WHERE nspname IN ('scientific', 'agent_memory')
    ) THEN
        RAISE EXCEPTION 'AMSL fresh-install baseline requires both application schemas to be absent';
    END IF;
END;
$$;

CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA scientific AUTHORIZATION agent_memory;
CREATE SCHEMA agent_memory AUTHORIZATION agent_memory;

CREATE TABLE scientific.sources (
    source_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    title TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_url TEXT,
    version_label TEXT,
    published_at DATE,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    canonical_key TEXT,
    content_sha256 TEXT,
    document_json JSONB,
    CONSTRAINT ck_scientific_source_sha256
        CHECK (content_sha256 IS NULL OR content_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck_scientific_source_json_object
        CHECK (document_json IS NULL OR jsonb_typeof(document_json) = 'object'),
    CONSTRAINT ck_scientific_source_identity_complete
        CHECK (
            (canonical_key IS NULL AND content_sha256 IS NULL AND document_json IS NULL)
            OR
            (canonical_key IS NOT NULL AND btrim(canonical_key) <> ''
             AND version_label IS NOT NULL AND btrim(version_label) <> ''
             AND content_sha256 IS NOT NULL AND document_json IS NOT NULL)
        )
);

CREATE TABLE scientific.knowledge_entries (
    entry_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_id BIGINT NOT NULL
        REFERENCES scientific.sources(source_id) ON DELETE RESTRICT,
    topic TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE scientific.evidence_releases (
    release_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    stage INTEGER NOT NULL CHECK (stage >= 0),
    release_status TEXT NOT NULL,
    released_at_utc TIMESTAMPTZ NOT NULL,
    agent_execution_verified BOOLEAN NOT NULL,
    access_policy TEXT NOT NULL,
    manifest_json JSONB NOT NULL,
    receipt_json JSONB NOT NULL,
    case_inventory_json JSONB NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    receipt_sha256 TEXT NOT NULL,
    case_inventory_sha256 TEXT NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_evidence_release UNIQUE (experiment_id, case_id, stage),
    CONSTRAINT ck_evidence_release_case CHECK (case_id ~ '^CA[0-9]+$'),
    CONSTRAINT ck_evidence_release_nonempty CHECK (
        btrim(experiment_id) <> '' AND btrim(release_status) <> ''
        AND btrim(access_policy) <> ''
    ),
    CONSTRAINT ck_evidence_release_json CHECK (
        jsonb_typeof(manifest_json) = 'object'
        AND jsonb_typeof(receipt_json) = 'object'
        AND jsonb_typeof(case_inventory_json) = 'object'
    ),
    CONSTRAINT ck_evidence_release_sha256 CHECK (
        manifest_sha256 ~ '^[0-9a-f]{64}$'
        AND receipt_sha256 ~ '^[0-9a-f]{64}$'
        AND case_inventory_sha256 ~ '^[0-9a-f]{64}$'
    )
);

CREATE TABLE scientific.release_evidence (
    release_evidence_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    release_id BIGINT NOT NULL
        REFERENCES scientific.evidence_releases(release_id) ON DELETE RESTRICT,
    source_id BIGINT NOT NULL
        REFERENCES scientific.sources(source_id) ON DELETE RESTRICT,
    snapshot_path TEXT NOT NULL,
    original_source_path TEXT NOT NULL,
    declared_sha256 TEXT NOT NULL,
    CONSTRAINT uq_release_snapshot UNIQUE (release_id, snapshot_path),
    CONSTRAINT uq_release_canonical_source UNIQUE (release_id, source_id),
    CONSTRAINT ck_release_evidence_sha256
        CHECK (declared_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck_release_evidence_paths CHECK (
        btrim(snapshot_path) <> '' AND btrim(original_source_path) <> ''
    )
);

CREATE TABLE scientific.source_embeddings (
    embedding_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_id BIGINT NOT NULL
        REFERENCES scientific.sources(source_id) ON DELETE RESTRICT,
    json_path TEXT[] NOT NULL
        CHECK (cardinality(json_path) BETWEEN 1 AND 8
               AND array_position(json_path, '') IS NULL),
    embedding_model TEXT NOT NULL
        CHECK (embedding_model = 'embeddinggemma:latest'),
    embedding vector(768) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_source_embedding_path
        UNIQUE (source_id, embedding_model, json_path)
);

CREATE TABLE agent_memory.episodes (
    episode_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    title TEXT NOT NULL,
    objective TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'IN_PROGRESS'
        CHECK (status IN ('IN_PROGRESS', 'COMPLETED', 'FAILED')),
    summary TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    run_id UUID,
    experiment_id TEXT,
    case_id TEXT,
    stage INTEGER,
    release_id BIGINT
        REFERENCES scientific.evidence_releases(release_id) ON DELETE RESTRICT,
    record_sha256 TEXT,
    run_record JSONB,
    trace_level TEXT,
    embedding_model TEXT,
    episode_embedding vector(768),
    CONSTRAINT chk_episode_dates
        CHECK (completed_at IS NULL OR completed_at >= started_at),
    CONSTRAINT uq_episodes_run_id UNIQUE (run_id),
    CONSTRAINT chk_episode_run_complete CHECK (
        run_id IS NULL OR (
            experiment_id IS NOT NULL AND btrim(experiment_id) <> ''
            AND case_id ~ '^CA[0-9]+$' AND stage >= 0
            AND release_id IS NOT NULL
            AND record_sha256 IS NOT NULL
            AND record_sha256 ~ '^[0-9a-f]{64}$'
            AND run_record IS NOT NULL
            AND jsonb_typeof(run_record) = 'object'
            AND run_record->>'run_id' = run_id::text
            AND run_record->>'case_id' = case_id
            AND run_record->>'experiment_id' = experiment_id
            AND (run_record->>'stage')::integer = stage
            AND trace_level IS NOT NULL
            AND trace_level = 'STAGE_OUTPUTS_ONLY'
            AND (embedding_model IS NULL) = (episode_embedding IS NULL)
        )
    ),
    CONSTRAINT chk_episode_embedding_pair CHECK (
        (episode_embedding IS NULL AND embedding_model IS NULL)
        OR (episode_embedding IS NOT NULL AND embedding_model = 'embeddinggemma:latest')
    )
);

CREATE TABLE agent_memory.memories (
    memory_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_episode_id BIGINT
        REFERENCES agent_memory.episodes(episode_id) ON DELETE RESTRICT,
    memory_type TEXT NOT NULL
        CHECK (memory_type IN ('FACT', 'LESSON', 'PROCEDURE', 'REFLECTION')),
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'CANDIDATE'
        CHECK (status IN ('CANDIDATE', 'VALIDATED', 'REJECTED', 'RETIRED')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    active_version_id BIGINT,
    CONSTRAINT chk_memories_active_only_if_validated
        CHECK ((status = 'VALIDATED' AND active_version_id IS NOT NULL)
            OR (status <> 'VALIDATED' AND active_version_id IS NULL))
);

CREATE TABLE agent_memory.memory_versions (
    version_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    memory_id BIGINT NOT NULL
        REFERENCES agent_memory.memories(memory_id) ON DELETE RESTRICT,
    version_number INTEGER NOT NULL CHECK (version_number > 0),
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    change_reason TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_memory_version UNIQUE (memory_id, version_number),
    CONSTRAINT uq_memory_version_owner UNIQUE (memory_id, version_id)
);

-- Add the reverse link after both tables exist.
ALTER TABLE agent_memory.memories
    ADD CONSTRAINT fk_memories_active_version_owner
        FOREIGN KEY (memory_id, active_version_id)
        REFERENCES agent_memory.memory_versions(memory_id, version_id) ON DELETE RESTRICT;

CREATE TABLE agent_memory.validations (
    validation_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    version_id BIGINT NOT NULL
        REFERENCES agent_memory.memory_versions(version_id) ON DELETE RESTRICT,
    validation_type TEXT NOT NULL
        CHECK (validation_type IN (
            'SOURCE_CHECK','CONSISTENCY_CHECK','EXPERT_REVIEW','OPERATOR_DECISION'
        )),
    outcome TEXT NOT NULL
        CHECK (outcome IN ('PASS', 'FAIL', 'INCONCLUSIVE')),
    rationale TEXT NOT NULL,
    evidence_notes TEXT,
    validator TEXT NOT NULL,
    validated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE agent_memory.embeddings (
    embedding_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    version_id BIGINT NOT NULL
        REFERENCES agent_memory.memory_versions(version_id) ON DELETE RESTRICT,
    embedding_model TEXT NOT NULL,
    embedding vector NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_version_embedding_model UNIQUE (version_id, embedding_model)
);

CREATE TABLE agent_memory.memory_evidence (
    evidence_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    version_id BIGINT NOT NULL
        REFERENCES agent_memory.memory_versions(version_id) ON DELETE RESTRICT,
    knowledge_entry_id BIGINT NOT NULL
        REFERENCES scientific.knowledge_entries(entry_id) ON DELETE RESTRICT,
    evidence_relation TEXT NOT NULL
        CHECK (evidence_relation IN ('SUPPORTS', 'CONTRADICTS', 'CONTEXT')),
    rationale TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_memory_evidence
        UNIQUE (version_id, knowledge_entry_id, evidence_relation)
);

CREATE INDEX idx_knowledge_source
    ON scientific.knowledge_entries(source_id);

CREATE INDEX idx_memories_episode
    ON agent_memory.memories(source_episode_id);

CREATE INDEX idx_validations_version
    ON agent_memory.validations(version_id);

CREATE INDEX idx_memory_evidence_version
    ON agent_memory.memory_evidence(version_id);

CREATE UNIQUE INDEX uq_scientific_canonical_source
    ON scientific.sources (canonical_key, version_label, content_sha256)
    WHERE canonical_key IS NOT NULL;

CREATE INDEX idx_evidence_releases_case_stage
    ON scientific.evidence_releases(case_id, stage);

CREATE INDEX idx_release_evidence_source
    ON scientific.release_evidence(source_id);

CREATE INDEX idx_episodes_case_stage
    ON agent_memory.episodes(case_id, stage)
    WHERE run_id IS NOT NULL;

COMMIT;
