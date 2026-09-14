#!/usr/bin/env python
"""Generate backend/migrations/schema.sql from the SQLAlchemy models.

The emitted file is the authoritative schema: roles, extensions, tables,
constraints, indexes and grants. Regenerate it whenever a model changes so
the hand-applied SQL never drifts from the ORM.

    python scripts/generate_schema.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from sqlalchemy import create_mock_engine  # noqa: E402
from sqlalchemy.schema import CreateIndex, CreateTable, SetColumnComment  # noqa: E402

from app.models import Base  # noqa: E402,F401  (import registers all tables)

OUTPUT = REPO_ROOT / "backend" / "migrations" / "schema.sql"

# Dependency-correct creation order.
TABLE_ORDER = [
    "system_settings",
    "video_sources",
    "identities",
    "recording_sessions",
    "tracks",
    "faces",
    "face_embeddings",
    "detections",
    "motion_events",
    "security_events",
    "alerts",
    "temporary_identity_expirations",
]

HEADER = f"""-- =====================================================================
-- Sentinel - Intelligent Video Surveillance & Analysis Platform
-- PostgreSQL schema (generated from the SQLAlchemy models)
--
-- Generated: {date.today().isoformat()}
-- Generator: python scripts/generate_schema.py
--
-- HOW TO APPLY
--   1) Create the role and database (see section 1; edit the password).
--   2) Connect to the sentinel database as a superuser or the owner:
--          psql -U postgres -d sentinel -f backend/migrations/schema.sql
--   3) Put the matching URL in .env:
--          DATABASE_URL=postgresql+psycopg://sentinel:<password>@localhost:5432/sentinel
--
-- NOTES
--   * Enum-like columns are VARCHAR + CHECK constraints, so no custom
--     PostgreSQL types are created and values are easy to inspect.
--   * Face embeddings are stored as BYTEA (little-endian float32) plus a
--     dimension column; pgvector is NOT required. See section 6 if you
--     would rather use pgvector.
--   * Large media (video, face crops) live on the filesystem. Only metadata
--     and paths relative to the configured storage roots are stored here.
-- =====================================================================

"""

ROLES = """
-- ---------------------------------------------------------------------
-- 1. ROLE AND DATABASE
--    Run these as a superuser, connected to the 'postgres' database.
--    CHANGE THE PASSWORD before running.
-- ---------------------------------------------------------------------
-- CREATE ROLE sentinel WITH LOGIN PASSWORD 'change-me-now';
-- CREATE DATABASE sentinel WITH OWNER sentinel ENCODING 'UTF8';
--
-- Optional read-only role for dashboards/reporting:
-- CREATE ROLE sentinel_readonly WITH LOGIN PASSWORD 'change-me-too';
--
-- Then connect to the sentinel database and run the rest of this file.

-- ---------------------------------------------------------------------
-- 2. EXTENSIONS
-- ---------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- fast identity name search

"""

FOOTER_TEMPLATE = """
-- ---------------------------------------------------------------------
-- 4. ADDITIONAL INDEXES
--    Time-ordered reads dominate this workload, so the hot paths get
--    descending composite indexes; partial indexes keep the "what is
--    happening right now" queries small.
-- ---------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS ix_security_events_source_started_desc
    ON security_events (source_id, started_at DESC);
CREATE INDEX IF NOT EXISTS ix_security_events_open_only
    ON security_events (source_id, dedup_key) WHERE is_open;
CREATE INDEX IF NOT EXISTS ix_alerts_active_only
    ON alerts (source_id, started_at DESC) WHERE state = 'ACTIVE';
CREATE INDEX IF NOT EXISTS ix_faces_pending_review
    ON faces (detected_at DESC)
    WHERE review_status = 'PENDING' AND recognition_state = 'UNFAMILIAR';
CREATE INDEX IF NOT EXISTS ix_detections_source_time_desc
    ON detections (source_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS ix_tracks_source_last_seen_desc
    ON tracks (source_id, last_seen_at DESC);
CREATE INDEX IF NOT EXISTS ix_identities_active_temporary
    ON identities (expires_at)
    WHERE category = 'TEMPORARY' AND status = 'ACTIVE';
CREATE INDEX IF NOT EXISTS ix_identities_name_trgm
    ON identities USING gin (lower(coalesce(display_name, generated_identifier))
                             gin_trgm_ops);
CREATE INDEX IF NOT EXISTS ix_recording_sessions_active
    ON recording_sessions (source_id) WHERE status = 'ACTIVE';

-- ---------------------------------------------------------------------
-- 5. GRANTS
--    The application role needs DML on every table and USAGE on the
--    sequences behind the SERIAL/BIGSERIAL primary keys.
-- ---------------------------------------------------------------------
GRANT USAGE ON SCHEMA public TO sentinel;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO sentinel;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO sentinel;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO sentinel;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT USAGE, SELECT ON SEQUENCES TO sentinel;

-- Optional read-only role:
-- GRANT USAGE ON SCHEMA public TO sentinel_readonly;
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO sentinel_readonly;
-- ALTER DEFAULT PRIVILEGES IN SCHEMA public
--     GRANT SELECT ON TABLES TO sentinel_readonly;

-- ---------------------------------------------------------------------
-- 6. OPTIONAL: pgvector instead of BYTEA embeddings
--    The application ships a clean vector-store abstraction and performs
--    similarity search in an in-memory index, so pgvector is optional.
--    To use it, install the extension and add a shadow column:
--
--      CREATE EXTENSION IF NOT EXISTS vector;
--      ALTER TABLE face_embeddings ADD COLUMN embedding vector(128);
--      CREATE INDEX ON face_embeddings
--          USING hnsw (embedding vector_cosine_ops);
-- ---------------------------------------------------------------------

-- ---------------------------------------------------------------------
-- 7. SEED DATA
--    Surveillance starts OFF; intelligence defaults to ON so that turning
--    surveillance on is a single action.
-- ---------------------------------------------------------------------
INSERT INTO system_settings (key, value, description, created_at, updated_at)
VALUES
    ('surveillance_active', '{"value": false}'::jsonb,
     'Global surveillance switch (parent state)', now(), now()),
    ('intelligence_active', '{"value": true}'::jsonb,
     'Global intelligence switch (requires surveillance)', now(), now())
ON CONFLICT (key) DO NOTHING;

-- =====================================================================
-- End of schema
-- =====================================================================
"""


def main() -> int:
    statements: list[str] = []

    def collect(sql, *_args, **_kwargs):
        text = str(sql.compile(dialect=engine.dialect)).strip()
        if text:
            statements.append(text + ";")

    engine = create_mock_engine("postgresql+psycopg://", collect)

    tables = Base.metadata.tables
    missing = set(tables) - set(TABLE_ORDER)
    if missing:
        raise SystemExit(
            f"TABLE_ORDER in scripts/generate_schema.py is out of date; add: {sorted(missing)}"
        )

    ddl: list[str] = []
    for name in TABLE_ORDER:
        table = tables[name]
        statements.clear()
        collect(CreateTable(table))
        ddl.append(f"-- {'-' * 66}\n-- Table: {name}\n-- {'-' * 66}")
        ddl.extend(statements)
        for index in sorted(table.indexes, key=lambda i: i.name or ""):
            statements.clear()
            collect(CreateIndex(index))
            ddl.extend(s.replace("CREATE INDEX", "CREATE INDEX IF NOT EXISTS", 1)
                       for s in statements)
        ddl.append("")

    body = (
        HEADER
        + ROLES
        + "-- ---------------------------------------------------------------------\n"
        + "-- 3. TABLES\n"
        + "-- ---------------------------------------------------------------------\n\n"
        + "\n".join(ddl)
        + FOOTER_TEMPLATE
    )
    body = body.replace("CREATE TABLE ", "CREATE TABLE IF NOT EXISTS ")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(body, encoding="utf-8")
    print(f"Wrote {OUTPUT} ({len(tables)} tables, {len(body.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
