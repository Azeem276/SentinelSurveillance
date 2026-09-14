-- =====================================================================
-- Sentinel - Intelligent Video Surveillance & Analysis Platform
-- PostgreSQL schema (generated from the SQLAlchemy models)
--
-- Generated: 2026-09-14
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

-- ---------------------------------------------------------------------
-- 3. TABLES
-- ---------------------------------------------------------------------

-- ------------------------------------------------------------------
-- Table: system_settings
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS system_settings (
	key VARCHAR(64) NOT NULL, 
	value JSONB NOT NULL, 
	description TEXT, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	CONSTRAINT pk_system_settings PRIMARY KEY (key)
);

-- ------------------------------------------------------------------
-- Table: video_sources
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS video_sources (
	id SERIAL NOT NULL, 
	uid VARCHAR(64) NOT NULL, 
	name VARCHAR(128) NOT NULL, 
	type VARCHAR(16) NOT NULL, 
	uri TEXT NOT NULL, 
	location VARCHAR(255), 
	description TEXT, 
	enabled BOOLEAN NOT NULL, 
	surveillance_enabled BOOLEAN NOT NULL, 
	intelligence_enabled BOOLEAN NOT NULL, 
	recognition_enabled BOOLEAN NOT NULL, 
	recording_enabled BOOLEAN NOT NULL, 
	loop_playback BOOLEAN NOT NULL, 
	status VARCHAR(16) NOT NULL, 
	last_error TEXT, 
	last_seen_at TIMESTAMP WITH TIME ZONE, 
	proximity_a FLOAT NOT NULL, 
	proximity_b FLOAT NOT NULL, 
	recognition_threshold FLOAT, 
	detection_confidence FLOAT, 
	temporary_retention_days INTEGER, 
	alert_policy JSONB NOT NULL, 
	calibration JSONB NOT NULL, 
	display_order INTEGER NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	CONSTRAINT pk_video_sources PRIMARY KEY (id), 
	CONSTRAINT ck_video_sources_proximity_a_gt_b CHECK (proximity_a > proximity_b), 
	CONSTRAINT ck_video_sources_proximity_b_positive CHECK (proximity_b > 0), 
	CONSTRAINT ck_video_sources_recognition_threshold_range CHECK (recognition_threshold IS NULL OR (recognition_threshold > 0 AND recognition_threshold <= 1)), 
	CONSTRAINT uq_video_sources_uid UNIQUE (uid)
);
CREATE INDEX IF NOT EXISTS ix_video_sources_enabled ON video_sources (enabled);
CREATE INDEX IF NOT EXISTS ix_video_sources_status ON video_sources (status);

-- ------------------------------------------------------------------
-- Table: identities
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS identities (
	id SERIAL NOT NULL, 
	generated_identifier VARCHAR(128) NOT NULL, 
	display_name VARCHAR(128), 
	category VARCHAR(16) NOT NULL, 
	status VARCHAR(16) NOT NULL, 
	first_detected_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	last_seen_at TIMESTAMP WITH TIME ZONE, 
	expires_at TIMESTAMP WITH TIME ZONE, 
	retention_days INTEGER, 
	notes TEXT, 
	thumbnail_path VARCHAR(512), 
	source_id INTEGER, 
	extra JSONB NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	CONSTRAINT pk_identities PRIMARY KEY (id), 
	CONSTRAINT ck_identities_temporary_requires_expiry CHECK ((category <> 'TEMPORARY') OR (expires_at IS NOT NULL)), 
	CONSTRAINT uq_identities_generated_identifier UNIQUE (generated_identifier), 
	CONSTRAINT fk_identities_source_id_video_sources FOREIGN KEY(source_id) REFERENCES video_sources (id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS ix_identities_category_status ON identities (category, status);
CREATE INDEX IF NOT EXISTS ix_identities_display_name ON identities (display_name);
CREATE INDEX IF NOT EXISTS ix_identities_expires_at ON identities (expires_at);

-- ------------------------------------------------------------------
-- Table: recording_sessions
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS recording_sessions (
	id SERIAL NOT NULL, 
	source_id INTEGER NOT NULL, 
	file_path TEXT NOT NULL, 
	status VARCHAR(16) NOT NULL, 
	started_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	ended_at TIMESTAMP WITH TIME ZONE, 
	duration_seconds FLOAT, 
	frame_count INTEGER NOT NULL, 
	fps FLOAT, 
	width INTEGER, 
	height INTEGER, 
	file_size_bytes BIGINT, 
	codec VARCHAR(16), 
	error TEXT, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	CONSTRAINT pk_recording_sessions PRIMARY KEY (id), 
	CONSTRAINT fk_recording_sessions_source_id_video_sources FOREIGN KEY(source_id) REFERENCES video_sources (id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS ix_recording_sessions_source_started ON recording_sessions (source_id, started_at);
CREATE INDEX IF NOT EXISTS ix_recording_sessions_status ON recording_sessions (status);

-- ------------------------------------------------------------------
-- Table: tracks
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tracks (
	id BIGSERIAL NOT NULL, 
	source_id INTEGER NOT NULL, 
	recording_id INTEGER, 
	track_key INTEGER NOT NULL, 
	object_class VARCHAR(64) NOT NULL, 
	status VARCHAR(16) NOT NULL, 
	first_seen_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	last_seen_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	first_frame INTEGER NOT NULL, 
	last_frame INTEGER NOT NULL, 
	duration_seconds FLOAT NOT NULL, 
	detection_count INTEGER NOT NULL, 
	max_confidence FLOAT NOT NULL, 
	identity_id INTEGER, 
	recognition_state VARCHAR(32) NOT NULL, 
	recognition_confidence FLOAT, 
	proximity_zone VARCHAR(16) NOT NULL, 
	min_distance_m FLOAT, 
	last_distance_m FLOAT, 
	entered_zone_a BOOLEAN NOT NULL, 
	entered_zone_b BOOLEAN NOT NULL, 
	trajectory JSONB NOT NULL, 
	snapshot_path VARCHAR(512), 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	CONSTRAINT pk_tracks PRIMARY KEY (id), 
	CONSTRAINT fk_tracks_source_id_video_sources FOREIGN KEY(source_id) REFERENCES video_sources (id) ON DELETE CASCADE, 
	CONSTRAINT fk_tracks_recording_id_recording_sessions FOREIGN KEY(recording_id) REFERENCES recording_sessions (id) ON DELETE SET NULL, 
	CONSTRAINT fk_tracks_identity_id_identities FOREIGN KEY(identity_id) REFERENCES identities (id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS ix_tracks_identity_id ON tracks (identity_id);
CREATE INDEX IF NOT EXISTS ix_tracks_object_class ON tracks (object_class);
CREATE INDEX IF NOT EXISTS ix_tracks_source_first_seen ON tracks (source_id, first_seen_at);
CREATE INDEX IF NOT EXISTS ix_tracks_source_track_key ON tracks (source_id, track_key);
CREATE INDEX IF NOT EXISTS ix_tracks_status ON tracks (status);

-- ------------------------------------------------------------------
-- Table: faces
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS faces (
	id BIGSERIAL NOT NULL, 
	source_id INTEGER NOT NULL, 
	track_id BIGINT, 
	identity_id INTEGER, 
	recording_id INTEGER, 
	detected_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	frame_number INTEGER NOT NULL, 
	image_path VARCHAR(512), 
	bbox_x1 FLOAT NOT NULL, 
	bbox_y1 FLOAT NOT NULL, 
	bbox_x2 FLOAT NOT NULL, 
	bbox_y2 FLOAT NOT NULL, 
	detection_confidence FLOAT NOT NULL, 
	quality_score FLOAT NOT NULL, 
	blur_score FLOAT, 
	brightness FLOAT, 
	face_pixels INTEGER, 
	quality_ok BOOLEAN NOT NULL, 
	quality_reason VARCHAR(255), 
	recognition_state VARCHAR(32) NOT NULL, 
	match_score FLOAT, 
	review_status VARCHAR(16) NOT NULL, 
	reviewed_at TIMESTAMP WITH TIME ZONE, 
	CONSTRAINT pk_faces PRIMARY KEY (id), 
	CONSTRAINT fk_faces_source_id_video_sources FOREIGN KEY(source_id) REFERENCES video_sources (id) ON DELETE CASCADE, 
	CONSTRAINT fk_faces_track_id_tracks FOREIGN KEY(track_id) REFERENCES tracks (id) ON DELETE SET NULL, 
	CONSTRAINT fk_faces_identity_id_identities FOREIGN KEY(identity_id) REFERENCES identities (id) ON DELETE SET NULL, 
	CONSTRAINT fk_faces_recording_id_recording_sessions FOREIGN KEY(recording_id) REFERENCES recording_sessions (id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS ix_faces_identity_id ON faces (identity_id);
CREATE INDEX IF NOT EXISTS ix_faces_recognition_state ON faces (recognition_state);
CREATE INDEX IF NOT EXISTS ix_faces_review_status ON faces (review_status);
CREATE INDEX IF NOT EXISTS ix_faces_source_detected ON faces (source_id, detected_at);
CREATE INDEX IF NOT EXISTS ix_faces_track_id ON faces (track_id);

-- ------------------------------------------------------------------
-- Table: face_embeddings
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS face_embeddings (
	id BIGSERIAL NOT NULL, 
	identity_id INTEGER NOT NULL, 
	face_id BIGINT, 
	vector BYTEA NOT NULL, 
	dim INTEGER NOT NULL, 
	model_name VARCHAR(64) NOT NULL, 
	origin VARCHAR(32) NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	CONSTRAINT pk_face_embeddings PRIMARY KEY (id), 
	CONSTRAINT uq_face_embeddings_face_identity UNIQUE (face_id, identity_id), 
	CONSTRAINT fk_face_embeddings_identity_id_identities FOREIGN KEY(identity_id) REFERENCES identities (id) ON DELETE CASCADE, 
	CONSTRAINT fk_face_embeddings_face_id_faces FOREIGN KEY(face_id) REFERENCES faces (id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS ix_face_embeddings_identity_id ON face_embeddings (identity_id);
CREATE INDEX IF NOT EXISTS ix_face_embeddings_model_name ON face_embeddings (model_name);

-- ------------------------------------------------------------------
-- Table: detections
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS detections (
	id BIGSERIAL NOT NULL, 
	source_id INTEGER NOT NULL, 
	track_id BIGINT, 
	recording_id INTEGER, 
	timestamp TIMESTAMP WITH TIME ZONE NOT NULL, 
	frame_number INTEGER NOT NULL, 
	object_class VARCHAR(64) NOT NULL, 
	confidence FLOAT NOT NULL, 
	bbox_x1 FLOAT NOT NULL, 
	bbox_y1 FLOAT NOT NULL, 
	bbox_x2 FLOAT NOT NULL, 
	bbox_y2 FLOAT NOT NULL, 
	distance_m FLOAT, 
	proximity_zone VARCHAR(16), 
	CONSTRAINT pk_detections PRIMARY KEY (id), 
	CONSTRAINT fk_detections_source_id_video_sources FOREIGN KEY(source_id) REFERENCES video_sources (id) ON DELETE CASCADE, 
	CONSTRAINT fk_detections_track_id_tracks FOREIGN KEY(track_id) REFERENCES tracks (id) ON DELETE CASCADE, 
	CONSTRAINT fk_detections_recording_id_recording_sessions FOREIGN KEY(recording_id) REFERENCES recording_sessions (id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS ix_detections_object_class ON detections (object_class);
CREATE INDEX IF NOT EXISTS ix_detections_recording_id ON detections (recording_id);
CREATE INDEX IF NOT EXISTS ix_detections_source_timestamp ON detections (source_id, timestamp);
CREATE INDEX IF NOT EXISTS ix_detections_track_id ON detections (track_id);

-- ------------------------------------------------------------------
-- Table: motion_events
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS motion_events (
	id BIGSERIAL NOT NULL, 
	source_id INTEGER NOT NULL, 
	recording_id INTEGER, 
	started_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	ended_at TIMESTAMP WITH TIME ZONE, 
	duration_seconds FLOAT, 
	peak_area_ratio FLOAT NOT NULL, 
	frame_start INTEGER NOT NULL, 
	frame_end INTEGER, 
	snapshot_path VARCHAR(512), 
	CONSTRAINT pk_motion_events PRIMARY KEY (id), 
	CONSTRAINT fk_motion_events_source_id_video_sources FOREIGN KEY(source_id) REFERENCES video_sources (id) ON DELETE CASCADE, 
	CONSTRAINT fk_motion_events_recording_id_recording_sessions FOREIGN KEY(recording_id) REFERENCES recording_sessions (id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS ix_motion_events_source_started ON motion_events (source_id, started_at);

-- ------------------------------------------------------------------
-- Table: security_events
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS security_events (
	id BIGSERIAL NOT NULL, 
	source_id INTEGER NOT NULL, 
	event_type VARCHAR(48) NOT NULL, 
	severity VARCHAR(16) NOT NULL, 
	track_id BIGINT, 
	identity_id INTEGER, 
	face_id BIGINT, 
	recording_id INTEGER, 
	motion_event_id BIGINT, 
	started_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	ended_at TIMESTAMP WITH TIME ZONE, 
	duration_seconds FLOAT, 
	is_open BOOLEAN NOT NULL, 
	occurrence_count INTEGER NOT NULL, 
	confidence FLOAT, 
	label VARCHAR(255), 
	message TEXT, 
	snapshot_path VARCHAR(512), 
	dedup_key VARCHAR(255), 
	metadata JSONB NOT NULL, 
	CONSTRAINT pk_security_events PRIMARY KEY (id), 
	CONSTRAINT fk_security_events_source_id_video_sources FOREIGN KEY(source_id) REFERENCES video_sources (id) ON DELETE CASCADE, 
	CONSTRAINT fk_security_events_track_id_tracks FOREIGN KEY(track_id) REFERENCES tracks (id) ON DELETE SET NULL, 
	CONSTRAINT fk_security_events_identity_id_identities FOREIGN KEY(identity_id) REFERENCES identities (id) ON DELETE SET NULL, 
	CONSTRAINT fk_security_events_face_id_faces FOREIGN KEY(face_id) REFERENCES faces (id) ON DELETE SET NULL, 
	CONSTRAINT fk_security_events_recording_id_recording_sessions FOREIGN KEY(recording_id) REFERENCES recording_sessions (id) ON DELETE SET NULL, 
	CONSTRAINT fk_security_events_motion_event_id_motion_events FOREIGN KEY(motion_event_id) REFERENCES motion_events (id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS ix_security_events_dedup ON security_events (source_id, dedup_key, is_open);
CREATE INDEX IF NOT EXISTS ix_security_events_event_type ON security_events (event_type);
CREATE INDEX IF NOT EXISTS ix_security_events_identity_id ON security_events (identity_id);
CREATE INDEX IF NOT EXISTS ix_security_events_open ON security_events (is_open);
CREATE INDEX IF NOT EXISTS ix_security_events_recording_id ON security_events (recording_id);
CREATE INDEX IF NOT EXISTS ix_security_events_source_started ON security_events (source_id, started_at);
CREATE INDEX IF NOT EXISTS ix_security_events_track_id ON security_events (track_id);

-- ------------------------------------------------------------------
-- Table: alerts
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS alerts (
	id BIGSERIAL NOT NULL, 
	source_id INTEGER NOT NULL, 
	security_event_id BIGINT, 
	track_id BIGINT, 
	identity_id INTEGER, 
	alert_type VARCHAR(24) NOT NULL, 
	state VARCHAR(16) NOT NULL, 
	reason VARCHAR(255) NOT NULL, 
	started_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	stopped_at TIMESTAMP WITH TIME ZONE, 
	stopped_by VARCHAR(64), 
	acknowledged BOOLEAN NOT NULL, 
	metadata JSONB NOT NULL, 
	CONSTRAINT pk_alerts PRIMARY KEY (id), 
	CONSTRAINT fk_alerts_source_id_video_sources FOREIGN KEY(source_id) REFERENCES video_sources (id) ON DELETE CASCADE, 
	CONSTRAINT fk_alerts_security_event_id_security_events FOREIGN KEY(security_event_id) REFERENCES security_events (id) ON DELETE SET NULL, 
	CONSTRAINT fk_alerts_track_id_tracks FOREIGN KEY(track_id) REFERENCES tracks (id) ON DELETE SET NULL, 
	CONSTRAINT fk_alerts_identity_id_identities FOREIGN KEY(identity_id) REFERENCES identities (id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS ix_alerts_source_started ON alerts (source_id, started_at);
CREATE INDEX IF NOT EXISTS ix_alerts_state ON alerts (state);
CREATE INDEX IF NOT EXISTS ix_alerts_track_id ON alerts (track_id);

-- ------------------------------------------------------------------
-- Table: temporary_identity_expirations
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS temporary_identity_expirations (
	id SERIAL NOT NULL, 
	identity_id INTEGER NOT NULL, 
	scheduled_for TIMESTAMP WITH TIME ZONE NOT NULL, 
	expired_at TIMESTAMP WITH TIME ZONE, 
	retention_days INTEGER, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	CONSTRAINT pk_temporary_identity_expirations PRIMARY KEY (id), 
	CONSTRAINT fk_temporary_identity_expirations_identity_id_identities FOREIGN KEY(identity_id) REFERENCES identities (id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS ix_temp_identity_exp_identity ON temporary_identity_expirations (identity_id);
CREATE INDEX IF NOT EXISTS ix_temp_identity_exp_scheduled ON temporary_identity_expirations (scheduled_for);

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
