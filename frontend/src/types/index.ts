// Domain types mirroring the backend schemas.
// Keeping the unions explicit means the UI cannot silently conflate
// "no face", "unrecognizable", "unfamiliar" and "familiar".

export type SourceType = 'FILE' | 'RTSP' | 'WEBCAM'

export type SourceStatus =
  | 'IDLE'
  | 'STARTING'
  | 'AVAILABLE'
  | 'UNAVAILABLE'
  | 'ENDED'
  | 'ERROR'

export type RecognitionState =
  | 'NO_FACE'
  | 'UNKNOWN_PENDING_RECOGNITION'
  | 'FACE_UNRECOGNIZABLE'
  | 'UNFAMILIAR'
  | 'TEMPORARY_FAMILIAR'
  | 'PERMANENT_FAMILIAR'

export type ProximityZone = 'FAR' | 'ZONE_A' | 'ZONE_B' | 'UNKNOWN'
export type IdentityCategory = 'PERMANENT' | 'TEMPORARY'
export type IdentityStatus = 'ACTIVE' | 'EXPIRED' | 'DELETED'
export type AlertType = 'BEEP' | 'CONTINUOUS_ALARM'
export type AlertState = 'ACTIVE' | 'STOPPED' | 'AUTO_CLEARED'
export type RecordingStatus = 'ACTIVE' | 'COMPLETED' | 'INTERRUPTED' | 'FAILED'
export type EventSeverity = 'INFO' | 'NOTICE' | 'WARNING' | 'CRITICAL'
export type ReviewStatus = 'PENDING' | 'CLASSIFIED' | 'DISMISSED'

export interface SourceRuntime {
  running: boolean
  status: SourceStatus
  recording: boolean
  recording_id: number | null
  intelligence: boolean
  motion: boolean
  active_tracks: number
  frames_read: number
  error: string | null
  stats: Record<string, number>
  capabilities: {
    width: number
    height: number
    fps: number
    frame_count: number | null
    is_live: boolean
  } | null
}

export interface VideoSource {
  id: number
  uid: string
  name: string
  type: SourceType
  uri: string
  location: string | null
  description: string | null
  enabled: boolean
  surveillance_enabled: boolean
  intelligence_enabled: boolean
  recognition_enabled: boolean
  recording_enabled: boolean
  loop_playback: boolean
  status: SourceStatus
  last_error: string | null
  last_seen_at: string | null
  proximity_a: number
  proximity_b: number
  recognition_threshold: number | null
  detection_confidence: number | null
  temporary_retention_days: number | null
  alert_policy: Record<string, unknown>
  calibration: Record<string, number>
  display_order: number
  created_at: string
  updated_at: string
  runtime: SourceRuntime
  unread_alerts?: number
  stream_url?: string | null
}

export interface SystemState {
  surveillance_active: boolean
  intelligence_active: boolean
  intelligence_switch: boolean
  running_sources: number
  recording_sources: number
  active_tracks: number
  active_alerts: number
  pending_review: number
  database_connected: boolean
  device: string
}

export interface BoundingBox {
  x1: number
  y1: number
  x2: number
  y2: number
}

/** One tracked object as drawn on the live overlay. */
export interface OverlayObject {
  track_id: number
  db_track_id: number | null
  class: string
  confidence: number
  bbox: BoundingBox | null
  recognition_state: RecognitionState
  identity_id: number | null
  identity: string | null
  label: string
  match_score: number
  proximity_zone: ProximityZone
  distance_m: number | null
  alarm: boolean
  duration_seconds: number
}

export interface FrameAnalysis {
  source_id: string
  frame_number: number
  timestamp: string
  width: number
  height: number
  motion: boolean
  objects: OverlayObject[]
  intelligence: boolean
  recording: boolean
  inference_ms: number
}

export interface Identity {
  id: number
  generated_identifier: string
  display_name: string | null
  category: IdentityCategory
  status: IdentityStatus
  first_detected_at: string
  last_seen_at: string | null
  expires_at: string | null
  retention_days: number | null
  notes: string | null
  thumbnail_path: string | null
  thumbnail_url: string | null
  source_id: number | null
  created_at: string
  embedding_count: number
  track_count: number
  label: string
}

export interface UnfamiliarFace {
  id: number
  source_id: number
  source_uid: string | null
  source_name: string | null
  track_id: number | null
  identity_id: number | null
  detected_at: string
  frame_number: number
  image_path: string | null
  image_url: string | null
  detection_confidence: number
  quality_score: number
  quality_ok: boolean
  quality_reason: string | null
  face_pixels: number | null
  recognition_state: RecognitionState
  match_score: number | null
  review_status: ReviewStatus
  suggested_identifier: string | null
}

export interface SecurityEvent {
  id: number
  source_id: number
  event_type: string
  severity: EventSeverity
  track_id: number | null
  identity_id: number | null
  face_id: number | null
  recording_id: number | null
  started_at: string
  ended_at: string | null
  duration_seconds: number | null
  is_open: boolean
  occurrence_count: number
  confidence: number | null
  label: string | null
  message: string | null
  event_metadata: Record<string, unknown>
}

export interface TimelineEntry {
  id: number
  timestamp: string
  event_type: string
  severity: string
  label: string | null
  message: string | null
  duration_seconds: number | null
  track_id: number | null
  identity_id: number | null
  identity_label: string | null
  recording_id: number | null
  object_class: string | null
  is_open: boolean
}

export interface Alert {
  id: number
  source_id: number
  alert_type: AlertType
  state: AlertState
  reason: string
  started_at: string
  stopped_at: string | null
  stopped_by: string | null
  track_id: number | null
  identity_id: number | null
  security_event_id: number | null
  acknowledged: boolean
}

export interface Recording {
  id: number
  source_id: number
  source_uid: string | null
  source_name: string | null
  file_path: string
  status: RecordingStatus
  started_at: string
  ended_at: string | null
  duration_seconds: number | null
  frame_count: number
  fps: number | null
  width: number | null
  height: number | null
  file_size_bytes: number | null
  error: string | null
  playback_url: string | null
  exists_on_disk: boolean
  event_count: number
}

export interface SourceAnalysis {
  source_id: number
  source_uid: string
  source_name: string
  window_hours: number
  objects: { object_class: string; count: number }[]
  total_objects: number
  people: number
  identities: {
    identity_id: number | null
    label: string
    category: string | null
    recognition_state: string
    appearances: number
    last_seen_at: string | null
  }[]
  permanent_count: number
  temporary_count: number
  unfamiliar_count: number
  unrecognizable_count: number
  motion_events: number
  recognition_events: number
  active_alerts: number
  event_counts: Record<string, number>
  timeline: TimelineEntry[]
  pending_review: number
  runtime: Record<string, unknown>
}

export interface ModelInfo {
  key: string
  filename: string
  installed: boolean
  path: string | null
  size_bytes: number | null
  approx_mb: number
  licence: string
  origin: string
  purpose: string
}

export interface Diagnostics {
  device: string
  hardware: Record<string, unknown>
  system: Record<string, unknown>
  sources: Record<string, unknown>[]
  recognition_index: Record<string, unknown>
  models: ModelInfo[]
  process: Record<string, unknown>
  websocket_subscribers: number
}

/** Messages pushed over the realtime channel. */
export type WsMessage =
  | { type: 'snapshot'; system: SystemState; sources: unknown[]; active_alerts: unknown[]; pending_review: number }
  | { type: 'ping' | 'pong' }
  | ({ type: 'frame_analysis' } & FrameAnalysis)
  | { type: 'motion'; source_id: string; active: boolean; area_ratio?: number }
  | { type: 'event'; source_id: string | null; event: SecurityEvent }
  | { type: 'alarm_started'; source_id: string | null; alert: Record<string, unknown> }
  | { type: 'alarm_stopped'; source_id: string | null; alert: Record<string, unknown> }
  | { type: 'alert_beep'; source_id: string | null; alert: Record<string, unknown> }
  | { type: 'alarms_cleared'; count: number }
  | { type: 'system_state'; state: SystemState }
  | { type: 'source_state'; source_id: string; state: Record<string, unknown> }
  | { type: 'recognition'; source_id: string; [k: string]: unknown }
  | { type: 'proximity'; source_id: string; [k: string]: unknown }
  | { type: 'track_started' | 'track_ended'; source_id: string; [k: string]: unknown }
  | { type: 'identities_expired'; identities: unknown[] }
  | { type: string; [k: string]: unknown }
