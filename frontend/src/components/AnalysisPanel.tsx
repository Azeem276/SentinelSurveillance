// Right-hand contextual analysis panel with the source event timeline.
//
// Section 42 of the brief: when unfamiliar faces await review the detailed
// analysis may be gated behind a review prompt. That gate is purely a UI
// affordance - surveillance, recording, intelligence, events and alerts all
// continue regardless, and the live status block above stays visible.
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { api } from '@/services/api'
import { useSentinelStore } from '@/stores/useSentinelStore'
import type { SourceAnalysis, VideoSource } from '@/types'
import {
  Empty, KeyValue, ProximityIndicator, RecognitionBadge, SeverityBadge, Stat,
  fmtDuration, fmtTime,
} from './primitives'

const WINDOWS = [1, 6, 24, 168]

export function AnalysisPanel({ source }: { source: VideoSource | null }) {
  const navigate = useNavigate()
  const [analysis, setAnalysis] = useState<SourceAnalysis | null>(null)
  const [hours, setHours] = useState(24)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const overlay = useSentinelStore((s) => s.overlays[source?.uid ?? ''])
  const live = useSentinelStore((s) => s.liveSources[source?.uid ?? ''])
  const pendingReview = useSentinelStore((s) => s.pendingReview)
  const gateDismissed = useSentinelStore((s) => s.reviewGateDismissed)
  const dismissGate = useSentinelStore((s) => s.dismissReviewGate)

  useEffect(() => {
    if (!source) {
      setAnalysis(null)
      return
    }
    let cancelled = false
    const load = async () => {
      setLoading(true)
      try {
        const data = await api.analysis(source.id, hours)
        if (!cancelled) {
          setAnalysis(data)
          setError(null)
        }
      } catch (e) {
        if (!cancelled) setError((e as Error).message)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    void load()
    // Aggregates are periodic, not realtime: the live block above covers
    // instantaneous state, so a slow refresh here is correct and cheap.
    const timer = window.setInterval(load, 15_000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [source, hours])

  if (!source) {
    return (
      <aside className="analysis-panel">
        <div className="panel-header">
          <span className="panel-title">ANALYSIS</span>
        </div>
        <div className="panel-body">
          <Empty>Select a source to see its analysis.</Empty>
        </div>
      </aside>
    )
  }

  const tracked = overlay?.objects ?? []
  const gated = pendingReview > 0 && !gateDismissed

  return (
    <aside className="analysis-panel">
      <div className="panel-header">
        <div>
          <div className="panel-title">{source.name}</div>
          <div className="muted mono" style={{ fontSize: 10 }}>
            {source.uid} · {source.type}
          </div>
        </div>
        <select
          className="select"
          style={{ width: 'auto' }}
          value={hours}
          onChange={(e) => setHours(Number(e.target.value))}
        >
          {WINDOWS.map((h) => (
            <option key={h} value={h}>
              {h < 24 ? `${h}h` : `${h / 24}d`}
            </option>
          ))}
        </select>
      </div>

      <div className="panel-body">
        {/* Live state is never gated. */}
        <section className="panel-section">
          <h4>Live</h4>
          <div className="stat-grid">
            <Stat value={tracked.length} label="Tracked now" />
            <Stat
              value={live?.recording ?? source.runtime.recording ? 'REC' : 'OFF'}
              label="Recording"
              tone={live?.recording ?? source.runtime.recording ? 'danger' : undefined}
            />
            <Stat
              value={live?.intelligence ?? source.runtime.intelligence ? 'ON' : 'OFF'}
              label="Intelligence"
              tone={live?.intelligence ?? source.runtime.intelligence ? 'ok' : undefined}
            />
            <Stat
              value={overlay ? `${Math.round(overlay.inference_ms)}ms` : '—'}
              label="Inference"
            />
          </div>

          {tracked.length > 0 && (
            <div className="stack" style={{ marginTop: 10 }}>
              {tracked.map((obj) => (
                <div
                  key={obj.track_id}
                  className="card"
                  style={{ padding: '7px 9px', background: 'var(--bg-2)' }}
                >
                  <div className="row wrap" style={{ gap: 6 }}>
                    <strong>{obj.label}</strong>
                    <span className="muted mono">#{obj.track_id}</span>
                    <span className="spacer" />
                    <span className="badge neutral">{obj.class}</span>
                  </div>
                  <div className="row wrap" style={{ gap: 6, marginTop: 5 }}>
                    {obj.class === 'person' && (
                      <RecognitionBadge state={obj.recognition_state} />
                    )}
                    <ProximityIndicator
                      zone={obj.proximity_zone}
                      distance={obj.distance_m}
                    />
                    {obj.alarm && <span className="badge critical">ALARM</span>}
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        {gated ? (
          <section className="panel-section">
            <h4>Analysis locked</h4>
            <div className="card" style={{ borderColor: 'var(--warn-dim)' }}>
              <p style={{ marginTop: 0 }}>
                <strong>{pendingReview}</strong> unfamiliar face
                {pendingReview === 1 ? '' : 's'} require review before the detailed
                analysis is shown.
              </p>
              <p className="muted" style={{ fontSize: 11.5 }}>
                Surveillance, recording, intelligence, event generation and alarms
                are unaffected — this is a review workflow only.
              </p>
              <div className="row" style={{ marginTop: 10 }}>
                <button className="btn primary" onClick={() => navigate('/review')}>
                  Review Now
                </button>
                <button className="btn ghost" onClick={dismissGate}>
                  Show anyway
                </button>
              </div>
            </div>
          </section>
        ) : (
          <>
            {error && (
              <div className="card" style={{ borderColor: 'var(--danger-dim)' }}>
                <span className="muted">{error}</span>
              </div>
            )}
            {!analysis && loading && <Empty>Loading analysis…</Empty>}

            {analysis && (
              <>
                <section className="panel-section">
                  <h4>Objects ({analysis.window_hours}h)</h4>
                  {analysis.objects.length === 0 ? (
                    <Empty>No objects detected in this window.</Empty>
                  ) : (
                    analysis.objects.map((o) => (
                      <KeyValue
                        key={o.object_class}
                        k={o.object_class}
                        v={o.count}
                      />
                    ))
                  )}
                  <div className="stat-grid" style={{ marginTop: 8 }}>
                    <Stat value={analysis.people} label="People" />
                    <Stat value={analysis.total_objects} label="Total objects" />
                  </div>
                </section>

                <section className="panel-section">
                  <h4>Identities</h4>
                  {analysis.identities.length === 0 ? (
                    <Empty>No identified people in this window.</Empty>
                  ) : (
                    <div className="stack">
                      {analysis.identities.map((identity, index) => (
                        <div
                          key={`${identity.identity_id ?? identity.label}-${index}`}
                          className="row"
                          style={{ gap: 6 }}
                        >
                          <RecognitionBadge
                            state={
                              identity.recognition_state as never
                            }
                          />
                          <span style={{ minWidth: 0, overflow: 'hidden',
                                         textOverflow: 'ellipsis' }}>
                            {identity.label}
                          </span>
                          <span className="spacer" />
                          <span className="mono muted">×{identity.appearances}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  <div className="stat-grid" style={{ marginTop: 8 }}>
                    <Stat value={analysis.permanent_count} label="Permanent" tone="ok" />
                    <Stat value={analysis.temporary_count} label="Temporary" tone="warn" />
                    <Stat
                      value={analysis.unfamiliar_count}
                      label="Unfamiliar"
                      tone={analysis.unfamiliar_count > 0 ? 'danger' : undefined}
                    />
                    <Stat value={analysis.unrecognizable_count} label="Unrecognizable" />
                  </div>
                </section>

                <section className="panel-section">
                  <h4>Activity</h4>
                  <KeyValue k="Motion events" v={analysis.motion_events} />
                  <KeyValue k="Recognition events" v={analysis.recognition_events} />
                  <KeyValue k="Active alerts" v={analysis.active_alerts} />
                  <KeyValue k="Pending review" v={analysis.pending_review} />
                </section>

                <section className="panel-section">
                  <h4>Event timeline</h4>
                  {analysis.timeline.length === 0 ? (
                    <Empty>No events recorded in this window.</Empty>
                  ) : (
                    <div className="timeline">
                      {analysis.timeline.map((entry) => (
                        <div
                          key={entry.id}
                          className={`timeline-item ${
                            entry.severity === 'CRITICAL' ? 'critical' : ''
                          }`}
                        >
                          <div className="timeline-time">{fmtTime(entry.timestamp)}</div>
                          <div className="timeline-body">
                            <div className="timeline-title">
                              <SeverityBadge severity={entry.severity} />
                              <strong style={{ fontSize: 11 }}>
                                {entry.event_type.replaceAll('_', ' ')}
                              </strong>
                              {entry.is_open && (
                                <span className="badge warning">OPEN</span>
                              )}
                            </div>
                            <div className="timeline-msg">
                              {entry.message ?? entry.label ?? '—'}
                            </div>
                            <div className="muted mono" style={{ fontSize: 10 }}>
                              {entry.identity_label && `${entry.identity_label} · `}
                              {entry.track_id != null && `track ${entry.track_id} · `}
                              {entry.duration_seconds != null &&
                                `${fmtDuration(entry.duration_seconds)} · `}
                              {entry.recording_id != null && `rec ${entry.recording_id}`}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </section>
              </>
            )}
          </>
        )}
      </div>
    </aside>
  )
}
