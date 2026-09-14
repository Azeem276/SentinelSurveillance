// Archive: source -> recording session -> events, with playback.
import { useEffect, useState } from 'react'

import { Empty, SeverityBadge, fmtBytes, fmtDateTime, fmtDuration } from '@/components/primitives'
import { api } from '@/services/api'
import { useSentinelStore } from '@/stores/useSentinelStore'
import type { Recording, SecurityEvent } from '@/types'

export function ArchivePage() {
  const sources = useSentinelStore((s) => s.sources)
  const toast = useSentinelStore((s) => s.toast)

  const [sourceId, setSourceId] = useState<number | ''>('')
  const [recordings, setRecordings] = useState<Recording[]>([])
  const [selected, setSelected] = useState<Recording | null>(null)
  const [events, setEvents] = useState<SecurityEvent[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    void api
      .recordings({ source_id: sourceId === '' ? undefined : sourceId, limit: 200 })
      .then(setRecordings)
      .catch((e) => toast('error', (e as Error).message))
      .finally(() => setLoading(false))
  }, [sourceId, toast])

  useEffect(() => {
    if (!selected) {
      setEvents([])
      return
    }
    void api
      .recordingEvents(selected.id)
      .then(setEvents)
      .catch(() => setEvents([]))
  }, [selected])

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1 className="page-title">Archive</h1>
          <div className="page-sub">
            Each uninterrupted capture is its own file. Interrupted sessions are
            kept and labelled, never appended to.
          </div>
        </div>
        <select
          className="select"
          style={{ width: 'auto' }}
          value={sourceId}
          onChange={(e) => setSourceId(e.target.value === '' ? '' : Number(e.target.value))}
        >
          <option value="">All sources</option>
          {sources.map((s) => (
            <option key={s.id} value={s.id}>
              {s.name}
            </option>
          ))}
        </select>
      </div>

      {selected && (
        <div className="card">
          <div className="row" style={{ marginBottom: 10 }}>
            <h3 className="card-title" style={{ margin: 0 }}>
              {selected.source_name} — {fmtDateTime(selected.started_at)}
            </h3>
            <span className="spacer" />
            <button className="btn ghost sm" onClick={() => setSelected(null)}>
              Close
            </button>
          </div>

          {selected.exists_on_disk ? (
            <video
              key={selected.id}
              controls
              style={{ width: '100%', maxHeight: 460, background: '#000',
                       borderRadius: 6 }}
              src={api.playbackUrl(selected.id)}
            />
          ) : (
            <Empty>The file for this session is no longer on disk.</Empty>
          )}

          <div className="row wrap" style={{ marginTop: 10, gap: 14 }}>
            <span className="muted mono">{selected.file_path}</span>
            <span className="muted">{fmtDuration(selected.duration_seconds)}</span>
            <span className="muted">{selected.frame_count} frames</span>
            <span className="muted">{fmtBytes(selected.file_size_bytes)}</span>
            {selected.error && (
              <span style={{ color: 'var(--warn)' }}>{selected.error}</span>
            )}
          </div>

          <h4 style={{ marginTop: 16, marginBottom: 6, fontSize: 11,
                       letterSpacing: '.8px', color: 'var(--text-2)' }}>
            EVENTS DURING THIS SESSION
          </h4>
          {events.length === 0 ? (
            <Empty>No events recorded during this session.</Empty>
          ) : (
            <table className="table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Event</th>
                  <th>Severity</th>
                  <th>Label</th>
                  <th>Message</th>
                </tr>
              </thead>
              <tbody>
                {events.slice(0, 40).map((ev) => (
                  <tr key={ev.id}>
                    <td className="mono" style={{ fontSize: 11 }}>
                      {fmtDateTime(ev.started_at)}
                    </td>
                    <td>{ev.event_type.replaceAll('_', ' ')}</td>
                    <td>
                      <SeverityBadge severity={ev.severity} />
                    </td>
                    <td>{ev.label ?? '—'}</td>
                    <td className="muted">{ev.message ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      <div className="card">
        <h3 className="card-title">Recording sessions</h3>
        {loading && recordings.length === 0 && <Empty>Loading…</Empty>}
        {!loading && recordings.length === 0 && (
          <Empty>
            No recordings yet. Activate surveillance to start capturing.
          </Empty>
        )}
        {recordings.length > 0 && (
          <table className="table">
            <thead>
              <tr>
                <th>Source</th>
                <th>Started</th>
                <th>Duration</th>
                <th>Frames</th>
                <th>Size</th>
                <th>Status</th>
                <th>File</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {recordings.map((rec) => (
                <tr key={rec.id}>
                  <td>{rec.source_name ?? rec.source_uid}</td>
                  <td className="mono" style={{ fontSize: 11 }}>
                    {fmtDateTime(rec.started_at)}
                  </td>
                  <td className="mono">{fmtDuration(rec.duration_seconds)}</td>
                  <td className="mono">{rec.frame_count}</td>
                  <td className="mono">{fmtBytes(rec.file_size_bytes)}</td>
                  <td>
                    <span
                      className={`badge ${
                        rec.status === 'COMPLETED'
                          ? 'permanent'
                          : rec.status === 'ACTIVE'
                            ? 'notice'
                            : rec.status === 'INTERRUPTED'
                              ? 'warning'
                              : 'unfamiliar'
                      }`}
                    >
                      {rec.status}
                    </span>
                  </td>
                  <td className="mono muted" style={{ fontSize: 10.5 }}>
                    {rec.file_path}
                    {!rec.exists_on_disk && (
                      <span style={{ color: 'var(--warn)' }}> (missing)</span>
                    )}
                  </td>
                  <td>
                    <button
                      className="btn ghost sm"
                      disabled={!rec.exists_on_disk}
                      onClick={() => setSelected(rec)}
                    >
                      Play
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
