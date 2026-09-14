// Persistent left-hand source panel. Scrolls to any number of sources.
import { useMemo, useState } from 'react'

import { api } from '@/services/api'
import { useSentinelStore } from '@/stores/useSentinelStore'
import type { VideoSource } from '@/types'

function SourceCard({
  source,
  selected,
  onSelect,
  onSettings,
}: {
  source: VideoSource
  selected: boolean
  onSelect: () => void
  onSettings: () => void
}) {
  const live = useSentinelStore((s) => s.liveSources[source.uid])
  const motion = useSentinelStore((s) => s.motion[source.uid])
  const alerts = useSentinelStore((s) => s.activeAlerts)
  const refreshSources = useSentinelStore((s) => s.refreshSources)
  const toast = useSentinelStore((s) => s.toast)
  const system = useSentinelStore((s) => s.system)
  const [busy, setBusy] = useState(false)

  const status = live?.status ?? source.runtime.status ?? source.status
  const recording = live?.recording ?? source.runtime.recording
  const intelligence = live?.intelligence ?? source.runtime.intelligence
  const running = ['AVAILABLE', 'STARTING'].includes(status)
  const alarming = alerts.some(
    (a) => a.source_id === source.id && a.alert_type === 'CONTINUOUS_ALARM',
  )

  const toggleIntelligence = async (event: React.MouseEvent) => {
    event.stopPropagation()
    setBusy(true)
    try {
      await api.setSourceIntelligence(source.id, !source.intelligence_enabled)
      await refreshSources()
    } catch (e) {
      toast('error', (e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const cls = [
    'source-card',
    selected ? 'selected' : '',
    alarming ? 'alarm' : motion ? 'motion' : '',
    status === 'UNAVAILABLE' || status === 'ERROR' ? 'offline' : '',
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <div className={cls} onClick={onSelect} role="button" tabIndex={0}
         onKeyDown={(e) => e.key === 'Enter' && onSelect()}>
      <div className="source-card-top">
        <span className="source-name">{source.name}</span>
        <span
          className="status-chip off"
          style={{ padding: '1px 6px', fontSize: 9 }}
          title={`Status: ${status}`}
        >
          <span
            className="dot"
            style={{
              background:
                status === 'AVAILABLE'
                  ? 'var(--ok)'
                  : status === 'UNAVAILABLE' || status === 'ERROR'
                    ? 'var(--danger)'
                    : 'var(--neutral)',
            }}
          />
          {status}
        </span>
      </div>

      <div className="source-meta">
        {source.type} · {source.location ?? source.uid}
      </div>
      <div className="source-meta">
        A {source.proximity_a}m / B {source.proximity_b}m
      </div>

      <div className="source-indicators">
        {recording && <span className="ind rec">REC</span>}
        {intelligence && <span className="ind intel">AI</span>}
        {!source.intelligence_enabled && <span className="ind">AI OFF</span>}
        {motion && <span className="ind motion">MOTION</span>}
        {alarming && <span className="ind alert">ALARM</span>}
        {running && !recording && <span className="ind live">LIVE</span>}
      </div>

      <div className="row" style={{ marginTop: 7, gap: 4 }}>
        <button
          className="btn ghost sm"
          disabled={busy || !system?.surveillance_active}
          onClick={toggleIntelligence}
          title={
            system?.surveillance_active
              ? 'Toggle intelligence for this source'
              : 'Surveillance is off'
          }
        >
          {source.intelligence_enabled ? 'AI off' : 'AI on'}
        </button>
        <button
          className="btn ghost sm"
          onClick={(e) => {
            e.stopPropagation()
            onSettings()
          }}
        >
          Settings
        </button>
      </div>
    </div>
  )
}

export function SourceList({ onSettings }: { onSettings: (source: VideoSource) => void }) {
  const sources = useSentinelStore((s) => s.sources)
  const selectedId = useSentinelStore((s) => s.selectedSourceId)
  const selectSource = useSentinelStore((s) => s.selectSource)
  const [filter, setFilter] = useState('')

  const visible = useMemo(() => {
    const needle = filter.trim().toLowerCase()
    if (!needle) return sources
    return sources.filter(
      (s) =>
        s.name.toLowerCase().includes(needle) ||
        s.uid.toLowerCase().includes(needle) ||
        (s.location ?? '').toLowerCase().includes(needle),
    )
  }, [sources, filter])

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <span>VIDEO SOURCES</span>
        <span className="mono">{sources.length}</span>
      </div>

      <div style={{ padding: '8px 8px 0' }}>
        <input
          className="input"
          placeholder="Filter sources…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
      </div>

      <div className="source-list">
        {visible.length === 0 && (
          <div className="empty">
            {sources.length === 0
              ? 'No sources configured. Add one from the Monitor toolbar.'
              : 'No sources match the filter.'}
          </div>
        )}
        {visible.map((source) => (
          <SourceCard
            key={source.id}
            source={source}
            selected={source.id === selectedId}
            onSelect={() => selectSource(source.id)}
            onSettings={() => onSettings(source)}
          />
        ))}
      </div>
    </aside>
  )
}
