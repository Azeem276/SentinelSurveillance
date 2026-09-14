// Main monitoring surface: single-source view and multi-source grid.
import { useState } from 'react'

import { AddSourceDialog } from '@/components/AddSourceDialog'
import { AnalysisPanel } from '@/components/AnalysisPanel'
import { SourceList } from '@/components/SourceList'
import { SourceSettings } from '@/components/SourceSettings'
import { VideoMonitor } from '@/components/VideoMonitor'
import { Empty, ProximityIndicator, RecognitionBadge } from '@/components/primitives'
import { useSentinelStore } from '@/stores/useSentinelStore'
import type { VideoSource } from '@/types'

function SingleView({ source }: { source: VideoSource }) {
  const overlay = useSentinelStore((s) => s.overlays[source.uid])
  const alerts = useSentinelStore((s) => s.activeAlerts)
  const stopAlert = useSentinelStore((s) => s.stopAlert)
  const sourceAlarms = alerts.filter(
    (a) => a.source_id === source.id && a.alert_type === 'CONTINUOUS_ALARM',
  )

  return (
    <>
      <VideoMonitor source={source} />

      {sourceAlarms.length > 0 && (
        <div className="card" style={{ marginTop: 12, borderColor: 'var(--danger)' }}>
          <div className="row">
            <strong style={{ color: 'var(--danger)' }}>
              ALARM ACTIVE — {sourceAlarms[0].reason.replaceAll('_', ' ')}
            </strong>
            <span className="spacer" />
            {sourceAlarms.map((alert) => (
              <button
                key={alert.id}
                className="btn danger"
                onClick={() => void stopAlert(alert.id)}
              >
                STOP ALARM #{alert.id}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="card" style={{ marginTop: 12 }}>
        <h3 className="card-title">Tracked objects</h3>
        {!overlay || overlay.objects.length === 0 ? (
          <Empty>
            {source.runtime.running
              ? 'No objects currently tracked.'
              : 'Source is not running. Activate surveillance to begin.'}
          </Empty>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Track</th>
                <th>Class</th>
                <th>Identity</th>
                <th>Recognition</th>
                <th>Proximity</th>
                <th>Conf.</th>
              </tr>
            </thead>
            <tbody>
              {overlay.objects.map((obj) => (
                <tr key={obj.track_id}>
                  <td className="mono">#{obj.track_id}</td>
                  <td>{obj.class}</td>
                  <td>{obj.identity ?? <span className="muted">—</span>}</td>
                  <td>
                    {obj.class === 'person' ? (
                      <RecognitionBadge state={obj.recognition_state} />
                    ) : (
                      <span className="muted">n/a</span>
                    )}
                  </td>
                  <td>
                    <ProximityIndicator
                      zone={obj.proximity_zone}
                      distance={obj.distance_m}
                    />
                  </td>
                  <td className="mono">{obj.confidence.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  )
}

function GridView({ sources }: { sources: VideoSource[] }) {
  const selectSource = useSentinelStore((s) => s.selectSource)
  const setViewMode = useSentinelStore((s) => s.setViewMode)

  if (sources.length === 0) return <Empty>No sources configured.</Empty>

  return (
    <div className="video-grid">
      {sources.map((source) => (
        <div key={source.id} className="grid-tile">
          <VideoMonitor
            source={source}
            compact
            onClick={() => {
              selectSource(source.id)
              setViewMode('single')
            }}
          />
          <div className="grid-tile-label">
            <span>
              <strong>{source.name}</strong>
              <span className="muted mono"> · {source.uid}</span>
            </span>
            <span className="muted mono">
              A {source.proximity_a}m / B {source.proximity_b}m
            </span>
          </div>
        </div>
      ))}
    </div>
  )
}

export function MonitorPage() {
  const sources = useSentinelStore((s) => s.sources)
  const selectedId = useSentinelStore((s) => s.selectedSourceId)
  const viewMode = useSentinelStore((s) => s.viewMode)
  const setViewMode = useSentinelStore((s) => s.setViewMode)
  const system = useSentinelStore((s) => s.system)

  const [settingsFor, setSettingsFor] = useState<VideoSource | null>(null)
  const [adding, setAdding] = useState(false)

  const selected = sources.find((s) => s.id === selectedId) ?? null

  return (
    <div className="workspace">
      <SourceList onSettings={setSettingsFor} />

      <main className="monitor">
        <div className="monitor-toolbar">
          <span className="monitor-title">
            {viewMode === 'grid' ? 'Grid View' : (selected?.name ?? 'Monitor')}
          </span>
          {viewMode === 'single' && selected && (
            <span className="muted mono">
              {selected.uid} · {selected.type}
              {selected.location ? ` · ${selected.location}` : ''}
            </span>
          )}

          <span className="spacer" />

          {!system?.surveillance_active && (
            <span className="badge warning">SURVEILLANCE OFF</span>
          )}

          <div className="nav-tabs">
            <button
              className={`nav-tab ${viewMode === 'single' ? 'active' : ''}`}
              onClick={() => setViewMode('single')}
            >
              Single
            </button>
            <button
              className={`nav-tab ${viewMode === 'grid' ? 'active' : ''}`}
              onClick={() => setViewMode('grid')}
            >
              Grid
            </button>
          </div>

          <button className="btn ghost sm" onClick={() => setAdding(true)}>
            + Add source
          </button>
          {selected && (
            <button
              className="btn ghost sm"
              onClick={() => setSettingsFor(selected)}
            >
              Settings
            </button>
          )}
        </div>

        <div className="monitor-body">
          {viewMode === 'grid' ? (
            <GridView sources={sources} />
          ) : selected ? (
            <SingleView source={selected} />
          ) : (
            <Empty>Select a source from the left panel.</Empty>
          )}
        </div>
      </main>

      <AnalysisPanel source={selected} />

      {settingsFor && (
        <SourceSettings source={settingsFor} onClose={() => setSettingsFor(null)} />
      )}
      {adding && <AddSourceDialog onClose={() => setAdding(false)} />}
    </div>
  )
}
