// Measured performance and model inventory - not claimed numbers.
import { useEffect, useState } from 'react'

import { Empty, KeyValue, Stat, fmtBytes } from '@/components/primitives'
import { api } from '@/services/api'
import type { Diagnostics } from '@/types'

export function DiagnosticsPage() {
  const [data, setData] = useState<Diagnostics | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const result = await api.diagnostics()
        if (!cancelled) {
          setData(result)
          setError(null)
        }
      } catch (e) {
        if (!cancelled) setError((e as Error).message)
      }
    }
    void load()
    const timer = window.setInterval(load, 3000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [])

  if (error && !data) {
    return (
      <div className="page">
        <div className="card">
          <Empty>{error}</Empty>
        </div>
      </div>
    )
  }
  if (!data) {
    return (
      <div className="page">
        <Empty>Loading diagnostics…</Empty>
      </div>
    )
  }

  const process = data.process as Record<string, number | string>
  const index = data.recognition_index as Record<string, number>
  const system = data.system as Record<string, number | boolean>

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1 className="page-title">Diagnostics</h1>
          <div className="page-sub">
            Live measurements, refreshed every 3 seconds. All inference is local.
          </div>
        </div>
        <div className="row">
          <span className="badge neutral">DEVICE {data.device.toUpperCase()}</span>
          <span className="badge neutral">
            {data.websocket_subscribers} WS CLIENT
            {data.websocket_subscribers === 1 ? '' : 'S'}
          </span>
        </div>
      </div>

      <div className="card">
        <h3 className="card-title">System</h3>
        <div className="stat-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
          <Stat value={String(system.running_sources ?? 0)} label="Running sources" />
          <Stat value={String(system.recording_sources ?? 0)} label="Recording" />
          <Stat value={String(system.active_tracks ?? 0)} label="Active tracks" />
          <Stat
            value={process.memory_mb != null ? `${process.memory_mb}MB` : '—'}
            label="Process memory"
          />
          <Stat
            value={process.cpu_percent != null ? `${process.cpu_percent}%` : '—'}
            label="Process CPU"
          />
          <Stat
            value={
              process.system_cpu_percent != null
                ? `${process.system_cpu_percent}%`
                : '—'
            }
            label="System CPU"
          />
          <Stat
            value={
              process.system_memory_percent != null
                ? `${process.system_memory_percent}%`
                : '—'
            }
            label="System memory"
          />
          <Stat value={String(process.threads ?? '—')} label="Threads" />
        </div>
      </div>

      <div className="card">
        <h3 className="card-title">Per-source performance</h3>
        {data.sources.length === 0 ? (
          <Empty>No sources are running. Activate surveillance to measure.</Empty>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Source</th>
                <th>Status</th>
                <th>Read</th>
                <th>Processed</th>
                <th>Dropped</th>
                <th>Tracks</th>
                <th>Proc. FPS</th>
                <th>Inference</th>
                <th>Detection</th>
                <th>Face</th>
                <th>Detector runs</th>
                <th>Recognitions</th>
                <th>Errors</th>
              </tr>
            </thead>
            <tbody>
              {data.sources.map((s) => {
                const row = s as Record<string, number | string | boolean | null>
                return (
                  <tr key={String(row.uid)}>
                    <td>
                      <strong>{String(row.name ?? row.uid)}</strong>
                      {row.recording ? (
                        <span className="ind rec" style={{ marginLeft: 6 }}>
                          REC
                        </span>
                      ) : null}
                      {row.intelligence ? (
                        <span className="ind intel" style={{ marginLeft: 4 }}>
                          AI
                        </span>
                      ) : null}
                    </td>
                    <td>{String(row.status)}</td>
                    <td className="mono">{String(row.frames_read ?? 0)}</td>
                    <td className="mono">{String(row.frames_processed ?? 0)}</td>
                    <td className="mono">{String(row.dropped_frames ?? 0)}</td>
                    <td className="mono">{String(row.active_tracks ?? 0)}</td>
                    <td className="mono">{row.processing_fps ?? '—'}</td>
                    <td className="mono">{row.inference_latency_ms ?? 0}ms</td>
                    <td className="mono">{row.detection_latency_ms ?? 0}ms</td>
                    <td className="mono">{row.face_latency_ms ?? 0}ms</td>
                    <td className="mono">{String(row.detector_runs ?? 0)}</td>
                    <td className="mono">{String(row.recognitions ?? 0)}</td>
                    <td className="mono">{String(row.errors ?? 0)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>

      <div className="card">
        <h3 className="card-title">Recognition index</h3>
        <KeyValue k="Identities" v={index.identities ?? 0} />
        <KeyValue k="Permanent" v={index.permanent ?? 0} />
        <KeyValue k="Temporary" v={index.temporary ?? 0} />
        <KeyValue k="Embeddings" v={index.embeddings ?? 0} />
        <KeyValue k="Match threshold" v={index.threshold ?? '—'} />
        <KeyValue k="Index version" v={index.version ?? 0} />
      </div>

      <div className="card">
        <h3 className="card-title">Hardware</h3>
        {Object.entries(data.hardware)
          .filter(([k]) => k !== 'gpu')
          .map(([k, v]) => (
            <KeyValue key={k} k={k.replaceAll('_', ' ')} v={String(v ?? '—')} />
          ))}
      </div>

      <div className="card">
        <h3 className="card-title">Local models</h3>
        <table className="table">
          <thead>
            <tr>
              <th>Model</th>
              <th>File</th>
              <th>Installed</th>
              <th>Size</th>
              <th>Licence</th>
              <th>Purpose</th>
            </tr>
          </thead>
          <tbody>
            {data.models.map((model) => (
              <tr key={model.key}>
                <td>
                  <strong>{model.key}</strong>
                </td>
                <td className="mono" style={{ fontSize: 11 }}>
                  {model.filename}
                </td>
                <td>
                  <span className={`badge ${model.installed ? 'permanent' : 'unfamiliar'}`}>
                    {model.installed ? 'YES' : 'MISSING'}
                  </span>
                </td>
                <td className="mono">
                  {model.installed ? fmtBytes(model.size_bytes) : `~${model.approx_mb}MB`}
                </td>
                <td className="muted">{model.licence}</td>
                <td className="muted">{model.purpose}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
