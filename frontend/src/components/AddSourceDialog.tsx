// Register a new video source.
//
// File sources are chosen from clips already present in the configured video
// directory (the backend refuses arbitrary paths), or uploaded here. The RTSP
// branch is live: the same VideoSource abstraction backs both.
import { useEffect, useState } from 'react'

import { api } from '@/services/api'
import { useSentinelStore } from '@/stores/useSentinelStore'
import { Modal } from './primitives'

export function AddSourceDialog({ onClose }: { onClose: () => void }) {
  const refreshSources = useSentinelStore((s) => s.refreshSources)
  const toast = useSentinelStore((s) => s.toast)

  const [type, setType] = useState<'FILE' | 'RTSP'>('FILE')
  const [uid, setUid] = useState('')
  const [name, setName] = useState('')
  const [location, setLocation] = useState('')
  const [uri, setUri] = useState('')
  const [files, setFiles] = useState<{ filename: string; size_bytes: number }[]>([])
  const [proximityA, setProximityA] = useState('10')
  const [proximityB, setProximityB] = useState('3')
  const [saving, setSaving] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    void api
      .availableVideos()
      .then((list) => {
        setFiles(list)
        if (list.length > 0 && !uri) setUri(list[0].filename)
      })
      .catch(() => setFiles([]))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const upload = async (file: File) => {
    setUploading(true)
    try {
      const body = new FormData()
      body.append('file', file)
      const response = await fetch('/api/sources/upload', { method: 'POST', body })
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}))
        throw new Error(detail.message ?? 'upload failed')
      }
      const result = await response.json()
      const filename = result.data.filename as string
      setFiles((prev) => [...prev, { filename, size_bytes: result.data.bytes }])
      setUri(filename)
      toast('ok', `Uploaded ${filename}`)
    } catch (e) {
      toast('error', (e as Error).message)
    } finally {
      setUploading(false)
    }
  }

  const submit = async () => {
    const a = Number(proximityA)
    const b = Number(proximityB)
    if (!uid.trim() || !name.trim() || !uri.trim()) {
      setError('Identifier, name and source URI are all required')
      return
    }
    if (!/^[A-Za-z0-9._-]+$/.test(uid.trim())) {
      setError('Identifier may only contain letters, digits, dot, dash and underscore')
      return
    }
    if (!(a > b) || b <= 0) {
      setError('Proximity A must be greater than Proximity B, and both above 0')
      return
    }

    setSaving(true)
    setError(null)
    try {
      await api.createSource({
        uid: uid.trim(),
        name: name.trim(),
        type,
        uri: uri.trim(),
        location: location.trim() || null,
        proximity_a: a,
        proximity_b: b,
      })
      await refreshSources()
      toast('ok', `Source ${name} created`)
      onClose()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <Modal
      title="Add video source"
      onClose={onClose}
      footer={
        <>
          <button className="btn ghost" onClick={onClose}>
            Cancel
          </button>
          <button className="btn primary" onClick={submit} disabled={saving}>
            {saving ? 'Creating…' : 'Create source'}
          </button>
        </>
      }
    >
      <div className="field">
        <label>Source type</label>
        <div className="row">
          <button
            className={`btn ${type === 'FILE' ? 'primary' : 'ghost'}`}
            onClick={() => setType('FILE')}
          >
            Video file
          </button>
          <button
            className={`btn ${type === 'RTSP' ? 'primary' : 'ghost'}`}
            onClick={() => setType('RTSP')}
          >
            RTSP camera
          </button>
        </div>
        <div className="hint">
          Both use the same VideoSource abstraction — the intelligence pipeline
          does not care which.
        </div>
      </div>

      <div className="form-row">
        <div className="field">
          <label htmlFor="new-uid">Identifier</label>
          <input
            id="new-uid"
            className="input"
            placeholder="camera_04"
            value={uid}
            onChange={(e) => setUid(e.target.value)}
          />
        </div>
        <div className="field">
          <label htmlFor="new-name">Display name</label>
          <input
            id="new-name"
            className="input"
            placeholder="Side Entrance"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </div>
      </div>

      <div className="field">
        <label htmlFor="new-loc">Location</label>
        <input
          id="new-loc"
          className="input"
          placeholder="East wall"
          value={location}
          onChange={(e) => setLocation(e.target.value)}
        />
      </div>

      {type === 'FILE' ? (
        <div className="field">
          <label htmlFor="new-file">Video clip</label>
          <select
            id="new-file"
            className="select"
            value={uri}
            onChange={(e) => setUri(e.target.value)}
          >
            {files.length === 0 && <option value="">No clips found</option>}
            {files.map((f) => (
              <option key={f.filename} value={f.filename}>
                {f.filename} ({(f.size_bytes / 1e6).toFixed(1)} MB)
              </option>
            ))}
          </select>
          <div className="row" style={{ marginTop: 8 }}>
            <input
              type="file"
              accept="video/mp4,video/x-matroska,video/avi,video/quicktime,.mp4,.mkv,.avi,.mov"
              disabled={uploading}
              onChange={(e) => {
                const file = e.target.files?.[0]
                if (file) void upload(file)
              }}
            />
            {uploading && <span className="muted">Uploading…</span>}
          </div>
          <div className="hint">
            Clips are read from the configured video directory only.
          </div>
        </div>
      ) : (
        <div className="field">
          <label htmlFor="new-rtsp">RTSP URL</label>
          <input
            id="new-rtsp"
            className="input"
            placeholder="rtsp://user:pass@192.168.1.50:554/stream1"
            value={uri}
            onChange={(e) => setUri(e.target.value)}
          />
          <div className="hint">
            Credentials are stored in the database, never exposed by the read APIs.
          </div>
        </div>
      )}

      <div className="form-row">
        <div className="field">
          <label htmlFor="new-a">Proximity A — recognition (m)</label>
          <input
            id="new-a"
            className="input"
            type="number"
            step="0.5"
            value={proximityA}
            onChange={(e) => setProximityA(e.target.value)}
          />
        </div>
        <div className="field">
          <label htmlFor="new-b">Proximity B — alarm (m)</label>
          <input
            id="new-b"
            className="input"
            type="number"
            step="0.5"
            value={proximityB}
            onChange={(e) => setProximityB(e.target.value)}
          />
        </div>
      </div>

      {error && <div className="form-error">{error}</div>}
    </Modal>
  )
}
