// Per-source configuration, including the two proximity boundaries.
//
// Client-side validation mirrors the backend rule (A > B > 0) so the operator
// gets immediate feedback; the backend validates again and is authoritative.
import { useState } from 'react'

import { api } from '@/services/api'
import { useSentinelStore } from '@/stores/useSentinelStore'
import type { VideoSource } from '@/types'
import { Modal } from './primitives'

const RETENTIONS = [
  { value: 1, label: '24 hours' },
  { value: 3, label: '3 days' },
  { value: 7, label: '7 days' },
  { value: 30, label: '30 days' },
]

interface Errors {
  proximity?: string
  threshold?: string
  name?: string
}

export function SourceSettings({
  source,
  onClose,
}: {
  source: VideoSource
  onClose: () => void
}) {
  const refreshSources = useSentinelStore((s) => s.refreshSources)
  const toast = useSentinelStore((s) => s.toast)

  const [name, setName] = useState(source.name)
  const [location, setLocation] = useState(source.location ?? '')
  const [proximityA, setProximityA] = useState(String(source.proximity_a))
  const [proximityB, setProximityB] = useState(String(source.proximity_b))
  const [threshold, setThreshold] = useState(
    source.recognition_threshold != null ? String(source.recognition_threshold) : '',
  )
  const [retention, setRetention] = useState(source.temporary_retention_days ?? 7)
  const [recognition, setRecognition] = useState(source.recognition_enabled)
  const [recording, setRecording] = useState(source.recording_enabled)
  const [loop, setLoop] = useState(source.loop_playback)
  const [tempPolicy, setTempPolicy] = useState(
    String(source.alert_policy?.temporary_familiar ?? 'beep'),
  )
  const [unrecognizablePolicy, setUnrecognizablePolicy] = useState(
    String(source.alert_policy?.unrecognizable_policy ?? 'ignore'),
  )
  const [fov, setFov] = useState(String(source.calibration?.vertical_fov_deg ?? 55))
  const [scale, setScale] = useState(String(source.calibration?.distance_scale ?? 1))
  const [errors, setErrors] = useState<Errors>({})
  const [saving, setSaving] = useState(false)

  const validate = (): Errors => {
    const next: Errors = {}
    const a = Number(proximityA)
    const b = Number(proximityB)
    if (!name.trim()) next.name = 'Name is required'
    if (!Number.isFinite(a) || a <= 0) {
      next.proximity = 'Proximity A must be greater than 0'
    } else if (!Number.isFinite(b) || b <= 0) {
      next.proximity = 'Proximity B must be greater than 0'
    } else if (a <= b) {
      next.proximity =
        'Proximity A (recognition zone) must be greater than Proximity B (alarm zone)'
    }
    if (threshold.trim()) {
      const t = Number(threshold)
      if (!Number.isFinite(t) || t <= 0 || t > 1) {
        next.threshold = 'Threshold must be between 0 and 1'
      }
    }
    return next
  }

  const save = async () => {
    const found = validate()
    setErrors(found)
    if (Object.keys(found).length > 0) return

    setSaving(true)
    try {
      await api.updateSource(source.id, {
        name: name.trim(),
        location: location.trim() || null,
        proximity_a: Number(proximityA),
        proximity_b: Number(proximityB),
        recognition_threshold: threshold.trim() ? Number(threshold) : null,
        temporary_retention_days: retention,
        recognition_enabled: recognition,
        recording_enabled: recording,
        loop_playback: loop,
        alert_policy: {
          temporary_familiar: tempPolicy,
          unrecognizable_policy: unrecognizablePolicy,
          permanent_familiar: 'none',
          beep_on_recognition: true,
        },
        calibration: {
          vertical_fov_deg: Number(fov) || 55,
          reference_height_m: Number(source.calibration?.reference_height_m ?? 1.7),
          distance_scale: Number(scale) || 1,
          visible_height_fraction: Number(
            source.calibration?.visible_height_fraction ?? 1,
          ),
        },
      })
      await refreshSources()
      toast('ok', `${name} settings saved`)
      onClose()
    } catch (e) {
      toast('error', (e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <Modal
      title={`Settings — ${source.name}`}
      onClose={onClose}
      footer={
        <>
          <button className="btn ghost" onClick={onClose}>
            Cancel
          </button>
          <button className="btn primary" onClick={save} disabled={saving}>
            {saving ? 'Saving…' : 'Save'}
          </button>
        </>
      }
    >
      <div className="field">
        <label htmlFor="src-name">Source name</label>
        <input
          id="src-name"
          className={`input ${errors.name ? 'error' : ''}`}
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        {errors.name && <div className="form-error">{errors.name}</div>}
      </div>

      <div className="form-row">
        <div className="field">
          <label htmlFor="src-loc">Location</label>
          <input
            id="src-loc"
            className="input"
            value={location}
            onChange={(e) => setLocation(e.target.value)}
          />
        </div>
        <div className="field">
          <label>Source type</label>
          <input className="input" value={`${source.type} · ${source.uri}`} disabled />
        </div>
      </div>

      <h4 style={{ margin: '18px 0 8px', fontSize: 11, letterSpacing: '.8px',
                   color: 'var(--text-2)' }}>
        PROXIMITY ZONES
      </h4>
      <div className="form-row">
        <div className="field">
          <label htmlFor="prox-a">Recognition zone — Proximity A (m)</label>
          <input
            id="prox-a"
            className={`input ${errors.proximity ? 'error' : ''}`}
            type="number"
            step="0.5"
            min="0.5"
            value={proximityA}
            onChange={(e) => setProximityA(e.target.value)}
          />
          <div className="hint">Face recognition starts inside this distance.</div>
        </div>
        <div className="field">
          <label htmlFor="prox-b">Alarm zone — Proximity B (m)</label>
          <input
            id="prox-b"
            className={`input ${errors.proximity ? 'error' : ''}`}
            type="number"
            step="0.5"
            min="0.5"
            value={proximityB}
            onChange={(e) => setProximityB(e.target.value)}
          />
          <div className="hint">Security rules fire inside this distance.</div>
        </div>
      </div>
      {errors.proximity && <div className="form-error">{errors.proximity}</div>}
      <div className="hint" style={{ marginBottom: 14 }}>
        Proximity A must be farther from the camera than Proximity B (A &gt; B &gt; 0).
      </div>

      <h4 style={{ margin: '18px 0 8px', fontSize: 11, letterSpacing: '.8px',
                   color: 'var(--text-2)' }}>
        RECOGNITION
      </h4>
      <div className="form-row">
        <div className="field">
          <label htmlFor="thr">Face recognition threshold</label>
          <input
            id="thr"
            className={`input ${errors.threshold ? 'error' : ''}`}
            type="number"
            step="0.01"
            min="0.05"
            max="1"
            placeholder="use global default"
            value={threshold}
            onChange={(e) => setThreshold(e.target.value)}
          />
          {errors.threshold && <div className="form-error">{errors.threshold}</div>}
        </div>
        <div className="field">
          <label htmlFor="ret">Temporary familiar retention</label>
          <select
            id="ret"
            className="select"
            value={retention}
            onChange={(e) => setRetention(Number(e.target.value))}
          >
            {RETENTIONS.map((r) => (
              <option key={r.value} value={r.value}>
                {r.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <h4 style={{ margin: '18px 0 8px', fontSize: 11, letterSpacing: '.8px',
                   color: 'var(--text-2)' }}>
        ALERT POLICY
      </h4>
      <div className="form-row">
        <div className="field">
          <label htmlFor="tp">Temporary familiar in alarm zone</label>
          <select
            id="tp"
            className="select"
            value={tempPolicy}
            onChange={(e) => setTempPolicy(e.target.value)}
          >
            <option value="none">No alert</option>
            <option value="beep">Single beep</option>
            <option value="continuous">Continuous alarm</option>
          </select>
        </div>
        <div className="field">
          <label htmlFor="up">Unrecognizable face in alarm zone</label>
          <select
            id="up"
            className="select"
            value={unrecognizablePolicy}
            onChange={(e) => setUnrecognizablePolicy(e.target.value)}
          >
            <option value="ignore">No alarm (recommended)</option>
            <option value="alarm">Treat as intruder</option>
          </select>
          <div className="hint">
            Failing to recognise a face is not proof of an intruder.
          </div>
        </div>
      </div>

      <h4 style={{ margin: '18px 0 8px', fontSize: 11, letterSpacing: '.8px',
                   color: 'var(--text-2)' }}>
        CALIBRATION
      </h4>
      <div className="form-row">
        <div className="field">
          <label htmlFor="fov">Vertical field of view (°)</label>
          <input
            id="fov"
            className="input"
            type="number"
            step="1"
            min="10"
            max="170"
            value={fov}
            onChange={(e) => setFov(e.target.value)}
          />
        </div>
        <div className="field">
          <label htmlFor="scale">Distance scale</label>
          <input
            id="scale"
            className="input"
            type="number"
            step="0.05"
            min="0.1"
            value={scale}
            onChange={(e) => setScale(e.target.value)}
          />
          <div className="hint">
            Multiplier to calibrate against a known landmark.
          </div>
        </div>
      </div>

      <div className="stack" style={{ marginTop: 12 }}>
        <label className="row">
          <input
            type="checkbox"
            checked={recognition}
            onChange={(e) => setRecognition(e.target.checked)}
          />
          Face recognition enabled
        </label>
        <label className="row">
          <input
            type="checkbox"
            checked={recording}
            onChange={(e) => setRecording(e.target.checked)}
          />
          Recording enabled
        </label>
        <label className="row">
          <input
            type="checkbox"
            checked={loop}
            onChange={(e) => setLoop(e.target.checked)}
          />
          Loop playback (file sources)
        </label>
      </div>
    </Modal>
  )
}
