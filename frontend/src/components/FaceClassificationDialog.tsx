// Classification dialogs for the unfamiliar-face review workflow.
//
// Naming is optional throughout: when the operator leaves the name blank the
// backend derives a stable identifier from the original detection timestamp
// (Unknown_YYYYMMDD_HHMMSS), which is previewed here so the behaviour is
// never a surprise.
import { useState } from 'react'

import { api } from '@/services/api'
import { useSentinelStore } from '@/stores/useSentinelStore'
import type { IdentityCategory, UnfamiliarFace } from '@/types'
import { Modal, fmtDateTime } from './primitives'

const RETENTION_PRESETS = [
  { value: '24h', label: '24 hours' },
  { value: '3d', label: '3 days' },
  { value: '7d', label: '7 days (default)' },
  { value: '30d', label: '30 days' },
  { value: 'custom', label: 'Custom…' },
]

function CategoryPicker({
  category,
  onChange,
}: {
  category: IdentityCategory
  onChange: (c: IdentityCategory) => void
}) {
  return (
    <div className="field">
      <label>Category</label>
      <div className="row">
        <button
          className={`btn ${category === 'PERMANENT' ? 'success' : 'ghost'}`}
          onClick={() => onChange('PERMANENT')}
        >
          Permanent Familiar
        </button>
        <button
          className={`btn ${category === 'TEMPORARY' ? '' : 'ghost'}`}
          style={
            category === 'TEMPORARY'
              ? { background: 'var(--warn)', borderColor: 'var(--warn)', color: '#231602' }
              : undefined
          }
          onClick={() => onChange('TEMPORARY')}
        >
          Temporary Familiar
        </button>
      </div>
      <div className="hint">
        {category === 'PERMANENT'
          ? 'Never expires. No alarm when entering the alarm zone.'
          : 'Expires automatically. Beeps once on recognition.'}
      </div>
    </div>
  )
}

function RetentionPicker({
  preset,
  onPreset,
  customDays,
  onCustomDays,
}: {
  preset: string
  onPreset: (p: string) => void
  customDays: string
  onCustomDays: (d: string) => void
}) {
  return (
    <div className="form-row">
      <div className="field">
        <label htmlFor="retention">Retention period</label>
        <select
          id="retention"
          className="select"
          value={preset}
          onChange={(e) => onPreset(e.target.value)}
        >
          {RETENTION_PRESETS.map((r) => (
            <option key={r.value} value={r.value}>
              {r.label}
            </option>
          ))}
        </select>
      </div>
      {preset === 'custom' && (
        <div className="field">
          <label htmlFor="customdays">Custom days</label>
          <input
            id="customdays"
            className="input"
            type="number"
            min="1"
            max="3650"
            value={customDays}
            onChange={(e) => onCustomDays(e.target.value)}
          />
        </div>
      )}
    </div>
  )
}

/** Classify a single reviewed face. */
export function FaceClassificationDialog({
  face,
  onClose,
  onDone,
}: {
  face: UnfamiliarFace
  onClose: () => void
  onDone: () => void
}) {
  const toast = useSentinelStore((s) => s.toast)
  const [category, setCategory] = useState<IdentityCategory>('PERMANENT')
  const [name, setName] = useState('')
  const [preset, setPreset] = useState('7d')
  const [customDays, setCustomDays] = useState('14')
  const [saving, setSaving] = useState(false)

  const submit = async () => {
    setSaving(true)
    try {
      const result = await api.classifyFace(face.id, {
        category,
        display_name: name.trim() || null,
        retention_preset: category === 'TEMPORARY' ? preset : null,
        retention_days:
          category === 'TEMPORARY' && preset === 'custom' ? Number(customDays) : null,
      })
      toast(
        'ok',
        `Identity created: ${(result.display_name as string) ?? (result.identifier as string)}`,
      )
      onDone()
      onClose()
    } catch (e) {
      toast('error', (e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <Modal
      title="Classify person"
      onClose={onClose}
      footer={
        <>
          <button className="btn ghost" onClick={onClose}>
            Cancel
          </button>
          <button className="btn primary" onClick={submit} disabled={saving}>
            {saving ? 'Saving…' : 'Save identity'}
          </button>
        </>
      }
    >
      <div className="row" style={{ alignItems: 'flex-start', gap: 14, marginBottom: 14 }}>
        {face.image_url && (
          <img
            src={face.image_url}
            alt="Detected face"
            style={{
              width: 120, height: 120, objectFit: 'cover',
              borderRadius: 6, border: '1px solid var(--border-strong)',
            }}
          />
        )}
        <div className="stack" style={{ fontSize: 11.5 }}>
          <div>
            <span className="muted">Detected: </span>
            <span className="mono">{fmtDateTime(face.detected_at)}</span>
          </div>
          <div>
            <span className="muted">Source: </span>
            {face.source_name ?? face.source_uid}
          </div>
          <div>
            <span className="muted">Quality: </span>
            <span className="mono">{(face.quality_score * 100).toFixed(0)}%</span>
            {face.face_pixels && (
              <span className="mono muted"> · {face.face_pixels}px</span>
            )}
          </div>
          <div>
            <span className="muted">Best match score: </span>
            <span className="mono">{face.match_score?.toFixed(3) ?? '—'}</span>
          </div>
        </div>
      </div>

      <CategoryPicker category={category} onChange={setCategory} />

      <div className="field">
        <label htmlFor="identity-name">Name (optional)</label>
        <input
          id="identity-name"
          className="input"
          placeholder="e.g. Mike"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <div className="hint">
          {name.trim()
            ? `Identity will be named "${name.trim()}".`
            : `Leave blank to use the generated identifier ${
                face.suggested_identifier ?? 'Unknown_YYYYMMDD_HHMMSS'
              }.`}
        </div>
      </div>

      {category === 'TEMPORARY' && (
        <RetentionPicker
          preset={preset}
          onPreset={setPreset}
          customDays={customDays}
          onCustomDays={setCustomDays}
        />
      )}
    </Modal>
  )
}

/** Classify several faces at once. */
export function BulkClassificationDialog({
  faces,
  onClose,
  onDone,
}: {
  faces: UnfamiliarFace[]
  onClose: () => void
  onDone: () => void
}) {
  const toast = useSentinelStore((s) => s.toast)
  const [category, setCategory] = useState<IdentityCategory>('TEMPORARY')
  const [name, setName] = useState('')
  const [preset, setPreset] = useState('7d')
  const [customDays, setCustomDays] = useState('14')
  const [merge, setMerge] = useState(false)
  const [saving, setSaving] = useState(false)

  const submit = async () => {
    setSaving(true)
    try {
      const results = await api.bulkClassify({
        face_ids: faces.map((f) => f.id),
        category,
        display_name: name.trim() || null,
        retention_preset: category === 'TEMPORARY' ? preset : null,
        retention_days:
          category === 'TEMPORARY' && preset === 'custom' ? Number(customDays) : null,
        merge_into_one_identity: merge,
      })
      toast('ok', `Classified ${results.length} face(s)`)
      onDone()
      onClose()
    } catch (e) {
      toast('error', (e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <Modal
      title={`Bulk classify — ${faces.length} face${faces.length === 1 ? '' : 's'}`}
      onClose={onClose}
      footer={
        <>
          <button className="btn ghost" onClick={onClose}>
            Cancel
          </button>
          <button className="btn primary" onClick={submit} disabled={saving}>
            {saving ? 'Saving…' : `Classify ${faces.length}`}
          </button>
        </>
      }
    >
      <div className="face-grid" style={{ marginBottom: 14, maxHeight: 180,
                                          overflowY: 'auto' }}>
        {faces.map((face) => (
          <div key={face.id} className="face-card">
            {face.image_url ? (
              <img src={face.image_url} alt="" />
            ) : (
              <div style={{ aspectRatio: '1', display: 'grid', placeItems: 'center',
                            background: 'var(--bg-3)' }}>
                no image
              </div>
            )}
            <div className="face-card-body">
              <div className="face-card-meta">{face.suggested_identifier}</div>
            </div>
          </div>
        ))}
      </div>

      <CategoryPicker category={category} onChange={setCategory} />

      <div className="field">
        <label htmlFor="bulk-name">Name (optional)</label>
        <input
          id="bulk-name"
          className="input"
          placeholder="Applied to every selected face"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <div className="hint">
          {name.trim()
            ? `All ${faces.length} identities will be named "${name.trim()}".`
            : 'Left blank, each face gets its own identifier from its detection time.'}
        </div>
      </div>

      {category === 'TEMPORARY' && (
        <RetentionPicker
          preset={preset}
          onPreset={setPreset}
          customDays={customDays}
          onCustomDays={setCustomDays}
        />
      )}

      <div className="card" style={{ borderColor: merge ? 'var(--warn-dim)' : undefined }}>
        <label className="row" style={{ alignItems: 'flex-start' }}>
          <input
            type="checkbox"
            checked={merge}
            onChange={(e) => setMerge(e.target.checked)}
            style={{ marginTop: 3 }}
          />
          <span>
            <strong>Merge into a single identity</strong>
            <div className="hint">
              Off by default. Only enable this when you are certain every selected
              face is the same person — merging distinct people corrupts the
              recognition dataset.
            </div>
          </span>
        </label>
      </div>
    </Modal>
  )
}
