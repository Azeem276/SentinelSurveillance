// Unfamiliar-face review: classify individually or in bulk.
import { useEffect, useState } from 'react'

import {
  BulkClassificationDialog, FaceClassificationDialog,
} from '@/components/FaceClassificationDialog'
import { MergeIdentityDialog } from '@/components/MergeIdentityDialog'
import { Empty, fmtDateTime, relativeTime } from '@/components/primitives'
import { api } from '@/services/api'
import { useSentinelStore } from '@/stores/useSentinelStore'
import type { UnfamiliarFace } from '@/types'

export function ReviewPage() {
  const faces = useSentinelStore((s) => s.unfamiliarFaces)
  const refreshReview = useSentinelStore((s) => s.refreshReview)
  const refreshIdentities = useSentinelStore((s) => s.refreshIdentities)
  const toast = useSentinelStore((s) => s.toast)

  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [classifying, setClassifying] = useState<UnfamiliarFace | null>(null)
  const [merging, setMerging] = useState<UnfamiliarFace | null>(null)
  const [bulkOpen, setBulkOpen] = useState(false)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    void refreshReview().finally(() => setLoading(false))
  }, [refreshReview])

  const toggle = (id: number) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const afterChange = async () => {
    setSelected(new Set())
    await Promise.all([refreshReview(), refreshIdentities()])
  }

  const dismiss = async (face: UnfamiliarFace) => {
    try {
      await api.dismissFace(face.id)
      toast('ok', 'Face dismissed')
      await afterChange()
    } catch (e) {
      toast('error', (e as Error).message)
    }
  }

  const quickClassify = async (face: UnfamiliarFace, category: 'PERMANENT' | 'TEMPORARY') => {
    try {
      const result = await api.classifyFace(face.id, {
        category,
        display_name: null,
        retention_preset: category === 'TEMPORARY' ? '7d' : null,
      })
      toast('ok', `Created ${result.identifier as string}`)
      await afterChange()
    } catch (e) {
      toast('error', (e as Error).message)
    }
  }

  const selectedFaces = faces.filter((f) => selected.has(f.id))

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1 className="page-title">Unfamiliar face review</h1>
          <div className="page-sub">
            Classifying a face adds its embedding to the recognition dataset
            immediately — the next frame uses it.
          </div>
        </div>
        <div className="row">
          <button
            className="btn"
            disabled={selected.size === 0}
            onClick={() => setBulkOpen(true)}
          >
            Bulk classify ({selected.size})
          </button>
          <button
            className="btn ghost"
            onClick={() =>
              setSelected(
                selected.size === faces.length
                  ? new Set()
                  : new Set(faces.map((f) => f.id)),
              )
            }
            disabled={faces.length === 0}
          >
            {selected.size === faces.length && faces.length > 0
              ? 'Clear selection'
              : 'Select all'}
          </button>
          <button className="btn ghost" onClick={() => void refreshReview()}>
            Refresh
          </button>
        </div>
      </div>

      {loading && faces.length === 0 && <Empty>Loading…</Empty>}

      {!loading && faces.length === 0 && (
        <div className="card">
          <Empty>
            No unfamiliar faces awaiting review. Detected unknown people will
            appear here automatically.
          </Empty>
        </div>
      )}

      {faces.length > 0 && (
        <div className="face-grid">
          {faces.map((face) => (
            <div
              key={face.id}
              className={`face-card ${selected.has(face.id) ? 'selected' : ''}`}
            >
              <input
                type="checkbox"
                className="face-card-check"
                checked={selected.has(face.id)}
                onChange={() => toggle(face.id)}
                aria-label={`Select face ${face.id}`}
              />
              {face.image_url ? (
                <img src={face.image_url} alt={`Unfamiliar face ${face.id}`} />
              ) : (
                <div
                  style={{
                    aspectRatio: '1', display: 'grid', placeItems: 'center',
                    background: 'var(--bg-3)', color: 'var(--text-2)',
                  }}
                >
                  no crop
                </div>
              )}
              <div className="face-card-body">
                <div style={{ fontSize: 11, fontWeight: 600 }}>
                  {face.suggested_identifier}
                </div>
                <div className="face-card-meta" title={fmtDateTime(face.detected_at)}>
                  {relativeTime(face.detected_at)}
                </div>
                <div className="face-card-meta">
                  {face.source_name ?? face.source_uid}
                </div>
                <div className="face-card-meta">
                  q {(face.quality_score * 100).toFixed(0)}%
                  {face.face_pixels ? ` · ${face.face_pixels}px` : ''}
                </div>
                {!!face.profile_samples && (
                  <div
                    className="face-card-meta"
                    title={
                      `${face.profile_samples} additional views of this face were ` +
                      'collected across different angles and lighting. All of them ' +
                      'are enrolled when you classify or merge this person.'
                    }
                  >
                    +{face.profile_samples} angles
                  </div>
                )}
                <div className="face-card-actions">
                  <button
                    className="btn success sm"
                    title="Permanent familiar, unnamed"
                    onClick={() => void quickClassify(face, 'PERMANENT')}
                  >
                    Perm
                  </button>
                  <button
                    className="btn sm"
                    style={{ background: 'var(--warn)', borderColor: 'var(--warn)',
                             color: '#231602' }}
                    title="Temporary familiar, unnamed, 7 days"
                    onClick={() => void quickClassify(face, 'TEMPORARY')}
                  >
                    Temp
                  </button>
                  <button
                    className="btn ghost sm"
                    onClick={() => setClassifying(face)}
                    title="Classify with a name"
                  >
                    Name…
                  </button>
                </div>
                <div className="face-card-actions" style={{ marginTop: 4 }}>
                  <button
                    className="btn ghost sm"
                    onClick={() => setMerging(face)}
                    title="This is somebody already enrolled — add this face to them"
                  >
                    Merge…
                  </button>
                  <button
                    className="btn ghost sm"
                    onClick={() => void dismiss(face)}
                  >
                    Dismiss
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {classifying && (
        <FaceClassificationDialog
          face={classifying}
          onClose={() => setClassifying(null)}
          onDone={() => void afterChange()}
        />
      )}
      {merging && (
        <MergeIdentityDialog
          face={merging}
          onClose={() => setMerging(null)}
          onDone={() => void afterChange()}
        />
      )}
      {bulkOpen && selectedFaces.length > 0 && (
        <BulkClassificationDialog
          faces={selectedFaces}
          onClose={() => setBulkOpen(false)}
          onDone={() => void afterChange()}
        />
      )}
    </div>
  )
}
