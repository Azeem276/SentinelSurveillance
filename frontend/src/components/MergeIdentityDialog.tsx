// Merge a reviewed face into an identity that already exists.
//
// This exists for the commonest review mistake: somebody already enrolled is
// detected as a new person, usually because the camera caught them at an
// angle their stored gallery does not cover. Creating yet another identity
// makes that worse - the person fragments across several records, none of
// which covers enough angles to recognise them reliably. Merging instead adds
// this appearance to the identity the operator knows it is, which corrects
// the record AND teaches the recogniser the angle it just failed on.
//
// Suggestions are ranked below the live recognition threshold on purpose:
// they are a prompt for a human decision, never an automatic match.
import { useEffect, useMemo, useState } from 'react'

import { api } from '@/services/api'
import { useSentinelStore } from '@/stores/useSentinelStore'
import type { Identity, MergeCandidate, UnfamiliarFace } from '@/types'
import { Modal, fmtDateTime } from './primitives'

/** How confident a suggestion is, in words an operator can act on. */
function confidenceLabel(score: number): { text: string; color: string } {
  if (score >= 0.5) return { text: 'strong match', color: 'var(--ok)' }
  if (score >= 0.35) return { text: 'possible match', color: 'var(--warn)' }
  return { text: 'weak match', color: 'var(--text-2)' }
}

export function MergeIdentityDialog({
  face,
  onClose,
  onDone,
}: {
  face: UnfamiliarFace
  onClose: () => void
  onDone: () => void
}) {
  const toast = useSentinelStore((s) => s.toast)
  const identities = useSentinelStore((s) => s.identities)
  const refreshIdentities = useSentinelStore((s) => s.refreshIdentities)

  const [candidates, setCandidates] = useState<MergeCandidate[]>([])
  const [loadingCandidates, setLoadingCandidates] = useState(true)
  const [selected, setSelected] = useState<number | null>(null)
  const [filter, setFilter] = useState('')
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    let cancelled = false
    setLoadingCandidates(true)
    void api
      .mergeCandidates(face.id)
      .then((rows) => {
        if (cancelled) return
        setCandidates(rows)
        // Pre-select only a genuinely strong suggestion. Anything weaker is
        // shown but left unselected so nobody merges by reflex.
        if (rows.length > 0 && rows[0].mean_score >= 0.5) {
          setSelected(rows[0].identity_id)
        }
      })
      .catch(() => {
        /* suggestions are a convenience; the full list still works */
      })
      .finally(() => !cancelled && setLoadingCandidates(false))
    if (identities.length === 0) void refreshIdentities()
    return () => {
      cancelled = true
    }
  }, [face.id, identities.length, refreshIdentities])

  const scoreById = useMemo(() => {
    const map = new Map<number, MergeCandidate>()
    for (const c of candidates) map.set(c.identity_id, c)
    return map
  }, [candidates])

  const familiar = useMemo(() => {
    const needle = filter.trim().toLowerCase()
    const list = identities.filter((i) => i.status === 'ACTIVE')
    const matching = needle
      ? list.filter(
          (i) =>
            (i.display_name ?? '').toLowerCase().includes(needle) ||
            i.generated_identifier.toLowerCase().includes(needle),
        )
      : list
    // Suggested identities float to the top, then the rest alphabetically.
    return [...matching].sort((a, b) => {
      const sa = scoreById.get(a.id)?.mean_score ?? -1
      const sb = scoreById.get(b.id)?.mean_score ?? -1
      if (sa !== sb) return sb - sa
      return (a.display_name ?? a.generated_identifier).localeCompare(
        b.display_name ?? b.generated_identifier,
      )
    })
  }, [identities, filter, scoreById])

  const chosen = selected !== null ? identities.find((i) => i.id === selected) : null

  const submit = async () => {
    if (selected === null) return
    setSaving(true)
    try {
      await api.mergeFace(face.id, selected)
      toast(
        'ok',
        `Merged into ${chosen?.display_name ?? chosen?.generated_identifier ?? 'identity'}`,
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
      title="Merge with existing identity"
      onClose={onClose}
      footer={
        <>
          <button className="btn ghost" onClick={onClose}>
            Cancel
          </button>
          <button
            className="btn primary"
            onClick={submit}
            disabled={saving || selected === null}
            title={
              selected === null ? 'Choose the person this face belongs to' : undefined
            }
          >
            {saving ? 'Merging…' : 'Merge'}
          </button>
        </>
      }
    >
      <div className="row" style={{ alignItems: 'flex-start', gap: 14, marginBottom: 12 }}>
        {face.image_url && (
          <img
            src={face.image_url}
            alt="Detected face"
            style={{
              width: 108, height: 108, objectFit: 'cover',
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
          {!!face.profile_samples && (
            <div>
              <span className="muted">Angles collected: </span>
              <span className="mono">{face.profile_samples}</span>
            </div>
          )}
          <div className="hint" style={{ marginTop: 4 }}>
            Merging adds this face — and every angle collected with it — to the
            identity you pick. Its category and retention are left unchanged.
          </div>
        </div>
      </div>

      {loadingCandidates && (
        <div className="hint" style={{ marginBottom: 8 }}>
          Comparing against enrolled identities…
        </div>
      )}
      {!loadingCandidates && candidates.length === 0 && (
        <div className="hint" style={{ marginBottom: 8 }}>
          No close matches found. Pick from the full list if you recognise them
          anyway — you can see things the recogniser cannot.
        </div>
      )}

      <div className="field">
        <label htmlFor="merge-filter">Identity</label>
        <input
          id="merge-filter"
          className="input"
          placeholder="Search familiar identities…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
      </div>

      <div
        style={{
          maxHeight: 280, overflowY: 'auto', border: '1px solid var(--border)',
          borderRadius: 6,
        }}
      >
        {familiar.length === 0 && (
          <div className="empty" style={{ padding: 16 }}>
            {identities.length === 0
              ? 'No familiar identities yet. Classify this face as a new person instead.'
              : 'No identities match the filter.'}
          </div>
        )}
        {familiar.map((identity: Identity) => {
          const candidate = scoreById.get(identity.id)
          const isSelected = selected === identity.id
          return (
            <label
              key={identity.id}
              className="row"
              style={{
                alignItems: 'center',
                gap: 10,
                padding: '8px 10px',
                cursor: 'pointer',
                borderBottom: '1px solid var(--border)',
                background: isSelected ? 'var(--bg-3)' : undefined,
              }}
            >
              <input
                type="radio"
                name="merge-identity"
                checked={isSelected}
                onChange={() => setSelected(identity.id)}
              />
              {identity.thumbnail_url ? (
                <img
                  src={identity.thumbnail_url}
                  alt=""
                  style={{
                    width: 36, height: 36, objectFit: 'cover', borderRadius: 4,
                    border: '1px solid var(--border-strong)',
                  }}
                />
              ) : (
                <div
                  style={{
                    width: 36, height: 36, borderRadius: 4, background: 'var(--bg-3)',
                    display: 'grid', placeItems: 'center', fontSize: 9,
                    color: 'var(--text-2)',
                  }}
                >
                  n/a
                </div>
              )}
              <span style={{ flex: 1, minWidth: 0 }}>
                <span style={{ fontWeight: 600, fontSize: 12 }}>
                  {identity.display_name ?? identity.generated_identifier}
                </span>
                <span className="muted mono" style={{ display: 'block', fontSize: 10 }}>
                  {identity.category} · {identity.embedding_count ?? 0} embeddings
                </span>
              </span>
              {candidate && (
                <span
                  className="mono"
                  style={{ fontSize: 10, color: confidenceLabel(candidate.mean_score).color }}
                  title={`Mean ${candidate.mean_score.toFixed(3)} over ${candidate.samples_compared} sample(s), best ${candidate.best_score.toFixed(3)}`}
                >
                  {confidenceLabel(candidate.mean_score).text} ·{' '}
                  {candidate.mean_score.toFixed(2)}
                </span>
              )}
            </label>
          )
        })}
      </div>
    </Modal>
  )
}
