// Familiar-face dataset management.
import { useEffect, useMemo, useState } from 'react'

import { CategoryBadge, Empty, Modal, fmtDateTime, relativeTime } from '@/components/primitives'
import { api } from '@/services/api'
import { useSentinelStore } from '@/stores/useSentinelStore'
import type { Identity, IdentityCategory } from '@/types'

function EditDialog({
  identity,
  onClose,
  onSaved,
}: {
  identity: Identity
  onClose: () => void
  onSaved: () => void
}) {
  const toast = useSentinelStore((s) => s.toast)
  const [name, setName] = useState(identity.display_name ?? '')
  const [category, setCategory] = useState<IdentityCategory>(identity.category)
  const [retention, setRetention] = useState(identity.retention_days ?? 7)
  const [notes, setNotes] = useState(identity.notes ?? '')
  const [saving, setSaving] = useState(false)

  const save = async () => {
    setSaving(true)
    try {
      await api.updateIdentity(identity.id, {
        display_name: name.trim() || null,
        category,
        retention_days: category === 'TEMPORARY' ? retention : null,
        notes: notes.trim() || null,
      })
      toast('ok', 'Identity updated; recognition index rebuilt')
      onSaved()
      onClose()
    } catch (e) {
      toast('error', (e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <Modal
      title={`Edit — ${identity.label}`}
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
        <label htmlFor="ed-name">Display name</label>
        <input
          id="ed-name"
          className="input"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder={identity.generated_identifier}
        />
        <div className="hint">
          Blank falls back to the generated identifier{' '}
          <span className="mono">{identity.generated_identifier}</span>.
        </div>
      </div>

      <div className="field">
        <label htmlFor="ed-cat">Category</label>
        <select
          id="ed-cat"
          className="select"
          value={category}
          onChange={(e) => setCategory(e.target.value as IdentityCategory)}
        >
          <option value="PERMANENT">Permanent familiar</option>
          <option value="TEMPORARY">Temporary familiar</option>
        </select>
      </div>

      {category === 'TEMPORARY' && (
        <div className="field">
          <label htmlFor="ed-ret">Retention</label>
          <select
            id="ed-ret"
            className="select"
            value={retention}
            onChange={(e) => setRetention(Number(e.target.value))}
          >
            <option value={1}>24 hours</option>
            <option value={3}>3 days</option>
            <option value={7}>7 days</option>
            <option value={30}>30 days</option>
          </select>
          <div className="hint">Re-saving restarts the retention window from now.</div>
        </div>
      )}

      <div className="field">
        <label htmlFor="ed-notes">Notes</label>
        <textarea
          id="ed-notes"
          className="textarea"
          rows={3}
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
        />
      </div>
    </Modal>
  )
}

/** Fold one identity into another when the same person has two records. */
function MergeIdentitiesDialog({
  identity,
  onClose,
  onDone,
}: {
  identity: Identity
  onClose: () => void
  onDone: () => void
}) {
  const toast = useSentinelStore((s) => s.toast)
  const identities = useSentinelStore((s) => s.identities)
  const [target, setTarget] = useState<number | null>(null)
  const [filter, setFilter] = useState('')
  const [saving, setSaving] = useState(false)

  const options = useMemo(() => {
    const needle = filter.trim().toLowerCase()
    return identities
      .filter((i) => i.status === 'ACTIVE' && i.id !== identity.id)
      .filter(
        (i) =>
          !needle ||
          (i.display_name ?? '').toLowerCase().includes(needle) ||
          i.generated_identifier.toLowerCase().includes(needle),
      )
  }, [identities, identity.id, filter])

  const chosen = options.find((i) => i.id === target) ?? null

  const submit = async () => {
    if (target === null) return
    setSaving(true)
    try {
      const result = await api.mergeIdentities(identity.id, target)
      const moved = (result.data as { embeddings_moved?: number }).embeddings_moved ?? 0
      toast('ok', `Merged — ${moved} embedding(s) moved`)
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
      title="Merge identities"
      onClose={onClose}
      footer={
        <>
          <button className="btn ghost" onClick={onClose}>
            Cancel
          </button>
          <button
            className="btn primary"
            onClick={submit}
            disabled={saving || target === null}
          >
            {saving ? 'Merging…' : 'Merge'}
          </button>
        </>
      }
    >
      <p style={{ fontSize: 12.5, marginTop: 0 }}>
        Move everything from{' '}
        <strong>{identity.display_name ?? identity.generated_identifier}</strong>{' '}
        ({identity.embedding_count} embeddings) into another identity, which
        survives.
      </p>
      <div className="hint" style={{ marginBottom: 12 }}>
        Embeddings, stored face crops and historical tracks all move across.
        The merged gallery covers more angles than either record did alone,
        which is the point. This cannot be undone automatically.
      </div>

      <div className="field">
        <label htmlFor="merge-target">Merge into</label>
        <input
          id="merge-target"
          className="input"
          placeholder="Search identities…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
      </div>

      <div
        style={{
          maxHeight: 240, overflowY: 'auto', border: '1px solid var(--border)',
          borderRadius: 6,
        }}
      >
        {options.length === 0 && (
          <div className="empty" style={{ padding: 16 }}>
            No other active identities to merge into.
          </div>
        )}
        {options.map((other) => (
          <label
            key={other.id}
            className="row"
            style={{
              alignItems: 'center', gap: 10, padding: '8px 10px', cursor: 'pointer',
              borderBottom: '1px solid var(--border)',
              background: target === other.id ? 'var(--bg-3)' : undefined,
            }}
          >
            <input
              type="radio"
              name="merge-target-identity"
              checked={target === other.id}
              onChange={() => setTarget(other.id)}
            />
            <span style={{ flex: 1, minWidth: 0 }}>
              <span style={{ fontWeight: 600, fontSize: 12 }}>
                {other.display_name ?? other.generated_identifier}
              </span>
              <span className="muted mono" style={{ display: 'block', fontSize: 10 }}>
                {other.category} · {other.embedding_count} embeddings
              </span>
            </span>
          </label>
        ))}
      </div>

      {chosen && (
        <div className="card" style={{ marginTop: 12 }}>
          <span style={{ fontSize: 12 }}>
            Result: <strong>{chosen.display_name ?? chosen.generated_identifier}</strong>{' '}
            with about{' '}
            <span className="mono">
              {chosen.embedding_count + identity.embedding_count}
            </span>{' '}
            embeddings.{' '}
            <span className="muted">
              {identity.display_name ?? identity.generated_identifier} is removed
              from recognition; its history is kept.
            </span>
          </span>
        </div>
      )}
    </Modal>
  )
}

export function IdentitiesPage() {
  const identities = useSentinelStore((s) => s.identities)
  const refreshIdentities = useSentinelStore((s) => s.refreshIdentities)
  const toast = useSentinelStore((s) => s.toast)

  const [filter, setFilter] = useState('')
  const [category, setCategory] = useState<'' | IdentityCategory>('')
  const [statusFilter, setStatusFilter] = useState('ACTIVE')
  const [editing, setEditing] = useState<Identity | null>(null)
  const [merging, setMerging] = useState<Identity | null>(null)
  const [rows, setRows] = useState<Identity[]>([])
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    void api
      .identities({
        category: category || undefined,
        status: statusFilter,
      })
      .then(setRows)
      .catch((e) => toast('error', (e as Error).message))
  }, [category, statusFilter, identities, toast])

  const visible = useMemo(() => {
    const needle = filter.trim().toLowerCase()
    if (!needle) return rows
    return rows.filter(
      (i) =>
        (i.display_name ?? '').toLowerCase().includes(needle) ||
        i.generated_identifier.toLowerCase().includes(needle),
    )
  }, [rows, filter])

  const remove = async (identity: Identity) => {
    if (
      !window.confirm(
        `Remove "${identity.label}" from recognition?\n\n` +
          'Historical detections and events are preserved; the identity simply ' +
          'stops matching future faces.',
      )
    ) {
      return
    }
    try {
      await api.deleteIdentity(identity.id)
      toast('ok', `${identity.label} removed from the recognition index`)
      await refreshIdentities()
    } catch (e) {
      toast('error', (e as Error).message)
    }
  }

  const runAction = async (label: string, fn: () => Promise<unknown>) => {
    setBusy(true)
    try {
      const result = (await fn()) as { data?: Record<string, unknown> }
      toast('ok', `${label}: ${JSON.stringify(result.data ?? {})}`)
      await refreshIdentities()
    } catch (e) {
      toast('error', (e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1 className="page-title">Familiar faces</h1>
          <div className="page-sub">
            Permanent identities never expire. Temporary identities are expired by
            a backend scheduler and stop matching immediately.
          </div>
        </div>
        <div className="row">
          <button
            className="btn ghost"
            disabled={busy}
            onClick={() => void runAction('Dataset enrolled', api.enrolDataset)}
            title="Enrol data/familiar_faces/{permanent,temporary}/<Name>/*.jpg"
          >
            Enrol dataset folder
          </button>
          <button
            className="btn ghost"
            disabled={busy}
            onClick={() => void runAction('Index rebuilt', api.syncIndex)}
          >
            Rebuild index
          </button>
          <button
            className="btn ghost"
            disabled={busy}
            onClick={() => void runAction('Expiration sweep', api.expireNow)}
          >
            Run expiry now
          </button>
        </div>
      </div>

      <div className="card">
        <div className="row" style={{ marginBottom: 12 }}>
          <input
            className="input"
            style={{ maxWidth: 260 }}
            placeholder="Search identities…"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
          <select
            className="select"
            style={{ width: 'auto' }}
            value={category}
            onChange={(e) => setCategory(e.target.value as '' | IdentityCategory)}
          >
            <option value="">All categories</option>
            <option value="PERMANENT">Permanent</option>
            <option value="TEMPORARY">Temporary</option>
          </select>
          <select
            className="select"
            style={{ width: 'auto' }}
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
          >
            <option value="ACTIVE">Active</option>
            <option value="EXPIRED">Expired</option>
            <option value="DELETED">Deleted</option>
          </select>
          <span className="spacer" />
          <span className="muted mono">{visible.length} shown</span>
        </div>

        {visible.length === 0 ? (
          <Empty>No identities match these filters.</Empty>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th />
                <th>Identity</th>
                <th>Identifier</th>
                <th>Category</th>
                <th>Status</th>
                <th>Embeddings</th>
                <th>Appearances</th>
                <th>Expires</th>
                <th>Last seen</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {visible.map((identity) => (
                <tr key={identity.id}>
                  <td>
                    {identity.thumbnail_url ? (
                      <img
                        src={identity.thumbnail_url}
                        alt=""
                        style={{
                          width: 34, height: 34, borderRadius: 4,
                          objectFit: 'cover', display: 'block',
                        }}
                      />
                    ) : (
                      <div
                        style={{
                          width: 34, height: 34, borderRadius: 4,
                          background: 'var(--bg-3)',
                        }}
                      />
                    )}
                  </td>
                  <td>
                    <strong>{identity.display_name ?? <span className="muted">unnamed</span>}</strong>
                  </td>
                  <td className="mono" style={{ fontSize: 11 }}>
                    {identity.generated_identifier}
                  </td>
                  <td>
                    <CategoryBadge category={identity.category} />
                  </td>
                  <td>
                    <span
                      className={`badge ${
                        identity.status === 'ACTIVE'
                          ? 'permanent'
                          : identity.status === 'EXPIRED'
                            ? 'unrecognizable'
                            : 'neutral'
                      }`}
                    >
                      {identity.status}
                    </span>
                  </td>
                  <td className="mono">{identity.embedding_count}</td>
                  <td className="mono">{identity.track_count}</td>
                  <td className="mono" style={{ fontSize: 11 }}>
                    {identity.expires_at ? fmtDateTime(identity.expires_at) : '—'}
                  </td>
                  <td className="mono" style={{ fontSize: 11 }}>
                    {relativeTime(identity.last_seen_at)}
                  </td>
                  <td>
                    <div className="row">
                      <button
                        className="btn ghost sm"
                        onClick={() => setEditing(identity)}
                      >
                        Edit
                      </button>
                      <button
                        className="btn ghost sm"
                        onClick={() => setMerging(identity)}
                        disabled={identity.status !== 'ACTIVE'}
                        title="This is the same person as another identity — combine them"
                      >
                        Merge…
                      </button>
                      <button
                        className="btn ghost sm"
                        onClick={() => void remove(identity)}
                      >
                        Remove
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {editing && (
        <EditDialog
          identity={editing}
          onClose={() => setEditing(null)}
          onSaved={() => void refreshIdentities()}
        />
      )}
      {merging && (
        <MergeIdentitiesDialog
          identity={merging}
          onClose={() => setMerging(null)}
          onDone={() => void refreshIdentities()}
        />
      )}
    </div>
  )
}
