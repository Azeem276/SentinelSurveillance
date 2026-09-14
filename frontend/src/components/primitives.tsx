// Small shared presentational pieces. Kept together because each is a few
// lines and they are used everywhere.
import type { ReactNode } from 'react'

import type { EventSeverity, IdentityCategory, ProximityZone, RecognitionState } from '@/types'

export function StatusChip({
  label,
  state,
  title,
}: {
  label: string
  state: 'on' | 'off' | 'warn' | 'danger'
  title?: string
}) {
  return (
    <span className={`status-chip ${state}`} title={title}>
      <span className="dot" />
      {label}
    </span>
  )
}

/** Colour semantics required by the brief: green/orange/alert. */
export function RecognitionBadge({
  state,
  label,
}: {
  state: RecognitionState
  label?: string | null
}) {
  const map: Record<RecognitionState, { cls: string; text: string }> = {
    PERMANENT_FAMILIAR: { cls: 'permanent', text: 'PERMANENT' },
    TEMPORARY_FAMILIAR: { cls: 'temporary', text: 'TEMPORARY' },
    UNFAMILIAR: { cls: 'unfamiliar', text: 'UNFAMILIAR' },
    FACE_UNRECOGNIZABLE: { cls: 'unrecognizable', text: 'UNRECOGNIZABLE' },
    UNKNOWN_PENDING_RECOGNITION: { cls: 'pending', text: 'PENDING' },
    NO_FACE: { cls: 'neutral', text: 'NO FACE' },
  }
  const item = map[state] ?? map.NO_FACE
  return <span className={`badge ${item.cls}`}>{label ?? item.text}</span>
}

export function CategoryBadge({ category }: { category: IdentityCategory | string | null }) {
  if (!category) return <span className="badge neutral">—</span>
  const cls = category === 'PERMANENT' ? 'permanent' : 'temporary'
  return <span className={`badge ${cls}`}>{category}</span>
}

export function ProximityIndicator({
  zone,
  distance,
}: {
  zone: ProximityZone
  distance?: number | null
}) {
  const cls =
    zone === 'ZONE_B' ? 'zone-b' : zone === 'ZONE_A' ? 'zone-a' : 'neutral'
  const text =
    zone === 'ZONE_B' ? 'ALARM ZONE' : zone === 'ZONE_A' ? 'RECOGNITION ZONE'
      : zone === 'FAR' ? 'FAR' : 'UNKNOWN'
  return (
    <span className={`badge ${cls}`}>
      {text}
      {distance != null && <span className="mono"> ~{distance.toFixed(1)}m</span>}
    </span>
  )
}

export function SeverityBadge({ severity }: { severity: EventSeverity | string }) {
  const cls = String(severity).toLowerCase()
  return <span className={`badge ${cls}`}>{severity}</span>
}

export function Stat({
  value,
  label,
  tone,
}: {
  value: ReactNode
  label: string
  tone?: 'ok' | 'warn' | 'danger'
}) {
  return (
    <div className={`stat ${tone ?? ''}`}>
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  )
}

export function KeyValue({ k, v }: { k: string; v: ReactNode }) {
  return (
    <div className="kv-row">
      <span className="kv-key">{k}</span>
      <span className="kv-val">{v}</span>
    </div>
  )
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>
}

export function Modal({
  title,
  onClose,
  children,
  footer,
}: {
  title: string
  onClose: () => void
  children: ReactNode
  footer?: ReactNode
}) {
  return (
    <div
      className="modal-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
      role="dialog"
      aria-modal="true"
      aria-label={title}
    >
      <div className="modal">
        <div className="modal-header">
          <span>{title}</span>
          <button className="btn ghost sm" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>
        <div className="modal-body">{children}</div>
        {footer && <div className="modal-footer">{footer}</div>}
      </div>
    </div>
  )
}

export const fmtTime = (iso: string | null | undefined): string => {
  if (!iso) return '—'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '—' : d.toLocaleTimeString([], {
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  })
}

export const fmtDateTime = (iso: string | null | undefined): string => {
  if (!iso) return '—'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '—' : d.toLocaleString()
}

export const fmtDuration = (seconds: number | null | undefined): string => {
  if (seconds == null) return '—'
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  if (m < 60) return `${m}m ${s}s`
  return `${Math.floor(m / 60)}h ${m % 60}m`
}

export const fmtBytes = (bytes: number | null | undefined): string => {
  if (!bytes) return '—'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let value = bytes
  let i = 0
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024
    i += 1
  }
  return `${value.toFixed(i === 0 ? 0 : 1)} ${units[i]}`
}

export const relativeTime = (iso: string | null | undefined): string => {
  if (!iso) return '—'
  const diff = Date.now() - new Date(iso).getTime()
  if (Number.isNaN(diff)) return '—'
  const s = Math.floor(diff / 1000)
  if (s < 60) return `${s}s ago`
  if (s < 3600) return `${Math.floor(s / 60)}m ago`
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`
  return `${Math.floor(s / 86400)}d ago`
}
