// Top-of-workspace banners: active alarms and the review prompt.
import { useNavigate } from 'react-router-dom'

import { useSentinelStore } from '@/stores/useSentinelStore'

export function AlertBanner() {
  const alerts = useSentinelStore((s) => s.activeAlerts)
  const sources = useSentinelStore((s) => s.sources)
  const stopAlert = useSentinelStore((s) => s.stopAlert)
  const stopAll = useSentinelStore((s) => s.stopAllAlerts)
  const toast = useSentinelStore((s) => s.toast)

  const continuous = alerts.filter((a) => a.alert_type === 'CONTINUOUS_ALARM')
  if (continuous.length === 0) return null

  const nameFor = (sourceId: number) =>
    sources.find((s) => s.id === sourceId)?.name ?? `source ${sourceId}`

  const handleStop = async (id: number) => {
    try {
      await stopAlert(id)
    } catch (e) {
      toast('error', (e as Error).message)
    }
  }

  return (
    <div className="alert-banner">
      <span className="blink">⚠</span>
      <span>
        {continuous.length === 1
          ? `ALARM — ${continuous[0].reason.replaceAll('_', ' ')} on ${nameFor(
              continuous[0].source_id,
            )}`
          : `${continuous.length} ACTIVE ALARMS`}
      </span>
      <span className="spacer" />
      {continuous.length === 1 ? (
        <button className="btn" onClick={() => handleStop(continuous[0].id)}>
          STOP ALARM
        </button>
      ) : (
        <>
          {continuous.slice(0, 3).map((alert) => (
            <button
              key={alert.id}
              className="btn sm"
              onClick={() => handleStop(alert.id)}
            >
              Stop #{alert.id}
            </button>
          ))}
          <button className="btn" onClick={() => void stopAll()}>
            STOP ALL ALARMS
          </button>
        </>
      )}
    </div>
  )
}

export function ReviewBanner() {
  const navigate = useNavigate()
  const pendingReview = useSentinelStore((s) => s.pendingReview)
  const dismissed = useSentinelStore((s) => s.reviewGateDismissed)
  const dismiss = useSentinelStore((s) => s.dismissReviewGate)

  if (pendingReview === 0 || dismissed) return null

  return (
    <div className="alert-banner review">
      <span>👤</span>
      <span>
        {pendingReview} new unfamiliar face{pendingReview === 1 ? '' : 's'} require
        review.
      </span>
      <span className="spacer" />
      <button className="btn sm" onClick={() => navigate('/review')}>
        Review Now
      </button>
      <button className="btn ghost sm" onClick={dismiss}>
        Later
      </button>
    </div>
  )
}

export function Toasts() {
  const toasts = useSentinelStore((s) => s.toasts)
  const dismiss = useSentinelStore((s) => s.dismissToast)
  if (toasts.length === 0) return null
  return (
    <div className="toasts">
      {toasts.map((toast) => (
        <div key={toast.id} className={`toast ${toast.kind}`} role="status">
          <span style={{ flex: 1 }}>{toast.text}</span>
          <button className="btn ghost sm" onClick={() => dismiss(toast.id)}>
            ✕
          </button>
        </div>
      ))}
    </div>
  )
}
