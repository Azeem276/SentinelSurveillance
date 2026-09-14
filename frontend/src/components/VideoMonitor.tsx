// Live video tile: MJPEG image plus a canvas overlay driven by WebSocket data.
//
// Boxes are drawn client-side from structured overlay messages rather than
// burned into the video, so labels stay crisp and the stream stays cheap.
import { useEffect, useRef } from 'react'

import { api } from '@/services/api'
import { useSentinelStore } from '@/stores/useSentinelStore'
import type { OverlayObject, RecognitionState, VideoSource } from '@/types'

const STATE_COLOUR: Record<RecognitionState, string> = {
  PERMANENT_FAMILIAR: '#22c55e',
  TEMPORARY_FAMILIAR: '#f59e0b',
  UNFAMILIAR: '#ef4444',
  FACE_UNRECOGNIZABLE: '#94a3b8',
  UNKNOWN_PENDING_RECOGNITION: '#38bdf8',
  NO_FACE: '#64748b',
}

function colourFor(obj: OverlayObject): string {
  if (obj.alarm) return '#ef4444'
  if (obj.class !== 'person') return '#8b5cf6'
  return STATE_COLOUR[obj.recognition_state] ?? '#64748b'
}

function secondLine(obj: OverlayObject): string | null {
  if (obj.class !== 'person') {
    return obj.distance_m != null ? `~${obj.distance_m.toFixed(1)}m` : null
  }
  const zone =
    obj.proximity_zone === 'ZONE_B'
      ? 'ALARM ZONE'
      : obj.proximity_zone === 'ZONE_A'
        ? 'ZONE A'
        : obj.proximity_zone === 'FAR'
          ? 'FAR'
          : ''
  const state =
    obj.identity
      ? obj.recognition_state === 'PERMANENT_FAMILIAR' ? 'PERMANENT' : 'TEMPORARY'
      : obj.recognition_state === 'UNFAMILIAR'
        ? 'UNFAMILIAR'
        : obj.recognition_state === 'FACE_UNRECOGNIZABLE'
          ? 'UNRECOGNIZABLE'
          : obj.recognition_state === 'UNKNOWN_PENDING_RECOGNITION'
            ? 'PENDING RECOGNITION'
            : ''
  return [state, zone].filter(Boolean).join(' · ') || null
}

function drawOverlay(
  canvas: HTMLCanvasElement,
  objects: OverlayObject[],
  frameWidth: number,
  frameHeight: number,
  compact: boolean,
): void {
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const rect = canvas.getBoundingClientRect()
  const dpr = window.devicePixelRatio || 1
  if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
    canvas.width = Math.max(1, Math.round(rect.width * dpr))
    canvas.height = Math.max(1, Math.round(rect.height * dpr))
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, rect.width, rect.height)
  if (!frameWidth || !frameHeight || rect.width === 0) return

  // The <img> uses object-fit: contain, so replicate its letterboxing.
  const scale = Math.min(rect.width / frameWidth, rect.height / frameHeight)
  const offsetX = (rect.width - frameWidth * scale) / 2
  const offsetY = (rect.height - frameHeight * scale) / 2

  const fontSize = compact ? 10 : 12
  ctx.lineJoin = 'round'

  for (const obj of objects) {
    if (!obj.bbox) continue
    const colour = colourFor(obj)
    const x = offsetX + obj.bbox.x1 * scale
    const y = offsetY + obj.bbox.y1 * scale
    const w = (obj.bbox.x2 - obj.bbox.x1) * scale
    const h = (obj.bbox.y2 - obj.bbox.y1) * scale

    ctx.strokeStyle = colour
    ctx.lineWidth = obj.alarm ? 3 : 2
    ctx.strokeRect(x, y, w, h)

    const title = `${obj.label}${obj.class === 'person' ? '' : ` ${obj.class}`} #${obj.track_id}`
    const sub = compact ? null : secondLine(obj)

    ctx.font = `600 ${fontSize}px ui-monospace, Menlo, Consolas, monospace`
    const titleWidth = ctx.measureText(title).width
    const subWidth = sub ? ctx.measureText(sub).width : 0
    const boxWidth = Math.max(titleWidth, subWidth) + 10
    const lineH = fontSize + 4
    const boxHeight = sub ? lineH * 2 + 4 : lineH + 4
    const labelY = y - boxHeight < 0 ? y + h : y - boxHeight

    ctx.fillStyle = colour
    ctx.fillRect(x, labelY, boxWidth, boxHeight)
    ctx.fillStyle = obj.recognition_state === 'TEMPORARY_FAMILIAR' ? '#1a1206' : '#ffffff'
    ctx.fillText(title, x + 5, labelY + lineH - 3)
    if (sub) ctx.fillText(sub, x + 5, labelY + lineH * 2 - 2)
  }
}

interface Props {
  source: VideoSource
  compact?: boolean
  onClick?: () => void
  showOverlay?: boolean
}

export function VideoMonitor({ source, compact = false, onClick, showOverlay = true }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const overlay = useSentinelStore((s) => s.overlays[source.uid])
  const motion = useSentinelStore((s) => s.motion[source.uid])
  const live = useSentinelStore((s) => s.liveSources[source.uid])
  const activeAlerts = useSentinelStore((s) => s.activeAlerts)

  const running = live?.status
    ? ['AVAILABLE', 'STARTING'].includes(live.status)
    : source.runtime.running
  const recording = live?.recording ?? source.runtime.recording
  const intelligence = live?.intelligence ?? source.runtime.intelligence
  const alarming = activeAlerts.some(
    (a) => a.source_id === source.id && a.alert_type === 'CONTINUOUS_ALARM',
  )
  const status = live?.status ?? source.runtime.status ?? source.status

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    if (!showOverlay || !overlay) {
      const ctx = canvas.getContext('2d')
      ctx?.clearRect(0, 0, canvas.width, canvas.height)
      return
    }
    drawOverlay(canvas, overlay.objects ?? [], overlay.width, overlay.height, compact)
  }, [overlay, compact, showOverlay])

  // Redraw on resize so boxes stay aligned with the letterboxed image.
  useEffect(() => {
    const handler = () => {
      const canvas = canvasRef.current
      if (canvas && overlay) {
        drawOverlay(canvas, overlay.objects ?? [], overlay.width, overlay.height, compact)
      }
    }
    window.addEventListener('resize', handler)
    return () => window.removeEventListener('resize', handler)
  }, [overlay, compact])

  const frameClass = [
    'video-frame',
    motion && !alarming ? 'motion' : '',
    alarming ? 'alarm' : '',
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <div
      className={frameClass}
      onClick={onClick}
      style={onClick ? { cursor: 'pointer' } : undefined}
    >
      {running ? (
        <img
          src={api.streamUrl(source.id)}
          alt={`${source.name} live view`}
          onError={(e) => {
            ;(e.currentTarget as HTMLImageElement).style.visibility = 'hidden'
          }}
        />
      ) : (
        <div className="video-placeholder">
          <div style={{ fontSize: 26 }}>◼</div>
          <div>
            <strong>{source.name}</strong>
            <div className="muted" style={{ marginTop: 4 }}>
              {status === 'UNAVAILABLE'
                ? 'Source unavailable'
                : status === 'ENDED'
                  ? 'Clip ended'
                  : 'Surveillance inactive'}
            </div>
            {live?.error && (
              <div className="muted mono" style={{ marginTop: 4, fontSize: 10 }}>
                {live.error}
              </div>
            )}
          </div>
        </div>
      )}

      <canvas ref={canvasRef} />

      <div className="video-badge">
        <span className="badge neutral">{source.name}</span>
        {recording && <span className="ind rec blink">● REC</span>}
        {intelligence && <span className="ind intel">INTELLIGENCE</span>}
        {!intelligence && running && <span className="ind">INTEL OFF</span>}
      </div>

      <div className="video-badge video-badge-right">
        {motion && <span className="ind motion">MOTION</span>}
        {alarming && <span className="ind alert">ALARM</span>}
        {overlay && overlay.objects.length > 0 && (
          <span className="ind live">{overlay.objects.length} TRACKED</span>
        )}
      </div>
    </div>
  )
}
