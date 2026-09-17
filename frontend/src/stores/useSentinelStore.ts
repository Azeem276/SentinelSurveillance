// Central client state.
//
// Live data arrives over the WebSocket and is merged here; REST is used only
// for the initial load and for mutations. Components subscribe to slices, so
// a 10 Hz overlay update does not re-render the identity list.
import { create } from 'zustand'

import { api } from '@/services/api'
import { alertAudio } from '@/services/alertAudio'
import type {
  Alert, FrameAnalysis, Identity, SystemState, UnfamiliarFace, VideoSource, WsMessage,
} from '@/types'

export type ViewMode = 'single' | 'grid'

interface LiveSourceState {
  status?: string
  recording?: boolean
  intelligence?: boolean
  motion?: boolean
  active_tracks?: number
  error?: string | null
}

interface SentinelState {
  // connection
  connected: boolean
  lastError: string | null

  // system
  system: SystemState | null
  sources: VideoSource[]
  selectedSourceId: number | null
  viewMode: ViewMode

  // realtime
  overlays: Record<string, FrameAnalysis>
  motion: Record<string, boolean>
  liveSources: Record<string, LiveSourceState>

  // security
  activeAlerts: Alert[]
  pendingReview: number
  reviewGateDismissed: boolean
  /** Backend-owned: alarms still fire when false, they just stay silent. */
  alarmSound: boolean

  // review / identities
  unfamiliarFaces: UnfamiliarFace[]
  identities: Identity[]

  // toasts
  toasts: { id: number; kind: 'info' | 'warn' | 'error' | 'ok'; text: string }[]

  // ---- actions
  setConnected: (v: boolean) => void
  setViewMode: (m: ViewMode) => void
  selectSource: (id: number | null) => void
  handleMessage: (msg: WsMessage) => void
  refreshSources: () => Promise<void>
  refreshSystem: () => Promise<void>
  refreshAlerts: () => Promise<void>
  refreshReview: () => Promise<void>
  refreshIdentities: () => Promise<void>
  refreshAll: () => Promise<void>
  stopAlert: (id: number) => Promise<void>
  stopAllAlerts: () => Promise<void>
  setAlarmSound: (enabled: boolean) => Promise<void>
  refreshAlarmSound: () => Promise<void>
  dismissReviewGate: () => void
  toast: (kind: 'info' | 'warn' | 'error' | 'ok', text: string) => void
  dismissToast: (id: number) => void
}

let toastSeq = 0

export const useSentinelStore = create<SentinelState>((set, get) => ({
  connected: false,
  lastError: null,
  system: null,
  sources: [],
  selectedSourceId: null,
  viewMode: 'single',
  overlays: {},
  motion: {},
  liveSources: {},
  activeAlerts: [],
  pendingReview: 0,
  reviewGateDismissed: false,
  alarmSound: true,
  unfamiliarFaces: [],
  identities: [],
  toasts: [],

  setConnected: (v) => set({ connected: v }),
  setViewMode: (m) => set({ viewMode: m }),
  selectSource: (id) => set({ selectedSourceId: id }),

  toast: (kind, text) => {
    const id = ++toastSeq
    set((s) => ({ toasts: [...s.toasts.slice(-4), { id, kind, text }] }))
    window.setTimeout(() => get().dismissToast(id), kind === 'error' ? 8000 : 4500)
  },
  dismissToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),

  dismissReviewGate: () => set({ reviewGateDismissed: true }),

  handleMessage: (msg) => {
    switch (msg.type) {
      case 'snapshot': {
        const snap = msg as Extract<WsMessage, { type: 'snapshot' }>
        const sound = (snap as unknown as { alarm_sound?: boolean }).alarm_sound
        if (sound !== undefined) {
          set({ alarmSound: sound })
          alertAudio.setMuted(!sound)
        }
        set({ system: snap.system, pendingReview: snap.pending_review })
        const ids = (snap.active_alerts as { id: number; alert_type?: string }[])
          .filter((a) => a.alert_type === 'CONTINUOUS_ALARM')
          .map((a) => a.id)
        alertAudio.syncActiveAlarms(ids)
        void get().refreshAlerts()
        break
      }
      case 'frame_analysis': {
        const frame = msg as unknown as FrameAnalysis
        set((s) => ({
          overlays: { ...s.overlays, [frame.source_id]: frame },
          motion: { ...s.motion, [frame.source_id]: frame.motion },
        }))
        break
      }
      case 'motion': {
        const m = msg as unknown as { source_id: string; active: boolean }
        set((s) => ({ motion: { ...s.motion, [m.source_id]: m.active } }))
        break
      }
      case 'source_state': {
        const m = msg as unknown as { source_id: string; state: LiveSourceState }
        set((s) => ({
          liveSources: { ...s.liveSources, [m.source_id]: m.state },
        }))
        break
      }
      case 'system_state': {
        const m = msg as unknown as { state: SystemState }
        set((s) => ({ system: { ...(s.system ?? ({} as SystemState)), ...m.state } }))
        break
      }
      case 'alarm_started': {
        const alert = (msg as unknown as {
          alert: { id: number; identity?: string; reason?: string }
        }).alert
        alertAudio.startAlarm(alert.id)
        get().toast('error', `ALARM: ${alert.reason ?? 'security alert'}`)
        void get().refreshAlerts()
        break
      }
      case 'alarm_stopped': {
        const alert = (msg as unknown as { alert: { id: number } }).alert
        alertAudio.stopAlarm(alert.id)
        void get().refreshAlerts()
        break
      }
      case 'alarm_sound': {
        const m = msg as unknown as { enabled: boolean }
        set({ alarmSound: m.enabled })
        alertAudio.setMuted(!m.enabled)
        break
      }
      case 'alarms_cleared': {
        alertAudio.stopAll()
        set({ activeAlerts: [] })
        break
      }
      case 'alert_beep': {
        alertAudio.beep()
        break
      }
      case 'event': {
        const ev = (msg as unknown as {
          event: { event_type: string; label?: string | null }
        }).event
        if (ev.event_type === 'UNKNOWN_FACE') {
          set((s) => ({ pendingReview: s.pendingReview + 1, reviewGateDismissed: false }))
          void get().refreshReview()
        }
        break
      }
      case 'recognition': {
        const m = msg as unknown as { identity?: string | null; state?: string }
        if (m.state === 'TEMPORARY_FAMILIAR' && m.identity) {
          get().toast('warn', `Temporary familiar recognised: ${m.identity}`)
        } else if (m.state === 'PERMANENT_FAMILIAR' && m.identity) {
          get().toast('ok', `Recognised ${m.identity}`)
        }
        break
      }
      case 'identities_expired': {
        const m = msg as unknown as { identities: { identifier: string }[] }
        get().toast('info', `${m.identities.length} temporary identity(s) expired`)
        void get().refreshIdentities()
        break
      }
      default:
        break
    }
  },

  refreshSources: async () => {
    try {
      set({ sources: await api.sources(), lastError: null })
    } catch (e) {
      set({ lastError: (e as Error).message })
    }
  },
  refreshSystem: async () => {
    try {
      set({ system: await api.systemState(), lastError: null })
    } catch (e) {
      set({ lastError: (e as Error).message })
    }
  },
  refreshAlerts: async () => {
    try {
      const alerts = await api.activeAlerts()
      set({ activeAlerts: alerts })
      alertAudio.syncActiveAlarms(
        alerts.filter((a) => a.alert_type === 'CONTINUOUS_ALARM').map((a) => a.id),
      )
    } catch {
      /* keep the last known alert state */
    }
  },
  refreshReview: async () => {
    try {
      const faces = await api.unfamiliarFaces()
      set({ unfamiliarFaces: faces, pendingReview: faces.length })
    } catch {
      /* non-fatal */
    }
  },
  refreshIdentities: async () => {
    try {
      set({ identities: await api.identities() })
    } catch {
      /* non-fatal */
    }
  },
  refreshAll: async () => {
    await Promise.all([
      get().refreshSystem(),
      get().refreshSources(),
      get().refreshAlerts(),
      get().refreshReview(),
      get().refreshIdentities(),
      get().refreshAlarmSound(),
    ])
    const { sources, selectedSourceId } = get()
    if (selectedSourceId === null && sources.length > 0) {
      set({ selectedSourceId: sources[0].id })
    }
  },

  stopAlert: async (id) => {
    await api.stopAlert(id)
    alertAudio.stopAlarm(id)
    await get().refreshAlerts()
    get().toast('ok', 'Alarm stopped')
  },
  stopAllAlerts: async () => {
    await api.stopAllAlerts()
    alertAudio.stopAll()
    await get().refreshAlerts()
    get().toast('ok', 'All alarms stopped')
  },

  // The mute switch lives on the backend so every open console agrees, and so
  // the choice survives a reload. Detection, recording and the alarm state
  // machine are untouched - this only decides whether anything is audible.
  setAlarmSound: async (enabled) => {
    set({ alarmSound: enabled })
    alertAudio.setMuted(!enabled)
    try {
      await api.setAlarmSound(enabled)
      get().toast('ok', enabled ? 'Alarm sound on' : 'Alarm sound muted')
    } catch (e) {
      set({ alarmSound: !enabled })
      alertAudio.setMuted(enabled)
      get().toast('error', (e as Error).message)
    }
  },
  refreshAlarmSound: async () => {
    try {
      const result = await api.alarmSound()
      const enabled = Boolean((result.data as { enabled?: boolean }).enabled)
      set({ alarmSound: enabled })
      alertAudio.setMuted(!enabled)
    } catch {
      /* non-fatal: keep the last known state */
    }
  },
}))

export const selectSelectedSource = (s: SentinelState): VideoSource | null =>
  s.sources.find((src) => src.id === s.selectedSourceId) ?? null
