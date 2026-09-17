// The WebSocket message reducer: every realtime message type the backend
// sends must land in the right slice, and alarm messages must drive the
// audio service exactly once per alert.
import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/services/api', () => ({
  api: {
    activeAlerts: vi.fn(async () => []),
    unfamiliarFaces: vi.fn(async () => []),
    identities: vi.fn(async () => []),
    sources: vi.fn(async () => []),
    systemState: vi.fn(async () => null),
    stopAlert: vi.fn(async () => ({ ok: true })),
    stopAllAlerts: vi.fn(async () => ({ ok: true })),
  },
}))

vi.mock('@/services/alertAudio', () => ({
  alertAudio: {
    syncActiveAlarms: vi.fn(),
    startAlarm: vi.fn(),
    stopAlarm: vi.fn(),
    stopAll: vi.fn(),
    beep: vi.fn(),
  },
}))

import { api } from '@/services/api'
import { alertAudio } from '@/services/alertAudio'
import type { SystemState, WsMessage } from '@/types'
import { useSentinelStore } from './useSentinelStore'

const system: SystemState = {
  surveillance_active: true,
  intelligence_active: true,
  intelligence_switch: true,
  running_sources: 2,
  recording_sources: 2,
  active_tracks: 1,
  active_alerts: 0,
  pending_review: 0,
  database_connected: true,
  device: 'cuda',
}

const initial = useSentinelStore.getState()

function dispatch(msg: unknown) {
  useSentinelStore.getState().handleMessage(msg as WsMessage)
}

beforeEach(() => {
  useSentinelStore.setState(initial, true)
  vi.clearAllMocks()
})

describe('handleMessage', () => {
  it('snapshot seeds system state, review count and only continuous alarms drive audio', () => {
    dispatch({
      type: 'snapshot',
      system,
      sources: [],
      pending_review: 3,
      active_alerts: [
        { id: 7, alert_type: 'CONTINUOUS_ALARM' },
        { id: 8, alert_type: 'BEEP' },
        { id: 9, alert_type: 'CONTINUOUS_ALARM' },
      ],
    })
    const s = useSentinelStore.getState()
    expect(s.system).toEqual(system)
    expect(s.pendingReview).toBe(3)
    expect(alertAudio.syncActiveAlarms).toHaveBeenCalledWith([7, 9])
    expect(api.activeAlerts).toHaveBeenCalledTimes(1)
  })

  it('frame_analysis is stored per source and updates the motion flag', () => {
    dispatch({ type: 'frame_analysis', source_id: 'cam_01', motion: true, objects: [] })
    dispatch({ type: 'frame_analysis', source_id: 'cam_02', motion: false, objects: [] })
    const s = useSentinelStore.getState()
    expect(Object.keys(s.overlays)).toEqual(['cam_01', 'cam_02'])
    expect(s.motion).toEqual({ cam_01: true, cam_02: false })
  })

  it('motion toggles a single source without touching the others', () => {
    dispatch({ type: 'motion', source_id: 'cam_01', active: true })
    dispatch({ type: 'motion', source_id: 'cam_02', active: true })
    dispatch({ type: 'motion', source_id: 'cam_01', active: false })
    expect(useSentinelStore.getState().motion).toEqual({ cam_01: false, cam_02: true })
  })

  it('source_state replaces the live state of that source only', () => {
    dispatch({ type: 'source_state', source_id: 'cam_01', state: { status: 'AVAILABLE', recording: true } })
    dispatch({ type: 'source_state', source_id: 'cam_02', state: { status: 'UNAVAILABLE', error: 'gone' } })
    dispatch({ type: 'source_state', source_id: 'cam_01', state: { status: 'IDLE', recording: false } })
    expect(useSentinelStore.getState().liveSources).toEqual({
      cam_01: { status: 'IDLE', recording: false },
      cam_02: { status: 'UNAVAILABLE', error: 'gone' },
    })
  })

  it('system_state merges partial updates over the existing state', () => {
    useSentinelStore.setState({ system })
    dispatch({ type: 'system_state', state: { surveillance_active: false, running_sources: 0 } })
    const s = useSentinelStore.getState().system!
    expect(s.surveillance_active).toBe(false)
    expect(s.running_sources).toBe(0)
    // untouched fields survive
    expect(s.device).toBe('cuda')
    expect(s.intelligence_switch).toBe(true)
  })

  it('alarm_started starts the tone for that alert, toasts, and refreshes alerts', () => {
    dispatch({ type: 'alarm_started', source_id: 'cam_01', alert: { id: 42, reason: 'unfamiliar_person_in_alarm_zone' } })
    expect(alertAudio.startAlarm).toHaveBeenCalledWith(42)
    expect(alertAudio.startAlarm).toHaveBeenCalledTimes(1)
    const toasts = useSentinelStore.getState().toasts
    expect(toasts).toHaveLength(1)
    expect(toasts[0].kind).toBe('error')
    expect(toasts[0].text).toContain('unfamiliar_person_in_alarm_zone')
    expect(api.activeAlerts).toHaveBeenCalledTimes(1)
  })

  it('alarm_stopped clears only that alert id', () => {
    dispatch({ type: 'alarm_stopped', source_id: 'cam_01', alert: { id: 42 } })
    expect(alertAudio.stopAlarm).toHaveBeenCalledWith(42)
    expect(alertAudio.stopAll).not.toHaveBeenCalled()
  })

  it('alarms_cleared silences everything and empties the active list', () => {
    useSentinelStore.setState({
      activeAlerts: [{ id: 1 } as never, { id: 2 } as never],
    })
    dispatch({ type: 'alarms_cleared', count: 2 })
    expect(alertAudio.stopAll).toHaveBeenCalledTimes(1)
    expect(useSentinelStore.getState().activeAlerts).toEqual([])
  })

  it('alert_beep plays a single beep and nothing else', () => {
    dispatch({ type: 'alert_beep', source_id: 'cam_01', alert: {} })
    expect(alertAudio.beep).toHaveBeenCalledTimes(1)
    expect(alertAudio.startAlarm).not.toHaveBeenCalled()
  })

  it('an UNKNOWN_FACE event raises the review count and re-opens the review gate', () => {
    useSentinelStore.setState({ pendingReview: 2, reviewGateDismissed: true })
    dispatch({ type: 'event', source_id: 'cam_01', event: { event_type: 'UNKNOWN_FACE' } })
    const s = useSentinelStore.getState()
    expect(s.pendingReview).toBe(3)
    expect(s.reviewGateDismissed).toBe(false)
    expect(api.unfamiliarFaces).toHaveBeenCalledTimes(1)
  })

  it('other events do not touch the review state', () => {
    useSentinelStore.setState({ pendingReview: 2, reviewGateDismissed: true })
    dispatch({ type: 'event', source_id: 'cam_01', event: { event_type: 'PROXIMITY_B_ENTERED' } })
    const s = useSentinelStore.getState()
    expect(s.pendingReview).toBe(2)
    expect(s.reviewGateDismissed).toBe(true)
    expect(api.unfamiliarFaces).not.toHaveBeenCalled()
  })

  it('recognition toasts differ for temporary and permanent familiars and stay silent otherwise', () => {
    dispatch({ type: 'recognition', source_id: 'cam_01', state: 'TEMPORARY_FAMILIAR', identity: 'John' })
    dispatch({ type: 'recognition', source_id: 'cam_01', state: 'PERMANENT_FAMILIAR', identity: 'Azeem' })
    dispatch({ type: 'recognition', source_id: 'cam_01', state: 'UNFAMILIAR', identity: null })
    dispatch({ type: 'recognition', source_id: 'cam_01', state: 'FACE_UNRECOGNIZABLE', identity: null })
    const toasts = useSentinelStore.getState().toasts
    expect(toasts.map((t) => [t.kind, t.text])).toEqual([
      ['warn', 'Temporary familiar recognised: John'],
      ['ok', 'Recognised Azeem'],
    ])
  })

  it('identities_expired toasts and reloads identities', () => {
    dispatch({ type: 'identities_expired', identities: [{ identifier: 'a' }, { identifier: 'b' }] })
    expect(useSentinelStore.getState().toasts[0].text).toContain('2 temporary identity')
    expect(api.identities).toHaveBeenCalledTimes(1)
  })

  it('ping and unknown message types are ignored', () => {
    const before = useSentinelStore.getState()
    dispatch({ type: 'ping' })
    dispatch({ type: 'something_new', payload: 1 })
    const after = useSentinelStore.getState()
    expect(after.system).toBe(before.system)
    expect(after.overlays).toBe(before.overlays)
    expect(after.toasts).toEqual([])
  })
})

describe('operator actions', () => {
  it('stopAlert calls the API, silences that alert and confirms with a toast', async () => {
    await useSentinelStore.getState().stopAlert(5)
    expect(api.stopAlert).toHaveBeenCalledWith(5)
    expect(alertAudio.stopAlarm).toHaveBeenCalledWith(5)
    expect(useSentinelStore.getState().toasts.at(-1)?.text).toBe('Alarm stopped')
  })

  it('stopAllAlerts calls the API and silences everything', async () => {
    await useSentinelStore.getState().stopAllAlerts()
    expect(api.stopAllAlerts).toHaveBeenCalledTimes(1)
    expect(alertAudio.stopAll).toHaveBeenCalled()
  })
})
