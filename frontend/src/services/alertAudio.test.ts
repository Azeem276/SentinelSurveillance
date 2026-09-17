// The alert-audio state machine, driven with a fake Web Audio API so the
// tests assert on what would be heard: nothing while muted, one tone while
// any continuous alarm is active, one beep per temporary-familiar recognition.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// ------------------------------------------------------------ fake WebAudio
class FakeParam {
  value = 0
  setValueAtTime = vi.fn()
  exponentialRampToValueAtTime = vi.fn()
  cancelScheduledValues = vi.fn()
}

class FakeNode {
  connect = vi.fn(() => this)
  disconnect = vi.fn()
}

class FakeOscillator extends FakeNode {
  type = 'sine'
  frequency = new FakeParam()
  start = vi.fn()
  stop = vi.fn()
}

class FakeGain extends FakeNode {
  gain = new FakeParam()
}

class FakeAudioContext {
  static instances: FakeAudioContext[] = []
  state = 'running'
  currentTime = 0
  destination = {}
  oscillators: FakeOscillator[] = []
  resume = vi.fn(async () => undefined)
  constructor() {
    FakeAudioContext.instances.push(this)
  }
  createOscillator() {
    const osc = new FakeOscillator()
    this.oscillators.push(osc)
    return osc
  }
  createGain() {
    return new FakeGain()
  }
}

async function loadService() {
  vi.resetModules()
  const mod = await import('./alertAudio')
  return mod
}

beforeEach(() => {
  FakeAudioContext.instances = []
  ;(window as unknown as { AudioContext: unknown }).AudioContext = FakeAudioContext
  window.localStorage.clear()
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
})

// ------------------------------------------------------------------ tests
describe('mute default and persistence', () => {
  it('starts muted on a browser that has never chosen', async () => {
    const { alertAudio } = await loadService()
    expect(alertAudio.isMuted).toBe(true)
  })

  it('honours a stored unmute', async () => {
    window.localStorage.setItem('sentinel.alertAudio.muted', 'false')
    const { alertAudio } = await loadService()
    expect(alertAudio.isMuted).toBe(false)
  })

  it('persists the operator toggle', async () => {
    const { alertAudio, MUTE_STORAGE_KEY } = await loadService()
    alertAudio.setMuted(false)
    expect(window.localStorage.getItem(MUTE_STORAGE_KEY)).toBe('false')
    alertAudio.setMuted(true)
    expect(window.localStorage.getItem(MUTE_STORAGE_KEY)).toBe('true')
  })
})

describe('continuous alarm tone', () => {
  it('makes no sound while muted, but still tracks the alarm state', async () => {
    const { alertAudio } = await loadService()
    const changes: boolean[] = []
    alertAudio.onAlarmChange((active) => changes.push(active))

    alertAudio.startAlarm(1)
    expect(alertAudio.alarmActive).toBe(true)
    expect(alertAudio.activeCount).toBe(1)
    expect(changes).toEqual([true])
    expect(FakeAudioContext.instances).toHaveLength(0)
  })

  it('starts one tone when unmuted and keeps it while any alarm remains', async () => {
    const { alertAudio } = await loadService()
    alertAudio.setMuted(false)

    alertAudio.startAlarm(1)
    alertAudio.startAlarm(2)
    alertAudio.startAlarm(1) // idempotent
    const ctx = FakeAudioContext.instances[0]
    expect(ctx.oscillators).toHaveLength(1)
    expect(ctx.oscillators[0].start).toHaveBeenCalledTimes(1)
    expect(alertAudio.activeCount).toBe(2)

    alertAudio.stopAlarm(1)
    expect(ctx.oscillators[0].stop).not.toHaveBeenCalled()
    expect(alertAudio.alarmActive).toBe(true)

    alertAudio.stopAlarm(2)
    expect(ctx.oscillators[0].stop).toHaveBeenCalledTimes(1)
    expect(alertAudio.alarmActive).toBe(false)
  })

  it('warbles between two frequencies while active', async () => {
    const { alertAudio } = await loadService()
    alertAudio.setMuted(false)
    alertAudio.startAlarm(1)
    const osc = FakeAudioContext.instances[0].oscillators[0]
    vi.advanceTimersByTime(420 * 3)
    const freqs = osc.frequency.setValueAtTime.mock.calls.map((c) => c[0])
    expect(freqs.length).toBeGreaterThanOrEqual(4)
    expect(new Set(freqs)).toEqual(new Set([660, 990]))

    alertAudio.stopAll()
    const before = osc.frequency.setValueAtTime.mock.calls.length
    vi.advanceTimersByTime(420 * 3)
    expect(osc.frequency.setValueAtTime.mock.calls.length).toBe(before)
  })

  it('unmuting during an active alarm starts the tone; muting stops it', async () => {
    const { alertAudio } = await loadService()
    alertAudio.startAlarm(9)
    expect(FakeAudioContext.instances).toHaveLength(0)

    alertAudio.setMuted(false)
    const osc = FakeAudioContext.instances[0].oscillators[0]
    expect(osc.start).toHaveBeenCalledTimes(1)

    alertAudio.setMuted(true)
    expect(osc.stop).toHaveBeenCalledTimes(1)
    expect(alertAudio.alarmActive).toBe(true) // still alarming, just silent
  })

  it('syncActiveAlarms reconciles with the backend list', async () => {
    const { alertAudio } = await loadService()
    alertAudio.setMuted(false)
    const changes: boolean[] = []
    alertAudio.onAlarmChange((active) => changes.push(active))

    alertAudio.syncActiveAlarms([1, 2])
    expect(alertAudio.activeCount).toBe(2)
    alertAudio.syncActiveAlarms([2, 1]) // same set: no change, no new tone
    expect(FakeAudioContext.instances[0].oscillators).toHaveLength(1)
    alertAudio.syncActiveAlarms([])
    expect(alertAudio.alarmActive).toBe(false)
    expect(changes).toEqual([true, false])
  })

  it('stopping an unknown alarm id is a no-op', async () => {
    const { alertAudio } = await loadService()
    const changes: boolean[] = []
    alertAudio.onAlarmChange((active) => changes.push(active))
    alertAudio.stopAlarm(123)
    alertAudio.stopAll()
    expect(changes).toEqual([])
  })
})

describe('beep', () => {
  it('is suppressed while muted', async () => {
    const { alertAudio } = await loadService()
    alertAudio.beep()
    expect(FakeAudioContext.instances).toHaveLength(0)
  })

  it('plays a short, self-stopping tone when unmuted', async () => {
    const { alertAudio } = await loadService()
    alertAudio.setMuted(false)
    alertAudio.beep()
    const osc = FakeAudioContext.instances[0].oscillators[0]
    expect(osc.start).toHaveBeenCalledTimes(1)
    expect(osc.stop).toHaveBeenCalledTimes(1)
    expect(osc.frequency.value).toBe(880)
    expect(alertAudio.alarmActive).toBe(false)
  })
})
