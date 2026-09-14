// Audio alert playback.
//
// This is the ONLY place in the UI that makes sound. Components never call
// play() themselves: they report backend alert state to this service, which
// owns the mapping from state to audio. Tones are synthesised with the Web
// Audio API so no asset files are needed and no sound can be triggered by a
// stray render.
//
// Policy (mirrors the backend rule engine):
//   permanent familiar  -> silent
//   temporary familiar  -> one short beep
//   unfamiliar in B     -> continuous alarm until the operator stops it

type AlarmListener = (active: boolean) => void

class AlertAudioService {
  private ctx: AudioContext | null = null
  private alarmOsc: OscillatorNode | null = null
  private alarmGain: GainNode | null = null
  private alarmTimer: number | null = null
  private activeAlarmIds = new Set<number>()
  private listeners = new Set<AlarmListener>()
  private muted = false
  private unlocked = false

  /** Browsers block audio until a user gesture; call this from a click. */
  unlock(): void {
    const ctx = this.ensureContext()
    if (ctx && ctx.state === 'suspended') void ctx.resume()
    this.unlocked = true
  }

  get isUnlocked(): boolean {
    return this.unlocked
  }

  setMuted(muted: boolean): void {
    this.muted = muted
    if (muted) this.stopTone()
    else if (this.activeAlarmIds.size > 0) this.startTone()
  }

  get isMuted(): boolean {
    return this.muted
  }

  private ensureContext(): AudioContext | null {
    if (this.ctx) return this.ctx
    const Ctor =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext?: typeof AudioContext })
        .webkitAudioContext
    if (!Ctor) return null
    this.ctx = new Ctor()
    return this.ctx
  }

  /** A single short beep: temporary familiar recognised. */
  beep(frequency = 880, durationMs = 180): void {
    if (this.muted) return
    const ctx = this.ensureContext()
    if (!ctx) return
    if (ctx.state === 'suspended') void ctx.resume()

    const osc = ctx.createOscillator()
    const gain = ctx.createGain()
    osc.type = 'sine'
    osc.frequency.value = frequency
    // Ramped envelope: an abrupt start/stop clicks unpleasantly.
    gain.gain.setValueAtTime(0.0001, ctx.currentTime)
    gain.gain.exponentialRampToValueAtTime(0.22, ctx.currentTime + 0.015)
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + durationMs / 1000)
    osc.connect(gain).connect(ctx.destination)
    osc.start()
    osc.stop(ctx.currentTime + durationMs / 1000 + 0.02)
  }

  /** Register a continuous alarm. Idempotent per alert id. */
  startAlarm(alertId: number): void {
    const wasEmpty = this.activeAlarmIds.size === 0
    this.activeAlarmIds.add(alertId)
    if (wasEmpty) {
      this.startTone()
      this.emit()
    }
  }

  /** Clear one alarm; the tone stops only when none remain. */
  stopAlarm(alertId: number): void {
    if (!this.activeAlarmIds.delete(alertId)) return
    if (this.activeAlarmIds.size === 0) {
      this.stopTone()
      this.emit()
    }
  }

  stopAll(): void {
    if (this.activeAlarmIds.size === 0) return
    this.activeAlarmIds.clear()
    this.stopTone()
    this.emit()
  }

  /** Reconcile with the authoritative backend list of active alarms. */
  syncActiveAlarms(ids: number[]): void {
    const next = new Set(ids)
    const changed =
      next.size !== this.activeAlarmIds.size ||
      [...next].some((id) => !this.activeAlarmIds.has(id))
    if (!changed) return
    this.activeAlarmIds = next
    if (next.size > 0) this.startTone()
    else this.stopTone()
    this.emit()
  }

  get alarmActive(): boolean {
    return this.activeAlarmIds.size > 0
  }

  get activeCount(): number {
    return this.activeAlarmIds.size
  }

  onAlarmChange(listener: AlarmListener): () => void {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  private emit(): void {
    for (const listener of this.listeners) listener(this.alarmActive)
  }

  private startTone(): void {
    if (this.muted || this.alarmOsc) return
    const ctx = this.ensureContext()
    if (!ctx) return
    if (ctx.state === 'suspended') void ctx.resume()

    const osc = ctx.createOscillator()
    const gain = ctx.createGain()
    osc.type = 'square'
    osc.frequency.value = 660
    gain.gain.value = 0.0001
    osc.connect(gain).connect(ctx.destination)
    osc.start()
    this.alarmOsc = osc
    this.alarmGain = gain

    // Two-tone warble, which carries much better than a flat tone.
    let high = false
    const pulse = () => {
      if (!this.alarmOsc || !this.alarmGain || !this.ctx) return
      high = !high
      const now = this.ctx.currentTime
      this.alarmOsc.frequency.setValueAtTime(high ? 990 : 660, now)
      this.alarmGain.gain.cancelScheduledValues(now)
      this.alarmGain.gain.setValueAtTime(0.0001, now)
      this.alarmGain.gain.exponentialRampToValueAtTime(0.2, now + 0.03)
      this.alarmGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.34)
    }
    pulse()
    this.alarmTimer = window.setInterval(pulse, 420)
  }

  private stopTone(): void {
    if (this.alarmTimer !== null) {
      window.clearInterval(this.alarmTimer)
      this.alarmTimer = null
    }
    if (this.alarmOsc) {
      try {
        this.alarmOsc.stop()
      } catch {
        /* already stopped */
      }
      this.alarmOsc.disconnect()
      this.alarmOsc = null
    }
    if (this.alarmGain) {
      this.alarmGain.disconnect()
      this.alarmGain = null
    }
  }
}

export const alertAudio = new AlertAudioService()
