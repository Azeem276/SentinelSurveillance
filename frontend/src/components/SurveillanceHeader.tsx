// Global header: system state at a glance plus the two master controls.
import { useState } from 'react'
import { NavLink } from 'react-router-dom'

import { api } from '@/services/api'
import { alertAudio } from '@/services/alertAudio'
import { useSentinelStore } from '@/stores/useSentinelStore'
import { StatusChip } from './primitives'

const TABS = [
  { to: '/', label: 'Monitor', end: true },
  { to: '/review', label: 'Review' },
  { to: '/identities', label: 'Identities' },
  { to: '/archive', label: 'Archive' },
  { to: '/diagnostics', label: 'Diagnostics' },
]

export function SurveillanceHeader() {
  const system = useSentinelStore((s) => s.system)
  const connected = useSentinelStore((s) => s.connected)
  const pendingReview = useSentinelStore((s) => s.pendingReview)
  const refreshAll = useSentinelStore((s) => s.refreshAll)
  const toast = useSentinelStore((s) => s.toast)
  const [busy, setBusy] = useState<string | null>(null)
  const [muted, setMuted] = useState(alertAudio.isMuted)

  const surveillanceOn = system?.surveillance_active ?? false
  const intelligenceOn = system?.intelligence_active ?? false

  const run = async (key: string, fn: () => Promise<unknown>, okMessage: string) => {
    setBusy(key)
    try {
      // Any header click is a user gesture: a good moment to unlock audio.
      alertAudio.unlock()
      await fn()
      await refreshAll()
      toast('ok', okMessage)
    } catch (e) {
      toast('error', (e as Error).message)
    } finally {
      setBusy(null)
    }
  }

  return (
    <header className="header">
      <div className="brand">
        <span className="brand-mark">S</span>
        <span>
          SENTINEL
          <span className="brand-sub"> · surveillance console</span>
        </span>
      </div>

      <nav className="nav-tabs">
        {TABS.map((tab) => (
          <NavLink
            key={tab.to}
            to={tab.to}
            end={tab.end}
            className={({ isActive }) => `nav-tab ${isActive ? 'active' : ''}`}
          >
            {tab.label}
            {tab.to === '/review' && pendingReview > 0 && (
              <span className="badge unfamiliar" style={{ marginLeft: 6 }}>
                {pendingReview}
              </span>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="header-spacer" />

      <div className="header-group">
        <StatusChip
          label={`SURVEILLANCE ${surveillanceOn ? 'ON' : 'OFF'}`}
          state={surveillanceOn ? 'on' : 'off'}
        />
        <StatusChip
          label={`INTELLIGENCE ${intelligenceOn ? 'ON' : 'OFF'}`}
          state={intelligenceOn ? 'on' : 'off'}
          title={
            surveillanceOn
              ? undefined
              : 'Intelligence requires surveillance to be active'
          }
        />
        <StatusChip
          label={connected ? 'LIVE' : 'RECONNECTING'}
          state={connected ? 'on' : 'warn'}
          title="Realtime WebSocket connection"
        />
        {system && !system.database_connected && (
          <StatusChip label="DATABASE DOWN" state="danger" />
        )}
        {system && system.active_alerts > 0 && (
          <StatusChip label={`${system.active_alerts} ALERT`} state="danger" />
        )}
        <StatusChip
          label={(system?.device ?? 'cpu').toUpperCase()}
          state="off"
          title="Inference device"
        />
      </div>

      <div className="header-group">
        <button
          className="btn ghost sm"
          onClick={() => {
            const next = !muted
            alertAudio.setMuted(next)
            setMuted(next)
          }}
          title={muted ? 'Unmute alert audio' : 'Mute alert audio'}
        >
          {muted ? '🔇' : '🔊'}
        </button>

        <button
          className={`btn ${surveillanceOn ? 'danger' : 'success'}`}
          disabled={busy !== null}
          onClick={() =>
            surveillanceOn
              ? run('surv', api.stopSurveillance, 'Surveillance stopped')
              : run('surv', api.startSurveillance, 'Surveillance activated')
          }
        >
          {busy === 'surv'
            ? 'WORKING…'
            : surveillanceOn
              ? 'STOP SURVEILLANCE'
              : 'ACTIVATE SURVEILLANCE'}
        </button>

        <button
          className={`btn ${intelligenceOn ? '' : 'primary'}`}
          disabled={busy !== null || !surveillanceOn}
          title={
            surveillanceOn
              ? undefined
              : 'Activate surveillance before enabling intelligence'
          }
          onClick={() =>
            intelligenceOn
              ? run('intel', api.stopIntelligence, 'Intelligence disabled')
              : run('intel', api.startIntelligence, 'Intelligence enabled')
          }
        >
          {busy === 'intel'
            ? 'WORKING…'
            : intelligenceOn
              ? 'DISABLE INTELLIGENCE'
              : 'ENABLE INTELLIGENCE'}
        </button>
      </div>
    </header>
  )
}
