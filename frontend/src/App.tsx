import { useEffect } from 'react'
import { Route, Routes } from 'react-router-dom'

import { AlertBanner, ReviewBanner, Toasts } from '@/components/AlertBanner'
import { SurveillanceHeader } from '@/components/SurveillanceHeader'
import { useRealtime } from '@/hooks/useRealtime'
import { ArchivePage } from '@/pages/ArchivePage'
import { DiagnosticsPage } from '@/pages/DiagnosticsPage'
import { IdentitiesPage } from '@/pages/IdentitiesPage'
import { MonitorPage } from '@/pages/MonitorPage'
import { ReviewPage } from '@/pages/ReviewPage'
import { alertAudio } from '@/services/alertAudio'
import { useSentinelStore } from '@/stores/useSentinelStore'

export default function App() {
  const refreshAll = useSentinelStore((s) => s.refreshAll)
  const lastError = useSentinelStore((s) => s.lastError)
  useRealtime()

  useEffect(() => {
    void refreshAll()
    // Browsers require a user gesture before audio can play; arm on first click.
    const unlock = () => alertAudio.unlock()
    window.addEventListener('pointerdown', unlock, { once: true })
    return () => window.removeEventListener('pointerdown', unlock)
  }, [refreshAll])

  return (
    <div className="app">
      <SurveillanceHeader />
      <div className="app-main">
        <AlertBanner />
        <ReviewBanner />
        {lastError && (
          <div className="alert-banner" style={{ background: 'var(--danger-dim)' }}>
            <span>⚠</span>
            <span>{lastError}</span>
          </div>
        )}
        <div className="app-route">
          <Routes>
            <Route path="/" element={<MonitorPage />} />
            <Route path="/review" element={<ReviewPage />} />
            <Route path="/identities" element={<IdentitiesPage />} />
            <Route path="/archive" element={<ArchivePage />} />
            <Route path="/diagnostics" element={<DiagnosticsPage />} />
            <Route path="*" element={<MonitorPage />} />
          </Routes>
        </div>
      </div>
      <Toasts />
    </div>
  )
}
