// Proximity A > B > 0 must be enforced before anything reaches the API, with
// a message the operator can act on. The backend re-validates; this is the
// immediate feedback path.
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/services/api', () => ({
  api: {
    updateSource: vi.fn(async () => ({})),
    sources: vi.fn(async () => []),
  },
}))

import { api } from '@/services/api'
import type { VideoSource } from '@/types'
import { SourceSettings } from './SourceSettings'

const source: VideoSource = {
  id: 1,
  uid: 'cam_01',
  name: 'Front door',
  type: 'FILE',
  uri: 'cam_01.mp4',
  location: null,
  description: null,
  enabled: true,
  surveillance_enabled: true,
  intelligence_enabled: true,
  recognition_enabled: true,
  recording_enabled: true,
  loop_playback: true,
  status: 'IDLE',
  last_error: null,
  last_seen_at: null,
  proximity_a: 10,
  proximity_b: 3,
  recognition_threshold: null,
  detection_confidence: null,
  temporary_retention_days: 7,
  alert_policy: {},
  calibration: {},
  display_order: 0,
  created_at: '2026-09-14T00:00:00Z',
  updated_at: '2026-09-14T00:00:00Z',
  runtime: { running: false, status: 'IDLE', recording: false, intelligence: false, motion: false, active_tracks: 0, frames_read: 0, error: null, stats: {} },
} as unknown as VideoSource

const A_B_MESSAGE =
  'Proximity A (recognition zone) must be greater than Proximity B (alarm zone)'

function setup() {
  const onClose = vi.fn()
  render(<SourceSettings source={source} onClose={onClose} />)
  const a = screen.getByLabelText(/Proximity A/) as HTMLInputElement
  const b = screen.getByLabelText(/Proximity B/) as HTMLInputElement
  const save = screen.getByRole('button', { name: 'Save' })
  return { onClose, a, b, save }
}

beforeEach(() => {
  vi.clearAllMocks()
})

describe('SourceSettings proximity validation', () => {
  it('pre-fills the current boundaries', () => {
    const { a, b } = setup()
    expect(a.value).toBe('10')
    expect(b.value).toBe('3')
  })

  it.each([
    ['3', '10'],
    ['5', '5'],
    ['2.9', '3'],
  ])('rejects A=%s with B=%s and does not call the API', async (aVal, bVal) => {
    const { a, b, save, onClose } = setup()
    fireEvent.change(a, { target: { value: aVal } })
    fireEvent.change(b, { target: { value: bVal } })
    fireEvent.click(save)

    expect(await screen.findByText(A_B_MESSAGE)).toBeInTheDocument()
    expect(api.updateSource).not.toHaveBeenCalled()
    expect(onClose).not.toHaveBeenCalled()
    expect(a).toHaveClass('error')
    expect(b).toHaveClass('error')
  })

  it.each([
    ['0', '3', 'Proximity A must be greater than 0'],
    ['-4', '3', 'Proximity A must be greater than 0'],
    ['abc', '3', 'Proximity A must be greater than 0'],
    ['10', '0', 'Proximity B must be greater than 0'],
    ['10', '', 'Proximity B must be greater than 0'],
  ])('rejects A=%s B=%s with "%s"', async (aVal, bVal, message) => {
    const { a, b, save } = setup()
    fireEvent.change(a, { target: { value: aVal } })
    fireEvent.change(b, { target: { value: bVal } })
    fireEvent.click(save)
    expect(await screen.findByText(message)).toBeInTheDocument()
    expect(api.updateSource).not.toHaveBeenCalled()
  })

  it('accepts A > B > 0, sends numbers to the API and closes', async () => {
    const { a, b, save, onClose } = setup()
    fireEvent.change(a, { target: { value: '12.5' } })
    fireEvent.change(b, { target: { value: '4' } })
    fireEvent.click(save)

    await waitFor(() => expect(api.updateSource).toHaveBeenCalledTimes(1))
    const [id, body] = (api.updateSource as ReturnType<typeof vi.fn>).mock.calls[0] as [number, Record<string, unknown>]
    expect(id).toBe(1)
    expect(body.proximity_a).toBe(12.5)
    expect(body.proximity_b).toBe(4)
    expect(typeof body.proximity_a).toBe('number')
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1))
    expect(screen.queryByText(A_B_MESSAGE)).not.toBeInTheDocument()
  })

  it('clears the error once the operator fixes the values', async () => {
    const { a, b, save } = setup()
    fireEvent.change(a, { target: { value: '2' } })
    fireEvent.click(save)
    expect(await screen.findByText(A_B_MESSAGE)).toBeInTheDocument()

    fireEvent.change(a, { target: { value: '8' } })
    fireEvent.change(b, { target: { value: '2' } })
    fireEvent.click(save)
    await waitFor(() => expect(api.updateSource).toHaveBeenCalledTimes(1))
    expect(screen.queryByText(A_B_MESSAGE)).not.toBeInTheDocument()
  })

  it('requires a name', async () => {
    render(<SourceSettings source={source} onClose={vi.fn()} />)
    fireEvent.change(screen.getByLabelText('Source name'), { target: { value: '   ' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))
    expect(await screen.findByText('Name is required')).toBeInTheDocument()
    expect(api.updateSource).not.toHaveBeenCalled()
  })
})
