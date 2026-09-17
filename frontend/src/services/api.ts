// Typed REST client. Every network call in the app goes through here.
import type {
  Alert, Diagnostics, Identity, IdentityCategory, MergeCandidate, ModelInfo,
  Recording, SecurityEvent, SourceAnalysis, SystemState, UnfamiliarFace, VideoSource,
} from '@/types'

const BASE = import.meta.env.VITE_API_BASE ?? ''

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code = 'error',
    readonly details: Record<string, unknown> = {},
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${BASE}${path}`, {
      ...init,
      headers: {
        ...(init.body && !(init.body instanceof FormData)
          ? { 'Content-Type': 'application/json' }
          : {}),
        ...init.headers,
      },
    })
  } catch (cause) {
    throw new ApiError(
      'Cannot reach the Sentinel backend. Is it running on port 8000?',
      0,
      'network_error',
    )
  }

  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`
    let code = 'error'
    let details: Record<string, unknown> = {}
    try {
      const body = await response.json()
      message = body.message ?? body.detail ?? message
      code = body.code ?? code
      details = body.details ?? {}
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(message, response.status, code, details)
  }

  if (response.status === 204) return undefined as T
  const text = await response.text()
  return (text ? JSON.parse(text) : undefined) as T
}

const get = <T>(path: string) => request<T>(path)
const post = <T>(path: string, body?: unknown) =>
  request<T>(path, { method: 'POST', body: body ? JSON.stringify(body) : undefined })
const patch = <T>(path: string, body: unknown) =>
  request<T>(path, { method: 'PATCH', body: JSON.stringify(body) })
const put = <T>(path: string, body: unknown) =>
  request<T>(path, { method: 'PUT', body: JSON.stringify(body) })
const del = <T>(path: string) => request<T>(path, { method: 'DELETE' })

export interface OperationResult {
  ok: boolean
  data: Record<string, unknown>
}

export const api = {
  // ----------------------------------------------------------- system
  health: () => get<{ status: string; database: { connected: boolean } }>('/api/system/health'),
  systemState: () => get<SystemState>('/api/system/state'),
  diagnostics: () => get<Diagnostics>('/api/system/diagnostics'),
  models: () => get<ModelInfo[]>('/api/system/models'),
  runtimeConfig: () => get<Record<string, unknown>>('/api/system/config'),
  alarmSound: () => get<OperationResult>('/api/system/alarm-sound'),
  setAlarmSound: (enabled: boolean) =>
    put<OperationResult>('/api/system/alarm-sound', { enabled }),

  // ---------------------------------------------------- surveillance
  startSurveillance: () => post<OperationResult>('/api/surveillance/start'),
  stopSurveillance: () => post<OperationResult>('/api/surveillance/stop'),
  startIntelligence: () => post<OperationResult>('/api/intelligence/start'),
  stopIntelligence: () => post<OperationResult>('/api/intelligence/stop'),

  // --------------------------------------------------------- sources
  sources: () => get<VideoSource[]>('/api/sources'),
  source: (id: number) => get<VideoSource>(`/api/sources/${id}`),
  createSource: (body: Record<string, unknown>) =>
    post<VideoSource>('/api/sources', body),
  updateSource: (id: number, body: Record<string, unknown>) =>
    patch<VideoSource>(`/api/sources/${id}`, body),
  deleteSource: (id: number) => del<OperationResult>(`/api/sources/${id}`),
  startSource: (id: number) => post<OperationResult>(`/api/sources/${id}/start`),
  stopSource: (id: number) => post<OperationResult>(`/api/sources/${id}/stop`),
  setSourceIntelligence: (id: number, on: boolean) =>
    post<OperationResult>(`/api/sources/${id}/intelligence/${on ? 'start' : 'stop'}`),
  availableVideos: () =>
    get<{ filename: string; size_bytes: number }[]>('/api/sources/available/files'),
  streamUrl: (id: number) => `${BASE}/api/sources/${id}/stream`,
  snapshotUrl: (id: number) => `${BASE}/api/sources/${id}/snapshot`,

  // -------------------------------------------------------- analysis
  analysis: (id: number, hours = 24) =>
    get<SourceAnalysis>(`/api/sources/${id}/analysis?hours=${hours}`),
  sourceEvents: (id: number, hours = 24, limit = 100) =>
    get<SecurityEvent[]>(`/api/sources/${id}/events?hours=${hours}&limit=${limit}`),

  // ------------------------------------------------------ identities
  identities: (params: { category?: IdentityCategory; status?: string; search?: string } = {}) => {
    const q = new URLSearchParams()
    if (params.category) q.set('category', params.category)
    if (params.status) q.set('status', params.status)
    if (params.search) q.set('search', params.search)
    return get<Identity[]>(`/api/identities?${q}`)
  },
  updateIdentity: (id: number, body: Record<string, unknown>) =>
    patch<Identity>(`/api/identities/${id}`, body),
  deleteIdentity: (id: number) => del<OperationResult>(`/api/identities/${id}`),
  mergeIdentities: (id: number, intoIdentityId: number) =>
    post<OperationResult>(`/api/identities/${id}/merge`, {
      into_identity_id: intoIdentityId,
    }),
  enrolDataset: () => post<OperationResult>('/api/identities/enrol-dataset'),
  syncIndex: () => post<OperationResult>('/api/identities/sync-index'),
  expireNow: () => post<OperationResult>('/api/identities/expire-now'),

  // ----------------------------------------------------------- faces
  unfamiliarFaces: (sourceId?: number) =>
    get<UnfamiliarFace[]>(
      `/api/faces/unfamiliar${sourceId ? `?source_id=${sourceId}` : ''}`,
    ),
  pendingCount: () => get<OperationResult>('/api/faces/pending-count'),
  classifyFace: (faceId: number, body: Record<string, unknown>) =>
    post<Record<string, unknown>>(`/api/faces/${faceId}/classify`, body),
  bulkClassify: (body: Record<string, unknown>) =>
    post<Record<string, unknown>[]>('/api/faces/classify-bulk', body),
  mergeFace: (faceId: number, identityId: number) =>
    post<Record<string, unknown>>(`/api/faces/${faceId}/merge`, {
      identity_id: identityId,
    }),
  mergeCandidates: (faceId: number) =>
    get<MergeCandidate[]>(`/api/faces/${faceId}/merge-candidates`),
  dismissFace: (faceId: number) => post<OperationResult>(`/api/faces/${faceId}/dismiss`),
  faceImageUrl: (path: string) => `${BASE}/api/faces/image/${path}`,

  // ---------------------------------------------------- events/alerts
  events: (params: { source_id?: number; hours?: number; limit?: number } = {}) => {
    const q = new URLSearchParams()
    if (params.source_id) q.set('source_id', String(params.source_id))
    q.set('hours', String(params.hours ?? 24))
    q.set('limit', String(params.limit ?? 100))
    return get<SecurityEvent[]>(`/api/events?${q}`)
  },
  recentEvents: (limit = 50) => get<SecurityEvent[]>(`/api/events/recent?limit=${limit}`),
  alerts: (params: { source_id?: number; state?: string } = {}) => {
    const q = new URLSearchParams()
    if (params.source_id) q.set('source_id', String(params.source_id))
    if (params.state) q.set('state', params.state)
    return get<Alert[]>(`/api/alerts?${q}`)
  },
  activeAlerts: () => get<Alert[]>('/api/alerts/active'),
  stopAlert: (id: number) => post<OperationResult>(`/api/alerts/${id}/stop`),
  stopAllAlerts: () => post<OperationResult>('/api/alerts/stop-all'),

  // -------------------------------------------------------- archive
  recordings: (params: { source_id?: number; limit?: number } = {}) => {
    const q = new URLSearchParams()
    if (params.source_id) q.set('source_id', String(params.source_id))
    q.set('limit', String(params.limit ?? 100))
    return get<Recording[]>(`/api/recordings?${q}`)
  },
  recording: (id: number) => get<Recording>(`/api/recordings/${id}`),
  recordingEvents: (id: number) => get<SecurityEvent[]>(`/api/recordings/${id}/events`),
  playbackUrl: (id: number) => `${BASE}/api/recordings/${id}/play`,
  archiveTree: () => get<Record<string, unknown>>('/api/archive/tree'),
}

export const wsUrl = () => {
  if (import.meta.env.VITE_WS_URL) return import.meta.env.VITE_WS_URL as string
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}/ws`
}
