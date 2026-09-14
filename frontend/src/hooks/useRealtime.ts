// WebSocket connection with automatic reconnection.
//
// A dropped socket must never require a page reload: the hook reconnects
// with capped exponential backoff and re-syncs authoritative state from REST
// once the connection is restored.
import { useEffect, useRef } from 'react'

import { wsUrl } from '@/services/api'
import { useSentinelStore } from '@/stores/useSentinelStore'
import type { WsMessage } from '@/types'

const MAX_BACKOFF_MS = 15_000

export function useRealtime(): void {
  const handleMessage = useSentinelStore((s) => s.handleMessage)
  const setConnected = useSentinelStore((s) => s.setConnected)
  const refreshAll = useSentinelStore((s) => s.refreshAll)

  const socketRef = useRef<WebSocket | null>(null)
  const attemptsRef = useRef(0)
  const timerRef = useRef<number | null>(null)
  const closedByUs = useRef(false)

  useEffect(() => {
    closedByUs.current = false

    const connect = () => {
      if (closedByUs.current) return
      let socket: WebSocket
      try {
        socket = new WebSocket(wsUrl())
      } catch {
        scheduleReconnect()
        return
      }
      socketRef.current = socket

      socket.onopen = () => {
        const reconnected = attemptsRef.current > 0
        attemptsRef.current = 0
        setConnected(true)
        // Re-sync after a gap: overlays are ephemeral but alerts are not.
        if (reconnected) void refreshAll()
      }

      socket.onmessage = (event) => {
        try {
          handleMessage(JSON.parse(event.data) as WsMessage)
        } catch {
          /* ignore malformed frames rather than tearing down the socket */
        }
      }

      socket.onerror = () => {
        /* onclose always follows; handled there */
      }

      socket.onclose = () => {
        setConnected(false)
        socketRef.current = null
        scheduleReconnect()
      }
    }

    const scheduleReconnect = () => {
      if (closedByUs.current) return
      attemptsRef.current += 1
      const delay = Math.min(500 * 2 ** (attemptsRef.current - 1), MAX_BACKOFF_MS)
      timerRef.current = window.setTimeout(connect, delay)
    }

    connect()

    return () => {
      closedByUs.current = true
      if (timerRef.current !== null) window.clearTimeout(timerRef.current)
      socketRef.current?.close()
      socketRef.current = null
    }
  }, [handleMessage, setConnected, refreshAll])
}
