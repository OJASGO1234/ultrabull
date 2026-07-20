// Live quotes: websocket-first (GrowwFeed relayed by our backend — true
// realtime pushes), with REST polling as automatic fallback. A REST call also
// provides the day baseline (open/high/low/prev_close) that ticks are merged
// into; the baseline refreshes every 5 minutes.
import { useEffect, useRef, useState } from 'react'
import { authFetch, getToken } from './auth'

export interface LiveQuote {
  ltp: number
  open: number | null
  high: number | null
  low: number | null
  prev_close: number | null
  day_change_pct: number | null
  ts: number
}

export interface LiveState {
  quotes: Record<string, LiveQuote>
  marketOpen: boolean | null
  live: boolean
  mode: 'ws' | 'poll' | null
}

const POLL_FAST_MS = 5_000
const POLL_SLOW_MS = 60_000
const BASELINE_MS = 5 * 60_000

function mergeTick(prev: LiveQuote | undefined, ltp: number, ts: number): LiveQuote {
  const base: LiveQuote = prev ?? {
    ltp, open: null, high: null, low: null, prev_close: null, day_change_pct: null, ts,
  }
  return {
    ...base,
    ltp,
    ts,
    high: base.high != null ? Math.max(base.high, ltp) : base.high,
    low: base.low != null ? Math.min(base.low, ltp) : base.low,
    day_change_pct:
      base.prev_close ? Math.round(10000 * (ltp - base.prev_close) / base.prev_close) / 100 : base.day_change_pct,
  }
}

export function useLiveQuotes(symbols: string[]): LiveState {
  const [state, setState] = useState<LiveState>({ quotes: {}, marketOpen: null, live: false, mode: null })
  const key = [...new Set(symbols)].sort().join(',')
  const wsDead = useRef(false)

  useEffect(() => {
    if (!key) {
      setState({ quotes: {}, marketOpen: null, live: false, mode: null })
      return
    }
    let stopped = false
    let ws: WebSocket | null = null
    let pollTimer: number | undefined
    let baselineTimer: number | undefined
    let wsActive = false

    const fetchBaseline = async (): Promise<boolean> => {
      try {
        const res = await authFetch(`/api/live?symbols=${encodeURIComponent(key)}`)
        if (!res.ok) return false
        const body = await res.json()
        if (!stopped) {
          setState((prev) => ({
            quotes: { ...prev.quotes, ...body.quotes },
            marketOpen: body.market_open ?? null,
            live: true,
            mode: wsActive ? 'ws' : prev.mode,
          }))
        }
        return true
      } catch {
        return false
      }
    }

    const startPolling = () => {
      if (stopped || pollTimer != null) return
      const tick = async () => {
        if (stopped) return
        let delay = POLL_FAST_MS
        if (!document.hidden) {
          const ok = await fetchBaseline()
          if (!stopped) {
            setState((prev) => ({ ...prev, mode: prev.live ? 'poll' : prev.mode }))
            delay = ok ? POLL_FAST_MS : POLL_SLOW_MS
          }
        }
        if (!stopped) pollTimer = window.setTimeout(() => void tick(), delay)
      }
      void tick()
    }

    const startWs = () => {
      if (stopped || wsDead.current) {
        startPolling()
        return
      }
      const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
      const token = getToken() ?? ''
      try {
        ws = new WebSocket(
          `${proto}://${window.location.host}/api/live/ws?symbols=${encodeURIComponent(key)}&token=${encodeURIComponent(token)}`,
        )
      } catch {
        wsDead.current = true
        startPolling()
        return
      }
      ws.onopen = () => {
        wsActive = true
        setState((prev) => ({ ...prev, mode: 'ws' }))
        void fetchBaseline() // day OHLC + prev_close for change calc
        baselineTimer = window.setInterval(() => void fetchBaseline(), BASELINE_MS)
      }
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data)
          if (msg.type === 'tick' && !stopped) {
            setState((prev) => {
              const quotes = { ...prev.quotes }
              for (const [sym, q] of Object.entries(msg.quotes as Record<string, { ltp: number; ts: number }>)) {
                quotes[sym] = mergeTick(quotes[sym], q.ltp, q.ts)
              }
              return { ...prev, quotes, live: true, mode: 'ws' }
            })
          } else if (msg.type === 'hb' && !stopped) {
            setState((prev) => ({ ...prev, marketOpen: msg.market_open ?? prev.marketOpen }))
          }
        } catch {
          // malformed frame — ignore
        }
      }
      ws.onclose = () => {
        wsActive = false
        if (baselineTimer != null) window.clearInterval(baselineTimer)
        if (!stopped) {
          wsDead.current = true // don't thrash reconnects; polling takes over
          startPolling()
        }
      }
      ws.onerror = () => ws?.close()
    }

    startWs()
    return () => {
      stopped = true
      if (pollTimer != null) window.clearTimeout(pollTimer)
      if (baselineTimer != null) window.clearInterval(baselineTimer)
      ws?.close()
    }
  }, [key])

  return state
}
