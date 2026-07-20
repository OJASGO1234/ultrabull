// Live quote polling (Groww feed via our backend). Polls every 5s while the
// market is open and the tab is visible; slows to 60s off-hours. Consumers
// just read the returned quotes map — a fetch failure keeps the last values.
import { useEffect, useRef, useState } from 'react'
import { authFetch } from './auth'

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
  live: boolean // at least one successful poll
}

const FAST_MS = 5_000
const SLOW_MS = 60_000

export function useLiveQuotes(symbols: string[]): LiveState {
  const [state, setState] = useState<LiveState>({ quotes: {}, marketOpen: null, live: false })
  const key = [...new Set(symbols)].sort().join(',')
  const failures = useRef(0)

  useEffect(() => {
    if (!key) {
      setState({ quotes: {}, marketOpen: null, live: false })
      return
    }
    let stopped = false
    let timer: number | undefined

    const tick = async () => {
      if (stopped) return
      let delay = FAST_MS
      if (!document.hidden) {
        try {
          const res = await authFetch(`/api/live?symbols=${encodeURIComponent(key)}`)
          if (res.ok) {
            const body = await res.json()
            if (!stopped) {
              failures.current = 0
              setState((prev) => ({
                quotes: { ...prev.quotes, ...body.quotes },
                marketOpen: body.market_open ?? null,
                live: true,
              }))
              delay = body.market_open ? FAST_MS : SLOW_MS
            }
          } else {
            failures.current += 1
            delay = SLOW_MS // backend says no (no keys / groww down) — don't hammer
          }
        } catch {
          failures.current += 1
          delay = failures.current > 2 ? SLOW_MS : FAST_MS * 2
        }
      }
      if (!stopped) timer = window.setTimeout(() => void tick(), delay)
    }

    void tick()
    return () => {
      stopped = true
      if (timer != null) window.clearTimeout(timer)
    }
  }, [key])

  return state
}
