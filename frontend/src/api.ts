import { authFetch } from './auth'
import type { AnalysisResponse, MomentumScreenResponse } from './types'

export interface RangeQuery {
  years?: number
  start?: string // YYYY-MM-DD
  end?: string // YYYY-MM-DD
}

/** History-window presets shared by the analysis tab and the holdings Analyse controls. */
export const RANGE_PRESETS = [
  { value: '0.25', label: '3 months' },
  { value: '0.5', label: '6 months' },
  { value: '1', label: '1 year' },
  { value: '2', label: '2 years' },
  { value: '3', label: '3 years' },
  { value: '5', label: '5 years' },
  { value: 'custom', label: 'Custom range…' },
]

export function rangeToQuery(range: RangeQuery): URLSearchParams {
  const q = new URLSearchParams()
  if (range.start) {
    q.set('start', range.start)
    if (range.end) q.set('end', range.end)
  } else {
    q.set('years', String(range.years ?? 0.5))
  }
  return q
}

export async function fetchAnalysis(symbol: string, range: RangeQuery, interval = '1d'): Promise<AnalysisResponse> {
  const q = rangeToQuery(range)
  if (interval && interval !== '1d') q.set('interval', interval)
  const res = await authFetch(`/api/analyze/${encodeURIComponent(symbol)}?${q}`)
  if (!res.ok) {
    let detail = `Request failed (${res.status})`
    try {
      const body = await res.json()
      if (body?.detail) detail = body.detail
    } catch {
      // non-JSON error body; keep the generic message
    }
    throw new Error(detail)
  }
  return res.json()
}

export async function fetchMomentumScreen(
  lookback: '3M' | '6M' | '9M' | '12M',
  opts: { topN?: number; years?: number } = {},
): Promise<MomentumScreenResponse> {
  const q = new URLSearchParams({ lookback, top_n: String(opts.topN ?? 10), years: String(opts.years ?? 2) })
  const res = await authFetch(`/api/momentum/screen?${q}`)
  if (!res.ok) {
    let detail = `Request failed (${res.status})`
    try {
      const body = await res.json()
      if (body?.detail) detail = body.detail
    } catch {
      // non-JSON error body; keep the generic message
    }
    throw new Error(detail)
  }
  return res.json()
}
