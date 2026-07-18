import type { AnalysisResponse } from './types'

export interface RangeQuery {
  years?: number
  start?: string // YYYY-MM-DD
  end?: string // YYYY-MM-DD
}

export async function fetchAnalysis(symbol: string, range: RangeQuery): Promise<AnalysisResponse> {
  const q = new URLSearchParams()
  if (range.start) {
    q.set('start', range.start)
    if (range.end) q.set('end', range.end)
  } else {
    q.set('years', String(range.years ?? 0.5))
  }
  const res = await fetch(`/api/analyze/${encodeURIComponent(symbol)}?${q}`)
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
