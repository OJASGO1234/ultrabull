import type { AnalysisResponse } from './types'

export async function fetchAnalysis(symbol: string, years: number): Promise<AnalysisResponse> {
  const res = await fetch(`/api/analyze/${encodeURIComponent(symbol)}?years=${years}`)
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
