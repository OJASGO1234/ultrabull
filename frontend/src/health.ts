// Validation-health data: calibration + pattern/gate decay + coverage.
import { authFetch } from './auth'

export interface Stat {
  n: number
  mean_excess_bps: number | null
  t_stat: number | null
  hit_rate: number | null
}

export interface ReliabilityBucket {
  bucket: string
  n: number
  predicted: number
  realized: number
}

export interface CalibrationHorizon {
  n: number
  ece: number
  brier: number
  brier_baseline: number
  overall_predicted: number
  overall_realized: number
  reliability: ReliabilityBucket[]
}

export type PatternStatus =
  | 'decayed' | 'negative' | 'unconfirmed' | 'emerging' | 'holding' | 'no_edge' | 'insufficient'

export interface PatternRow {
  pattern: string
  validated_2026: boolean
  early: Stat
  recent_12m: Stat
  full: Stat
  status: PatternStatus
}

export interface AuditReport {
  generated_at: string
  symbols_used: number
  calibration: {
    cutoff?: string
    max_date?: string
    fit_bars?: number
    holdout_bars?: number
    note?: string
    horizons: Record<string, CalibrationHorizon>
  }
  pattern_decay: { headline_horizon_bars: number; cutoff: string; patterns: PatternRow[] }
  regime_gate: {
    gate: string
    status: string
    horizon_bars?: number
    note?: string
    full?: { gate_gain_bps: number | null; all_longs: Stat; up_market_only: Stat }
    recent_12m?: { gate_gain_bps: number | null; all_longs: Stat; up_market_only: Stat }
  }
  coverage: {
    universe_size: number
    held: number
    covered: string[]
    not_covered: string[]
    coverage_pct: number | null
  }
}

export async function fetchAudit(): Promise<{ available: boolean; report?: AuditReport; detail?: string }> {
  const res = await authFetch('/api/audit')
  if (!res.ok) throw new Error(`audit fetch failed (${res.status})`)
  return res.json()
}

export async function runAudit(): Promise<{ ok: boolean; generated_at: string; decayed: string[] }> {
  const res = await authFetch('/api/audit/run', { method: 'POST' })
  if (!res.ok) {
    let detail = `audit run failed (${res.status})`
    try {
      const b = await res.json()
      if (b?.detail) detail = b.detail
    } catch {
      // keep generic
    }
    throw new Error(detail)
  }
  return res.json()
}
