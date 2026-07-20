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

export interface CalibrationFitHorizon {
  k: number
  raw_brier?: number
  calibrated_brier?: number
  raw_ece?: number
  calibrated_ece?: number
  n_holdout: number
  note?: string
}

export interface AuditReport {
  generated_at: string
  symbols_used: number
  calibration_fit: { fit_at: string; method: string; horizons: Record<string, CalibrationFitHorizon> } | null
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

// ---- steward (Phase 1.5) ----

export interface StewardProposal {
  pattern: string
  action: 'disable' | 'reenable' | 'watch'
  confidence: 'low' | 'medium' | 'high'
  needs_human_review?: boolean
  rationale: string
}

export interface StewardAssessment {
  generated_at: string
  audit_at: string
  model: string
  briefing: string
  overall: 'stable' | 'watch' | 'action_needed'
  proposals: StewardProposal[]
}

export async function fetchSteward(): Promise<{ assessment: StewardAssessment | null; overrides: string[] }> {
  const res = await authFetch('/api/steward')
  if (!res.ok) throw new Error(`steward fetch failed (${res.status})`)
  return res.json()
}

export async function runSteward(): Promise<{ ok: boolean; assessment: StewardAssessment }> {
  const res = await authFetch('/api/steward/run', { method: 'POST' })
  if (!res.ok) {
    let detail = `steward run failed (${res.status})`
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

export async function applyOverride(pattern: string, disabled: boolean): Promise<{ ok: boolean; disabled: string[] }> {
  const res = await authFetch('/api/steward/apply', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pattern, disabled }),
  })
  if (!res.ok) throw new Error(`apply failed (${res.status})`)
  return res.json()
}

export async function runCalibrate(): Promise<{ ok: boolean; states: number; bars: number; horizons: Record<string, number> }> {
  const res = await authFetch('/api/calibrate/run', { method: 'POST' })
  if (!res.ok) throw new Error(`recalibrate failed (${res.status})`)
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
