// Continuous-monitoring prototype client (api/momentum_monitor.py).
// Live mode only from the UI (no demo controls here) -- demo mode exists
// for scripted/API-level demonstration, see core/MOMENTUM_STRATEGY.md's
// "Continuous Monitoring Prototype" section.
import { authFetch } from './auth'

export interface MonitorRecommendation {
  id: number
  symbol: string
  as_of_date: string
  suggested_entry: number
  sl_est: number | null
  tp_est: number | null
  absolute_momentum: number | null
  relative_momentum: number | null
  rank: number | null
  status: 'pending' | 'executed' | 'cancelled' | 'closed'
  executed_at: string | null
  actual_entry: number | null
  closed_at: string | null
  exit_price: number | null
  exit_reason: string | null
  close_realized_pnl: number | null
}

export interface MonitorOpenPosition {
  recommendation_id: number
  symbol: string
  qty: number
  suggested_entry: number
  actual_entry: number | null
  current_price: number | null
  unrealized_pnl: number | null
  stop: number | null
  target: number | null
  executed_at: string | null
}

export interface MonitorCycleResult {
  ok: boolean
  run_at: string
  mode: 'live' | 'demo'
  market_open: boolean
  lookback: string
  recommendation_generation: {
    skipped: boolean
    reason: string | null
    as_of_date: string | null
    screened: number
    candidates: number
    created: { id: number; symbol: string; suggested_entry: number }[]
    already_existed: { id: number; symbol: string; suggested_entry: number }[]
  }
  executions: {
    attempted: number
    filled: { id: number; symbol: string; suggested_entry: number; actual_entry: number; qty: number; slippage_pct: number | null }[]
    skipped_market_closed: { id: number; symbol: string }[]
    skipped_insufficient_info: { id: number; symbol: string }[]
  }
  position_monitoring: {
    checked: number
    closed: { id: number; symbol: string; reason: string; exit_price: number; realized_pnl: number | null; return_pct: number | null; days_held: number }[]
    still_open: { id: number; symbol: string; current_price?: number; unrealized_pnl?: number | null; days_held?: number }[]
  }
  summary: string
}

export interface MonitorSchedulerStatus {
  enabled: boolean
  timezone: string
  schedule_time: string
  scheduler_user: string
  last_scheduled_run_at: string | null
  next_run_at: string | null
  last_error: string | null
}

export interface MonitorStatus {
  mode: 'live' | 'demo'
  market_open: boolean
  scheduler: MonitorSchedulerStatus
  last_run: MonitorCycleResult | null
  latest_recommendations: MonitorRecommendation[]
  open_positions: MonitorOpenPosition[]
  recent_outcomes: MonitorRecommendation[]
  recommendations_today: number
}

async function _asJson<T>(res: Response, failMsg: string): Promise<T> {
  if (!res.ok) {
    let detail = failMsg
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

export function runMonitorCycle(opts: {
  lookback?: string
  top_n?: number
  symbols?: string
} = {}): Promise<MonitorCycleResult> {
  return authFetch('/api/momentum/monitor/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(opts),
  }).then((r) => _asJson<MonitorCycleResult>(r, `monitor run failed (${r.status})`))
}

export function fetchMonitorStatus(): Promise<MonitorStatus> {
  return authFetch('/api/momentum/monitor/status').then((r) => _asJson<MonitorStatus>(r, `monitor status failed (${r.status})`))
}
