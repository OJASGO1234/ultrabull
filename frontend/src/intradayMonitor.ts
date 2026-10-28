// Intraday paper-monitoring PROTOTYPE client (api/intraday_monitor.py) --
// a SEPARATE experiment from the daily momentum monitor (momentumMonitor.ts).
// The signal engine here is explicitly a test/demo-grade placeholder, not a
// validated intraday strategy -- see that module's docstring.
import { authFetch } from './auth'

export interface IntradayOpenPosition {
  execution_id: number
  signal_id: number
  symbol: string
  direction: 'BUY'
  actual_entry: number
  current_price: number | null
  stop_loss: number | null
  target: number | null
  unrealized_pnl: number | null
  qty: number
  executed_at: string
  holding_minutes: number
}

export interface IntradayRecentSignal {
  id: number
  symbol: string
  direction: string
  status: 'pending' | 'executed' | 'cancelled' | 'closed' | 'rejected'
  signal_timestamp: string
  suggested_entry: number
  rejection_reason?: string | null
}

export interface IntradayStatus {
  mode: 'live' | 'demo'
  market_open: boolean
  timeframe: string
  monitoring_active: boolean
  in_session: boolean
  in_entry_window: boolean
  start_time: string
  entry_cutoff: string
  square_off_time: string
  symbols_monitored: number
  data_source: string | null
  last_data_ts: string | null
  data_stale: boolean
  last_bar: string | null
  last_bar_info?: { bar_ts: string; finalized_at: string; symbol_count: number; symbols: string[] } | null
  data_stale_symbols?: string[]
  data_future_symbols?: string[]
  data_missing_symbols?: string[]
  signals_today: number
  trades_today: number
  max_trades_per_day: number
  open_positions: IntradayOpenPosition[]
  max_open_positions: number
  daily_pnl: number
  max_daily_loss: number
  kill_switch: boolean
  square_off_done: boolean
  recent_signals: IntradayRecentSignal[]
  last_error: string | null
}

export interface IntradayRunResult {
  ok: boolean
  mode: 'live' | 'demo'
  run_at: string
  in_session: boolean
  in_entry_window: boolean
  square_off_due: boolean
  data: { source: string | null; as_of: string | null; stale: boolean }
  bars_finalized: { symbol: string; bar_ts: string; open: number; high: number; low: number; close: number }[]
  signals: { created: { id: number; symbol: string; suggested_entry: number }[]; already_existed: { id: number; symbol: string; suggested_entry: number }[] }
  executions: { attempted: number; filled: { id: number; symbol: string; actual_entry_price: number; qty: number }[]; skipped: { id: number; symbol: string; reason?: string; error?: string }[] }
  position_monitoring: { checked: number; closed: { symbol: string; reason: string; exit_price: number; realized_pnl: number | null }[]; still_open: { symbol: string }[] }
  risk: { open_positions: number; max_open_positions: number; trades_today: number; max_trades_per_day: number; realized_pnl_today: number; max_daily_loss: number; kill_switch: boolean }
  skipped_reason: string | null
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

export function fetchIntradayStatus(): Promise<IntradayStatus> {
  return authFetch('/api/intraday/status').then((r) => _asJson<IntradayStatus>(r, `intraday status failed (${r.status})`))
}

export function startIntradayMonitor(): Promise<{ ok: boolean; monitoring_active: boolean }> {
  return authFetch('/api/intraday/start', { method: 'POST' }).then((r) => _asJson(r, `start failed (${r.status})`))
}

export function stopIntradayMonitor(): Promise<{ ok: boolean; monitoring_active: boolean }> {
  return authFetch('/api/intraday/stop', { method: 'POST' }).then((r) => _asJson(r, `stop failed (${r.status})`))
}

export function runIntradayOnce(opts: {
  demo?: boolean
  demo_now?: string
  demo_prices?: Record<string, number>
} = {}): Promise<IntradayRunResult> {
  return authFetch('/api/intraday/run-once', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(opts),
  }).then((r) => _asJson<IntradayRunResult>(r, `run-once failed (${r.status})`))
}
