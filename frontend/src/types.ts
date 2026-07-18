export interface Point {
  time: string
  value: number
}

export interface OhlcvRow {
  time: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface Detection {
  date: string
  pattern: string
  score: number | null
  strength: string | null
  regime: string | null
  direction: string | null
  alignment: string | null
  qualified: boolean
  english: string | null
}

export interface SignalRow {
  bar_idx: number
  date: string | null
  side: string
  tag: string
  score: number | null
  priority: number | null
  enter_at: string | null
  delay_bars: number | null
  sl_atr: number | null
  tp_atr: number | null
  trail_atr: number | null
}

export interface Trade {
  entry_bar: number
  exit_bar: number
  entry_date: string | null
  exit_date: string | null
  side: string
  tag: string
  entry_price: number | null
  exit_price: number | null
  size_shares: number | null
  sl_price: number | null
  tp_price: number | null
  pnl: number | null
  fees: number | null
  reason: string
}

export interface Metrics {
  trades: number
  win_rate: number | null
  avg_win: number | null
  avg_loss: number | null
  profit_factor: number | 'inf' | null
  expectancy: number | null
  median_hold_bars: number | null
  return_total: number | null
  max_drawdown: number | null
  sharpe_daily: number | null
}

export interface ByPatternRow {
  tag: string
  trades: number
  win_rate: number | null
  total_pnl: number | null
  avg_pnl: number | null
  profit_factor: number | 'inf' | null
}

export interface Audit {
  checked: number
  mismatch_count: number
  ok: boolean
}

export interface SignalSummary {
  total_hits?: number | null
  per_week_all?: number | null
  qualified_hits?: number | null
  per_week_qualified?: number | null
  by_family?: Record<string, number>
}

export interface AnalysisResponse {
  symbol: string
  years: number
  bars: number
  last_date: string
  ohlcv: OhlcvRow[]
  indicators: {
    ma20: Point[]
    ma50: Point[]
    bb_upper: Point[]
    bb_lower: Point[]
    rsi: Point[]
    atr: Point[]
    macd: Point[]
    macd_signal: Point[]
    macd_hist: Point[]
  }
  detections: Detection[]
  signals: SignalRow[]
  trades: Trade[]
  equity: Point[]
  metrics: Metrics
  by_pattern: ByPatternRow[]
  audit: Audit
  signal_summary: SignalSummary
  summary_md: string
}
