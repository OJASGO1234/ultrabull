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
  side: string | null
  ref_close: number | null
  sl_est: number | null
  tp_est: number | null
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
  ref_close: number | null
  sl_est: number | null
  tp_est: number | null
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

export interface EventRow {
  date: string
  type: 'DIVIDEND' | 'SPLIT' | 'EARNINGS' | string
  label: string
}

export interface EventRef {
  date: string
  label: string
}

/** Latest market/sector regime + relative-strength snapshot (drives the gates). */
export interface MarketContext {
  market_regime: 'uptrend' | 'downtrend' | 'neutral' | null
  sector_regime: 'uptrend' | 'downtrend' | 'neutral' | null
  sector: string | null
  rs_20: number | null
  rs_60: number | null
  adx: number | null
  breadth: number | null
  dist_days: number | null
}

/** Forward-looking results-cycle / corporate-action timing. */
export interface Timing {
  next_earnings: string | null
  trading_days_to_next_earnings: number | null
  in_earnings_blackout: boolean
  last_earnings: EventRef | null
  days_since_last_earnings: number | null
  results_overdue: boolean
  earnings_data_stale: boolean
  next_ex_dividend: string | null
  last_dividend: EventRef | null
  last_split: EventRef | null
}

export interface Level {
  price: number
  kind: 'support' | 'resistance'
  touches: number
  strength: number | null
  last_touch: string
  source: 'swings' | '52w' | string
  distance_pct: number | null
}

export interface OutlookHorizon {
  p_up: number | null
  median_move: number | null
  typical_range: [number | null, number | null]
  n: number | null
  verdict: 'up' | 'down' | 'sideways' | 'unknown'
  fallback: boolean
}

export interface Outlook {
  state: {
    key: string
    market: string
    trend: string
    rs: string
    rsi_band: string
    rsi: number
  }
  horizons: Record<'1d' | '1w' | '1m' | '3m', OutlookHorizon>
  nearest_support: number | null
  nearest_resistance: number | null
  note: string
}

export interface AdviceFactor {
  name: string
  score: number
  detail: string
}

export interface Advice {
  as_of: string
  close: number
  score: number
  fresh: { verdict: string; reason: string }
  holding: { verdict: string; reason: string }
  factors: AdviceFactor[]
  plan: {
    entry_zone_low: number | null
    entry_zone_high: number | null
    stop: number
    stop_basis: string
    target1: number
    target2: number | null
    reward_risk: number | null
  }
  note: string
}

export interface FyPoint {
  fy: string
  value_cr: number
}

export interface FundFlag {
  level: 'good' | 'warn' | 'bad'
  text: string
}

export interface Fundamentals {
  available: boolean
  snapshot: {
    market_cap_cr: number | null
    trailing_pe: number | null
    price_to_book: number | null
    debt_to_equity: number | null
    dividend_yield_pct: number | null
  }
  annual: {
    revenue?: FyPoint[]
    net_income?: FyPoint[]
    total_debt?: FyPoint[]
    net_margin?: { fy: string; value_pct: number }[]
  }
  quarterly: {
    revenue?: { quarter: string; value_cr: number; yoy_pct: number | null } | null
    net_income?: { quarter: string; value_cr: number; yoy_pct: number | null } | null
  }
  flags: FundFlag[]
  promoter: {
    holding_pct: number | null
    institutions_pct: number | null
    pledge_pct: number | null
    history: unknown[]
    source_note: string
  }
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
    adx: Point[]
  }
  events: EventRow[]
  context: MarketContext
  timing: Timing
  levels: Level[]
  outlook: Outlook | null
  advice: Advice | null
  fundamentals: Fundamentals | null
  interpretations: { rsi: string | null; macd: string | null; equity: string | null } | null
  indexes: { name: string; points: Point[] }[]
  detections: Detection[]
  signals: SignalRow[]
  trades: Trade[]
  equity: Point[]
  metrics: Metrics
  by_pattern: ByPatternRow[]
  audit: Audit
  signal_summary: SignalSummary
  summary_md: string
  cached?: boolean
}
