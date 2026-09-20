// time is a date string ('YYYY-MM-DD') on daily bars, or a UNIX-seconds number on intraday
export interface Point {
  time: string | number
  value: number
}

export interface OhlcvRow {
  time: string | number
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
  /** True when no sector index is mapped, so rs_* is measured against NIFTY
   * itself — market data reaching even the stock-only view. */
  rs_vs_market?: boolean
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
  raw_p_up?: number | null
  calibrated?: boolean
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
  market_context_applied?: boolean
  nearest_support: number | null
  nearest_resistance: number | null
  note: string
}

export interface AdviceFactor {
  name: string
  score: number
  detail: string
  /** "setup" = evidence about this stock; "environment" = the tape; "veto" =
   * a sentinel-scored blocker (earnings blackout) that belongs to neither sum. */
  axis?: 'setup' | 'environment' | 'veto'
  /** True in the stock-only view: shown for transparency, excluded from the score. */
  ignored?: boolean
}

/** A tape condition vetoing or dragging an otherwise-fine setup. */
export interface AdviceBlocker {
  name: string
  detail: string
  clears_when: string
}

export interface Advice {
  as_of: string
  close: number
  score: number
  setup_score?: number
  env_score?: number
  market_context_applied?: boolean
  blockers?: AdviceBlocker[]
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
    if_breakout: IfBreakout | null
  }
  note: string
}

export type ViewName = 'market_aware' | 'stock_only'

export interface AnalysisView {
  label: string
  advice: Advice | null
  outlook: Outlook | null
  /** The whole backtest is recomputed per view, so a view is self-consistent. */
  metrics: Metrics | null
  key_indicators: KeyIndicators | null
  by_pattern: ByPatternRow[]
  signals: SignalRow[]
  trades: Trade[]
  equity: Point[]
  /** Per-view: detection SCORES carry a sector-relative term in the
   * market-aware view and drop it in stock-only. Gating is identical. */
  detections?: Detection[]
  /** stock_only only: what was lifted/kept, and what that costs. `lifted_gates`
   * (backtest filters) and `lifted_score_factors` (advice-score inputs) are
   * deliberately separate — they are not the same set. */
  lifted_gates?: string[]
  lifted_score_factors?: string[]
  already_off?: string[]
  kept_filters?: string[]
  warning?: string
}

export type AnalysisViews = Record<ViewName, AnalysisView>

export interface KeyFlag {
  label: string
  state: 'bullish' | 'bearish' | 'neutral'
  detail: string
  value: string | null
  axis?: 'setup' | 'environment'
  /** True in the stock-only view: still shown, but excluded from the lean. */
  ignored?: boolean
}

export interface KeyIndicators {
  lean: 'bullish' | 'bearish' | 'mixed'
  bullish_count: number
  bearish_count: number
  flags: KeyFlag[]
  note: string
}

/** Conditional path when the target sits at resistance: what changes if it clears. */
export interface IfBreakout {
  trigger: number | null
  target: number | null
  target_basis: string | null
  stop_after: number | null
  note: string | null
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

export interface MomentumLookback {
  months: number
  bars: number
  absolute_momentum: number | null
  relative_momentum: number | null
}

export interface Momentum {
  benchmark: string
  as_of: string
  lookbacks: Record<'3M' | '6M' | '9M' | '12M', MomentumLookback>
  technical_confirmation: {
    close_above_ma20: boolean
    close_above_ma50: boolean
    ma20_above_ma50: boolean
    confirmed: boolean
  }
  note: string
}

export interface MomentumScreenRow {
  symbol: string
  rank: number
  absolute_momentum: number | null
  relative_momentum: number | null
  technical_confirmed: boolean
  selected: boolean
  close: number | null
  signal: 'long' | null
  entry_est: number | null
  sl_est: number | null
  tp_est: number | null
  sl_atr?: number
  tp_atr?: number
}

export interface MomentumScreenResponse {
  lookback: string
  lookback_bars: number
  as_of: string | null
  n_symbols: number
  config: {
    top_n: number
    use_relative_momentum: boolean
    use_technical_confirmation: boolean
    use_market_regime: boolean
  }
  ranked: MomentumScreenRow[]
  note: string
  cached: boolean
}

export interface AnalysisResponse {
  symbol: string
  years: number
  bars: number
  last_date: string
  interval?: string
  intraday?: boolean
  cap_note?: string | null
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
  /** Both views of the same run; top-level advice/outlook mirror `market_aware`. */
  views: AnalysisViews | null
  key_indicators: KeyIndicators | null
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
  momentum: Momentum | null
  cached?: boolean
}
