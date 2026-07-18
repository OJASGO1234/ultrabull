/** Plain-English knowledge base per pattern: what the shape is, what it
 * suggests, and why the stop/target are placed with ATR multiples. Wording is
 * calibrated to the project's own event-study results — no overclaiming. */

export interface PatternInfo {
  meaning: string
  implication: string
}

export const PATTERN_INFO: Record<string, PatternInfo> = {
  BullishEngulfing: {
    meaning: 'A down day fully engulfed by a bigger up day — buyers overwhelmed the prior day\'s sellers.',
    implication: 'Mild bullish reversal signal; historically strongest when it appears after a dip.',
  },
  BearishEngulfing: {
    meaning: 'An up day fully engulfed by a bigger down day — sellers absorbed all of the prior day\'s buying.',
    implication: 'Bearish; our 4y study shows stocks underperform their trend for ~2 weeks after this (the one short setup with measured edge).',
  },
  Hammer: {
    meaning: 'Price sold off hard intraday but closed near the top, leaving a long lower wick — sellers ran out.',
    implication: 'Bullish reversal, best after weakness; the strongest long pattern in our 4y study (~+21bps/5d edge).',
  },
  Breakout20: {
    meaning: 'Close above the highest high of the last 20 days on elevated volume.',
    implication: 'Trend-continuation long. Only has measured edge when the market is in uptrend AND the stock leads its sector — which is exactly what the gates require.',
  },
  Breakout55: {
    meaning: 'Close above the highest high of the last 55 days on elevated volume — a longer, more significant base.',
    implication: 'Stronger version of Breakout20; same caveat — edge appears only gated and over ~2+ weeks.',
  },
  PullbackToMA20: {
    meaning: 'In an uptrend, price dipped to the 20-day average and closed back above it — a routine trend retest.',
    implication: 'Buy-the-dip continuation; small edge, works best in calm uptrends away from earnings.',
  },
  BB_BuyDip: {
    meaning: 'Price touched the lower Bollinger band while the stock is in an uptrend — statistically stretched to the downside.',
    implication: 'Mean-reversion long: expect a snap back toward the middle band, not a big trend move — hence the tighter target.',
  },
  MACD_Cross_Zeroish_Down: {
    meaning: 'Momentum (MACD) crossed down through its signal line right around the zero line — early trend-roll signal.',
    implication: 'Short setup; small sample in our study but consistently positive when market-gated.',
  },
  RSI_40_Up: { meaning: 'RSI crossed up through 40.', implication: 'Disabled: no measured edge in our 4y study.' },
  RSI_60_Down: { meaning: 'RSI crossed down through 60.', implication: 'Disabled: no measured edge in our 4y study.' },
  Doji: { meaning: 'Open and close nearly equal — indecision.', implication: 'Context only, not tradeable (no direction).' },
  MACD_Cross_Zeroish_Up: {
    meaning: 'MACD crossed up through its signal near zero.',
    implication: 'Disabled: tested significantly NEGATIVE (whipsaw-prone) in our 4y study.',
  },
  BB_SellFade: {
    meaning: 'Price touched the upper Bollinger band in a downtrend.',
    implication: 'Disabled: lost money in the 4y study even after gating.',
  },
  Breakdown20: { meaning: 'Close below the 20-day low on volume.', implication: 'Disabled: shorting breakdowns showed no edge on NSE large-caps.' },
  Breakdown55: { meaning: 'Close below the 55-day low on volume.', implication: 'Disabled: same — no measured short edge.' },
  VCP_Setup: {
    meaning: 'Volatility contraction near the 52-week high: shrinking daily ranges and drying volume in an uptrend — the stock is "coiling".',
    implication: 'Context only, not traded: our study showed the coil itself has no entry edge — the tradeable event is the breakout FROM it. Treat as a watchlist flag for an upcoming Breakout20/55.',
  },
}

export const patternInfo = (p: string): PatternInfo =>
  PATTERN_INFO[p] ?? { meaning: p, implication: '' }
