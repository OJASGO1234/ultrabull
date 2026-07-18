import type { MarketContext, Timing } from '../types'
import type { Theme } from '../theme'
import { fmt } from './chart'

interface Props {
  context: MarketContext
  timing: Timing
  theme: Theme
}

const regimeColor = (t: Theme, regime: string | null) =>
  regime === 'uptrend' ? t.deltaUpText : regime === 'downtrend' ? t.deltaDownText : undefined

const regimeLabel = (regime: string | null) =>
  regime === 'uptrend' ? '↑ Uptrend' : regime === 'downtrend' ? '↓ Downtrend' : '· Neutral'

/** Timing intelligence: where the market/sector stands and where we are in the
 * results cycle. This is the "should I even be pressing buy this week?" card —
 * it mirrors the exact gates the signal engine applies. */
export function ContextCard({ context: c, timing: tm, theme: t }: Props) {
  const rsColor = (v: number | null) => (v == null ? undefined : v >= 0 ? t.deltaUpText : t.deltaDownText)

  const justReported =
    !tm.earnings_data_stale &&
    tm.days_since_last_earnings != null &&
    tm.days_since_last_earnings <= 5

  const earningsValue = () => {
    if (tm.next_earnings) {
      const d = tm.trading_days_to_next_earnings
      return d != null ? `${tm.next_earnings} (${d} td)` : tm.next_earnings
    }
    if (justReported) return 'just reported'
    if (tm.earnings_data_stale) return 'unknown'
    if (tm.results_overdue) return 'due any day'
    return '—'
  }

  const tiles: { label: string; value: string; color?: string; note?: string }[] = [
    {
      label: 'Market (NIFTY)',
      value: regimeLabel(c.market_regime),
      color: regimeColor(t, c.market_regime),
      note: c.market_regime === 'downtrend' ? 'long entries gated off' : undefined,
    },
    {
      label: c.sector ? `Sector (${c.sector})` : 'Sector',
      value: regimeLabel(c.sector_regime),
      color: regimeColor(t, c.sector_regime),
    },
    {
      label: 'Rel. strength 20d',
      value: c.rs_20 == null ? '—' : fmt.pct(c.rs_20, 1),
      color: rsColor(c.rs_20),
      note: c.rs_20 != null && c.rs_20 < 0 ? 'lagging sector — breakouts gated' : 'vs sector index',
    },
    {
      label: 'Rel. strength 60d',
      value: c.rs_60 == null ? '—' : fmt.pct(c.rs_60, 1),
      color: rsColor(c.rs_60),
    },
    {
      label: 'Next results',
      value: earningsValue(),
      color: tm.in_earnings_blackout ? t.critical : tm.results_overdue ? t.warning : undefined,
      note: tm.in_earnings_blackout
        ? 'blackout: no fresh entries'
        : justReported
          ? 'watch post-results volatility'
          : tm.results_overdue
            ? `last results ${tm.days_since_last_earnings}d ago`
            : tm.earnings_data_stale
              ? 'no reliable dates from Yahoo'
              : undefined,
    },
    {
      label: 'Last results',
      value: tm.last_earnings && !tm.earnings_data_stale
        ? `${tm.last_earnings.date} (${tm.days_since_last_earnings}d ago)`
        : '—',
    },
    {
      label: 'Dividend',
      value: tm.next_ex_dividend
        ? `ex-date ${tm.next_ex_dividend}`
        : tm.last_dividend
          ? `${tm.last_dividend.label} on ${tm.last_dividend.date}`
          : '—',
      note: tm.next_ex_dividend ? 'upcoming' : tm.last_dividend ? 'most recent' : undefined,
    },
    {
      label: 'ADX (trend strength)',
      value: c.adx == null ? '—' : c.adx.toFixed(1),
      note: c.adx == null ? undefined : c.adx >= 25 ? 'trending' : c.adx <= 15 ? 'rangebound' : 'transitional',
    },
    {
      label: 'Market breadth',
      value: c.breadth == null ? '—' : `${Math.round(c.breadth * 100)}%`,
      color: c.breadth == null ? undefined : c.breadth >= 0.55 ? t.deltaUpText : c.breadth <= 0.35 ? t.deltaDownText : undefined,
      note: 'universe above own 50DMA',
    },
    {
      label: 'Distribution days',
      value: c.dist_days == null ? '—' : `${Math.round(c.dist_days)} / 25d`,
      color: c.dist_days == null ? undefined : c.dist_days >= 5 ? t.critical : c.dist_days >= 4 ? t.warning : undefined,
      note: c.dist_days != null && c.dist_days >= 5 ? 'institutional selling — longs gated off' : 'NIFTY down days on rising volume',
    },
  ]

  return (
    <div className="card">
      <h2>Timing context — the gates behind today's signals</h2>
      <div className="tiles">
        {tiles.map((tile) => (
          <div className="tile" key={tile.label}>
            <div className="tile-label">{tile.label}</div>
            <div className="tile-value" style={tile.color ? { color: tile.color } : undefined}>
              {tile.value}
            </div>
            {tile.note && <div className="tile-note">{tile.note}</div>}
          </div>
        ))}
      </div>
    </div>
  )
}
