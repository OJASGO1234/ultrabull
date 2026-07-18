import type { Metrics } from '../types'
import type { Theme } from '../theme'
import { fmt } from './chart'

interface Props {
  metrics: Metrics
  theme: Theme
}

/** Stat tiles: label + value; delta-colored text only where sign = good/bad. */
export function MetricsCards({ metrics: m, theme: t }: Props) {
  const deltaColor = (v: number | null | undefined) =>
    v == null ? undefined : v >= 0 ? t.deltaUpText : t.deltaDownText

  const tiles: { label: string; value: string; color?: string }[] = [
    { label: 'Trades', value: String(m.trades) },
    { label: 'Win rate', value: fmt.pct(m.win_rate) },
    { label: 'Profit factor', value: fmt.ratio(m.profit_factor) },
    {
      label: 'Expectancy / trade',
      value: m.expectancy == null ? '—' : `₹${fmt.int(m.expectancy)}`,
      color: deltaColor(m.expectancy),
    },
    {
      label: 'Total return',
      value: fmt.pct(m.return_total, 2),
      color: deltaColor(m.return_total),
    },
    { label: 'Max drawdown', value: fmt.pct(m.max_drawdown, 2), color: deltaColor(m.max_drawdown) },
    { label: 'Sharpe (daily)', value: m.sharpe_daily == null ? '—' : m.sharpe_daily.toFixed(2) },
    { label: 'Median hold', value: m.median_hold_bars == null ? '—' : `${fmt.price(m.median_hold_bars, 0)} bars` },
  ]

  return (
    <div className="tiles">
      {tiles.map((tile) => (
        <div className="tile" key={tile.label}>
          <div className="tile-label">{tile.label}</div>
          <div className="tile-value" style={tile.color ? { color: tile.color } : undefined}>
            {tile.value}
          </div>
        </div>
      ))}
    </div>
  )
}
