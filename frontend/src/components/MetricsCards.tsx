import type { Metrics } from '../types'
import type { Theme } from '../theme'
import { fmt } from './chart'
import { MetricCard, type Tone } from './ui'

interface Props {
  metrics: Metrics
  theme: Theme
}

/** Stat tiles: label + value; delta-colored text only where sign = good/bad. */
export function MetricsCards({ metrics: m }: Props) {
  const deltaTone = (v: number | null | undefined): Tone | undefined =>
    v == null ? undefined : v >= 0 ? 'up' : 'down'

  const tiles: { label: string; value: string; tone?: Tone }[] = [
    { label: 'Trades', value: String(m.trades) },
    { label: 'Win rate', value: fmt.pct(m.win_rate) },
    { label: 'Profit factor', value: fmt.ratio(m.profit_factor) },
    {
      label: 'Expectancy / trade',
      value: m.expectancy == null ? '—' : `₹${fmt.int(m.expectancy)}`,
      tone: deltaTone(m.expectancy),
    },
    {
      label: 'Total return',
      value: fmt.pct(m.return_total, 2),
      tone: deltaTone(m.return_total),
    },
    { label: 'Max drawdown', value: fmt.pct(m.max_drawdown, 2), tone: deltaTone(m.max_drawdown) },
    { label: 'Sharpe (daily)', value: m.sharpe_daily == null ? '—' : m.sharpe_daily.toFixed(2) },
    { label: 'Median hold', value: m.median_hold_bars == null ? '—' : `${fmt.price(m.median_hold_bars, 0)} bars` },
  ]

  return (
    <div className="tiles">
      {tiles.map((tile) => (
        <MetricCard key={tile.label} label={tile.label} display={tile.value} tone={tile.tone} />
      ))}
    </div>
  )
}
