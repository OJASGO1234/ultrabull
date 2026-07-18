import type { Outlook, OutlookHorizon } from '../types'
import type { Theme } from '../theme'
import { fmt } from './chart'

interface Props {
  outlook: Outlook
  theme: Theme
}

const HORIZON_LABELS: Record<string, string> = {
  '1d': 'Today',
  '1w': '1 week',
  '1m': '1 month',
  '3m': '3 months',
}

const VERDICT_TEXT: Record<string, string> = {
  up: '↑ Up bias',
  down: '↓ Down bias',
  sideways: '→ Sideways',
  unknown: '—',
}

function verdictColor(t: Theme, v: string): string | undefined {
  if (v === 'up') return t.deltaUpText
  if (v === 'down') return t.deltaDownText
  return undefined
}

function HorizonTile({ name, h, theme: t }: { name: string; h: OutlookHorizon; theme: Theme }) {
  const [lo, hi] = h.typical_range
  return (
    <div className="tile">
      <div className="tile-label">{HORIZON_LABELS[name] ?? name}</div>
      <div className="tile-value" style={{ color: verdictColor(t, h.verdict) }}>
        {VERDICT_TEXT[h.verdict]}
      </div>
      <div className="tile-note">
        {h.p_up != null && `${Math.round(h.p_up * 100)}% of similar setups closed higher`}
        {h.median_move != null && ` · median ${fmt.pct(h.median_move, 1)}`}
        {lo != null && hi != null && ` · typical ${fmt.pct(lo, 1)} to ${fmt.pct(hi, 1)}`}
        {h.n != null && ` · n=${fmt.int(h.n)}`}
      </div>
    </div>
  )
}

const stateWords = (o: Outlook) =>
  [
    `NIFTY ${o.state.market === 'up' ? 'uptrend' : 'downtrend'}`,
    `stock trend ${o.state.trend === 'up' ? 'up' : 'down'}`,
    o.state.rs === '+' ? 'leading sector' : 'lagging sector',
    `RSI ${o.state.rsi} (${o.state.rsi_band})`,
  ].join(' · ')

/** Empirical bias per horizon (state-conditioned base rates), plus the
 * scenario frame: nearest support/resistance as trigger and invalidation. */
export function OutlookCard({ outlook: o, theme: t }: Props) {
  const order: Array<'1d' | '1w' | '1m' | '3m'> = ['1d', '1w', '1m', '3m']
  return (
    <div className="card">
      <div className="card-head">
        <h2>Outlook — base rates for today's setup</h2>
        <span className="hover-readout">{stateWords(o)}</span>
      </div>
      <div className="tiles">
        {order.map((name) => (
          <HorizonTile key={name} name={name} h={o.horizons[name]} theme={t} />
        ))}
      </div>
      {(o.nearest_support != null || o.nearest_resistance != null) && (
        <p className="chart-note">
          Trade frame:{' '}
          {o.nearest_resistance != null && (
            <>bullish on a close above <b>{fmt.price(o.nearest_resistance)}</b> (nearest resistance)</>
          )}
          {o.nearest_support != null && o.nearest_resistance != null && ' · '}
          {o.nearest_support != null && (
            <>bias invalidated on a close below <b>{fmt.price(o.nearest_support)}</b> (nearest support)</>
          )}
        </p>
      )}
      <p className="chart-note">{o.note}</p>
    </div>
  )
}
