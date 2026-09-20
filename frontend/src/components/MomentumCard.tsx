import type { Momentum } from '../types'
import type { Theme } from '../theme'
import { fmt } from './chart'

interface Props {
  momentum: Momentum
  theme: Theme
}

const LOOKBACK_ORDER: Array<'3M' | '6M' | '9M' | '12M'> = ['3M', '6M', '9M', '12M']

/** Diagnostic panel for the momentum / relative-strength strategy
 * (core/momentum.py + core/strategies/momentum_strategy.py): absolute and
 * relative momentum at each of 3/6/9/12 months, plus the same technical
 * trend-confirmation flags the strategy trades on. This is a per-symbol
 * readout, not the cross-sectional rank/signal (that lives in the
 * GET /api/momentum/screen universe scan, not fetched here) — so no
 * entry/stop/target is shown unless/until this symbol is looked up there. */
export function MomentumCard({ momentum: m, theme: t }: Props) {
  const posColor = (v: number | null) =>
    v == null ? t.muted : v > 0 ? t.deltaUpText : v < 0 ? t.deltaDownText : t.textSecondary

  const tech = m.technical_confirmation
  const techRows: Array<[string, boolean]> = [
    ['Close > MA20', tech.close_above_ma20],
    ['Close > MA50', tech.close_above_ma50],
    ['MA20 > MA50', tech.ma20_above_ma50],
  ]

  return (
    <div className="card">
      <div className="card-head">
        <h2>Momentum / relative strength</h2>
        <span className="hover-readout">as of {m.as_of} · vs {m.benchmark}</span>
      </div>

      <div className="tiles">
        {LOOKBACK_ORDER.map((label) => {
          const lb = m.lookbacks[label]
          if (!lb) return null
          return (
            <div className="tile" key={label}>
              <div className="tile-label">{label} momentum ({lb.months}mo / {lb.bars} bars)</div>
              <div className="tile-value" style={{ color: posColor(lb.absolute_momentum) }}>
                {fmt.pct(lb.absolute_momentum)}
              </div>
              <div className="tile-note">
                relative to {m.benchmark}:{' '}
                <span style={{ color: posColor(lb.relative_momentum) }}>
                  {fmt.pct(lb.relative_momentum)}
                </span>
              </div>
            </div>
          )
        })}
      </div>

      <ul className="factor-list">
        {techRows.map(([label, ok]) => (
          <li key={label}>
            <span className="factor-mark" style={{ color: ok ? t.deltaUpText : t.deltaDownText }}>
              {ok ? '✓' : '✗'}
            </span>
            <b>{label}</b>
          </li>
        ))}
        <li>
          <span className="factor-mark" style={{ color: tech.confirmed ? t.deltaUpText : t.muted }}>
            {tech.confirmed ? '✓' : '·'}
          </span>
          <b>Technical trend confirmed</b> — {tech.confirmed
            ? 'this symbol currently passes the strategy\'s trend filter'
            : 'this symbol does not currently pass the strategy\'s trend filter'}
        </li>
      </ul>

      <p className="chart-note">{m.note}</p>
    </div>
  )
}
