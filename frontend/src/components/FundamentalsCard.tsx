import type { Fundamentals, FyPoint } from '../types'
import type { Theme } from '../theme'
import { fmt } from './chart'

interface Props {
  fundamentals: Fundamentals
  theme: Theme
}

const flagColor = (t: Theme, level: string) =>
  level === 'good' ? t.deltaUpText : level === 'bad' ? t.deltaDownText : t.warning

const flagMark = (level: string) => (level === 'good' ? '✓' : level === 'bad' ? '✗' : '!')

function trendCells(points: FyPoint[] | undefined, t: Theme) {
  if (!points?.length) return null
  return points.map((p, i) => {
    const prev = i > 0 ? points[i - 1].value_cr : null
    const color =
      prev == null || prev === 0 ? undefined : p.value_cr >= prev ? t.deltaUpText : t.deltaDownText
    return (
      <td key={p.fy} className="num" style={i > 0 ? { color } : undefined}>
        {fmt.compact(p.value_cr)}
      </td>
    )
  })
}

/** Fundamental context: holding structure, valuation snapshot, and multi-year
 * revenue/profit/debt trends with plain flags. Display-only — the timing
 * engine deliberately does not trade on these; they're the user's
 * fundamental-research companion. */
export function FundamentalsCard({ fundamentals: f, theme: t }: Props) {
  if (!f.available) {
    return (
      <div className="card">
        <h2>Fundamentals</h2>
        <p className="empty">No fundamental data available from Yahoo for this symbol.</p>
      </div>
    )
  }

  const s = f.snapshot
  const years = (f.annual.revenue ?? f.annual.net_income ?? []).map((p) => p.fy)

  const tiles: { label: string; value: string; note?: string }[] = [
    { label: 'Market cap', value: s.market_cap_cr == null ? '—' : `₹${fmt.compact(s.market_cap_cr)} cr` },
    { label: 'P/E (trailing)', value: s.trailing_pe == null ? '—' : s.trailing_pe.toFixed(1) },
    { label: 'P/B', value: s.price_to_book == null ? '—' : s.price_to_book.toFixed(1) },
    {
      label: 'Debt / equity',
      value: s.debt_to_equity == null ? '—' : `${(s.debt_to_equity / 100).toFixed(2)}x`,
    },
    {
      label: 'Promoter / insiders',
      value: f.promoter.holding_pct == null ? 'n/a' : `${f.promoter.holding_pct.toFixed(1)}%`,
      note: f.promoter.holding_pct == null ? 'not reported by Yahoo' : undefined,
    },
    {
      label: 'Institutions',
      value: f.promoter.institutions_pct == null ? 'n/a' : `${f.promoter.institutions_pct.toFixed(1)}%`,
    },
    {
      label: 'Pledged shares',
      value: f.promoter.pledge_pct == null ? 'n/a' : `${f.promoter.pledge_pct.toFixed(1)}%`,
      note: f.promoter.pledge_pct == null ? 'needs NSE shareholding feed' : undefined,
    },
  ]

  return (
    <div className="card">
      <h2>Fundamentals — trend flags (display-only)</h2>
      <p className="chart-note">
        The timing engine deliberately does not trade on these — fundamentals decide <i>what</i> to
        own; the technical layers above decide <i>when</i>. Red flags here are a reason to question a
        long thesis regardless of the chart.
      </p>

      {f.flags.length > 0 && (
        <div className="chip-row">
          {f.flags.map((fl, i) => (
            <span key={i} className="chip chip-on" style={{ color: flagColor(t, fl.level), borderColor: flagColor(t, fl.level) }}>
              {flagMark(fl.level)} {fl.text}
            </span>
          ))}
        </div>
      )}

      <div className="tiles">
        {tiles.map((tile) => (
          <div className="tile" key={tile.label}>
            <div className="tile-label">{tile.label}</div>
            <div className="tile-value" style={{ fontSize: 16 }}>{tile.value}</div>
            {tile.note && <div className="tile-note">{tile.note}</div>}
          </div>
        ))}
      </div>

      {years.length > 0 && (
        <div className="table-wrap" style={{ marginTop: 10 }}>
          <table>
            <thead>
              <tr>
                <th>₹ crore (FY)</th>
                {years.map((y) => <th key={y} className="num">{y}</th>)}
              </tr>
            </thead>
            <tbody>
              {f.annual.revenue?.length ? <tr><td>Revenue</td>{trendCells(f.annual.revenue, t)}</tr> : null}
              {f.annual.net_income?.length ? <tr><td>Net profit</td>{trendCells(f.annual.net_income, t)}</tr> : null}
              {f.annual.net_margin?.length ? (
                <tr>
                  <td>Net margin</td>
                  {f.annual.net_margin.map((p) => (
                    <td key={p.fy} className="num">{p.value_pct.toFixed(1)}%</td>
                  ))}
                </tr>
              ) : null}
              {f.annual.total_debt?.length ? <tr><td>Total debt</td>{trendCells(f.annual.total_debt, t)}</tr> : null}
            </tbody>
          </table>
        </div>
      )}

      {f.quarterly.net_income && (
        <p className="chart-note">
          Latest quarter ({f.quarterly.net_income.quarter}): net profit ₹{fmt.compact(f.quarterly.net_income.value_cr)} cr
          {f.quarterly.net_income.yoy_pct != null && ` (${f.quarterly.net_income.yoy_pct > 0 ? '+' : ''}${f.quarterly.net_income.yoy_pct.toFixed(0)}% YoY)`}
          {f.quarterly.revenue?.yoy_pct != null && ` · revenue ${f.quarterly.revenue.yoy_pct > 0 ? '+' : ''}${f.quarterly.revenue.yoy_pct.toFixed(0)}% YoY`}
        </p>
      )}

      <p className="chart-note">{f.promoter.source_note} Source: Yahoo Finance statements — cross-check material numbers before acting.</p>
    </div>
  )
}
