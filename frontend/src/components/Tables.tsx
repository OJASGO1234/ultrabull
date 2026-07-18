import type { ByPatternRow, Detection, SignalRow, Trade } from '../types'
import type { Theme } from '../theme'
import { fmt } from './chart'

const pnlStyle = (t: Theme, v: number | null) =>
  v == null ? undefined : { color: v >= 0 ? t.deltaUpText : t.deltaDownText }

export function TradesTable({ trades, theme: t }: { trades: Trade[]; theme: Theme }) {
  if (!trades.length) return <p className="empty">No trades were taken in this window.</p>
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Entry</th>
            <th>Exit</th>
            <th>Side</th>
            <th>Pattern</th>
            <th className="num">Entry ₹</th>
            <th className="num">Exit ₹</th>
            <th className="num">Shares</th>
            <th className="num">P&L ₹</th>
            <th>Exit reason</th>
          </tr>
        </thead>
        <tbody>
          {trades.map((tr, i) => (
            <tr key={i}>
              <td>{tr.entry_date ?? '—'}</td>
              <td>{tr.exit_date ?? '—'}</td>
              <td>{tr.side}</td>
              <td>{tr.tag}</td>
              <td className="num">{fmt.price(tr.entry_price)}</td>
              <td className="num">{fmt.price(tr.exit_price)}</td>
              <td className="num">{fmt.int(tr.size_shares)}</td>
              <td className="num" style={pnlStyle(t, tr.pnl)}>{fmt.int(tr.pnl)}</td>
              <td>{tr.reason}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function ByPatternTable({ rows, theme: t }: { rows: ByPatternRow[]; theme: Theme }) {
  if (!rows.length) return <p className="empty">No per-pattern stats (no closed trades).</p>
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Pattern</th>
            <th className="num">Trades</th>
            <th className="num">Win rate</th>
            <th className="num">Total P&L ₹</th>
            <th className="num">Avg P&L ₹</th>
            <th className="num">Profit factor</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.tag}>
              <td>{r.tag}</td>
              <td className="num">{r.trades}</td>
              <td className="num">{fmt.pct(r.win_rate, 0)}</td>
              <td className="num" style={pnlStyle(t, r.total_pnl)}>{fmt.int(r.total_pnl)}</td>
              <td className="num">{fmt.int(r.avg_pnl)}</td>
              <td className="num">{fmt.ratio(r.profit_factor)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function SignalsTable({ signals }: { signals: SignalRow[] }) {
  if (!signals.length) return <p className="empty">No executable signals were compiled.</p>
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Date</th>
            <th>Pattern</th>
            <th>Side</th>
            <th className="num">Score</th>
            <th>Enter at</th>
            <th className="num">Delay</th>
            <th className="num">SL ×ATR</th>
            <th className="num">TP ×ATR</th>
            <th className="num">Trail ×ATR</th>
          </tr>
        </thead>
        <tbody>
          {signals.map((s, i) => (
            <tr key={i}>
              <td>{s.date ?? '—'}</td>
              <td>{s.tag}</td>
              <td>{s.side}</td>
              <td className="num">{fmt.int(s.score)}</td>
              <td>{s.enter_at ?? '—'}</td>
              <td className="num">{fmt.int(s.delay_bars)}</td>
              <td className="num">{fmt.price(s.sl_atr, 1)}</td>
              <td className="num">{fmt.price(s.tp_atr, 1)}</td>
              <td className="num">{fmt.price(s.trail_atr, 1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function DetectionsList({ detections }: { detections: Detection[] }) {
  const qualified = detections.filter((d) => d.qualified)
  const recent = qualified.slice(-10).reverse()
  if (!recent.length) return <p className="empty">No qualified detections in this window.</p>
  return (
    <ul className="detections">
      {recent.map((d, i) => (
        <li key={i}>
          <span className={`dir-badge dir-${d.direction ?? 'neutral'}`}>
            {d.direction === 'bullish' ? '↑' : d.direction === 'bearish' ? '↓' : '·'}
          </span>
          <span className="detection-text">{d.english ?? `${d.date} — ${d.pattern}`}</span>
        </li>
      ))}
    </ul>
  )
}
