import type { ByPatternRow, Detection, EventRow, Level, SignalRow, Trade } from '../types'
import type { Theme } from '../theme'
import { fmt } from './chart'
import { patternInfo } from '../patternInfo'

export function LevelsTable({ levels, theme: t }: { levels: Level[]; theme: Theme }) {
  if (!levels.length) return <p className="empty">No swing levels detected near the current price.</p>
  const rows = [...levels].sort((a, b) => b.price - a.price)
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Level ₹</th>
            <th>Type</th>
            <th className="num">Distance</th>
            <th className="num">Touches</th>
            <th className="num">Strength</th>
            <th>Last tested</th>
            <th>Source</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((lv, i) => (
            <tr key={i}>
              <td className="num">{fmt.price(lv.price)}</td>
              <td style={{ color: lv.kind === 'support' ? t.deltaUpText : t.deltaDownText }}>
                {lv.kind}
              </td>
              <td className="num">
                {lv.distance_pct == null ? '—' : `${lv.distance_pct > 0 ? '+' : ''}${lv.distance_pct.toFixed(1)}%`}
              </td>
              <td className="num">{lv.touches}</td>
              <td className="num">{lv.strength == null ? '—' : `${fmt.int(lv.strength)}/100`}</td>
              <td>{lv.last_touch}</td>
              <td>{lv.source === '52w' ? '52-week extreme' : 'swing cluster'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

const EVENT_NAMES: Record<string, string> = {
  EARNINGS: 'Results announced',
  DIVIDEND: 'Dividend (ex-date)',
  SPLIT: 'Split / bonus',
}

export function EventsTable({ events }: { events: EventRow[] }) {
  if (!events.length) {
    return <p className="empty">No corporate events (earnings / dividends / splits) in this window.</p>
  }
  const recent = [...events].reverse()
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Date</th>
            <th>Event</th>
            <th>Detail</th>
          </tr>
        </thead>
        <tbody>
          {recent.map((ev, i) => (
            <tr key={i}>
              <td>{ev.date}</td>
              <td>{EVENT_NAMES[ev.type] ?? ev.type}</td>
              <td>{ev.label}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

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
            <th>Enter at</th>
            <th className="num">Ref ₹</th>
            <th className="num">Stop ₹</th>
            <th className="num">Target ₹</th>
            <th className="num">SL ×ATR</th>
            <th className="num">TP ×ATR</th>
            <th className="num">Trail ×ATR</th>
          </tr>
        </thead>
        <tbody>
          {signals.map((s, i) => (
            <tr key={i}>
              <td>{s.date ?? '—'}</td>
              <td title={patternInfo(s.tag).meaning}>{s.tag}</td>
              <td>{s.side}</td>
              <td>{s.enter_at ?? '—'}{s.delay_bars ? ` +${fmt.int(s.delay_bars)}bar` : ''}</td>
              <td className="num">{fmt.price(s.ref_close)}</td>
              <td className="num">{fmt.price(s.sl_est)}</td>
              <td className="num">{fmt.price(s.tp_est)}</td>
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

function detectionPlan(d: Detection): string | null {
  if (d.ref_close == null || d.sl_est == null || d.tp_est == null) return null
  const long = d.side !== 'short'
  const slPct = Math.abs(d.sl_est / d.ref_close - 1) * 100
  const tpPct = Math.abs(d.tp_est / d.ref_close - 1) * 100
  const rr = slPct > 0 ? (tpPct / slPct).toFixed(1) : '—'
  return (
    `Plan if acting: ${long ? 'buy' : 'sell short'} near ₹${fmt.price(d.ref_close)}; ` +
    `stop ₹${fmt.price(d.sl_est)} (${long ? '−' : '+'}${slPct.toFixed(1)}%, beyond one normal ` +
    `swing so routine noise doesn't stop you out); target ₹${fmt.price(d.tp_est)} ` +
    `(${long ? '+' : '−'}${tpPct.toFixed(1)}%, ${rr}:1 reward-to-risk).`
  )
}

export function DetectionsList({ detections }: { detections: Detection[] }) {
  const qualified = detections.filter((d) => d.qualified)
  const recent = qualified.slice(-10).reverse()
  if (!recent.length) return <p className="empty">No qualified detections in this window.</p>
  return (
    <ul className="detections">
      {recent.map((d, i) => {
        const info = patternInfo(d.pattern)
        const plan = detectionPlan(d)
        return (
          <li key={i}>
            <span className={`dir-badge dir-${d.direction ?? 'neutral'}`}>
              {d.direction === 'bullish' ? '↑' : d.direction === 'bearish' ? '↓' : '·'}
            </span>
            <span className="detection-text">
              <b>{d.date} — {d.pattern}</b> ({d.direction}, {d.alignment})
              <br />
              {info.meaning} {info.implication}
              {plan && (
                <>
                  <br />
                  {plan}
                </>
              )}
            </span>
          </li>
        )
      })}
    </ul>
  )
}
