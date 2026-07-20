import { useCallback, useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { AreaSeries, type Time } from 'lightweight-charts'
import type { Theme } from '../theme'
import { fmt } from './chart'
import { makeChart } from './chart'
import {
  fetchPaper, fetchPaperHistory, paperTrade, resetPaper, setPaperConfig,
  type PaperHistory, type PaperState,
} from '../paper'

function EquityCurve({ data, theme: t }: { data: { date: string; equity: number }[]; theme: Theme }) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = ref.current
    if (!el || data.length < 2) return
    const chart = makeChart(el, t, 220)
    const s = chart.addSeries(AreaSeries, {
      lineColor: t.blue, topColor: t.areaWash, bottomColor: 'transparent',
      lineWidth: 2, lastValueVisible: true, priceLineVisible: false,
    })
    s.setData(data.map((p) => ({ time: p.date as Time, value: p.equity })))
    chart.timeScale().fitContent()
    return () => chart.remove()
  }, [data, t])
  if (data.length < 2) return <p className="chart-note">Equity curve appears once there are a few daily snapshots.</p>
  return <div ref={ref} className="chart" />
}

export function PaperPage({ theme: t }: { theme: Theme }) {
  const [state, setState] = useState<PaperState | null>(null)
  const [history, setHistory] = useState<PaperHistory | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [flash, setFlash] = useState<string | null>(null)

  // trade form
  const [symbol, setSymbol] = useState('')
  const [side, setSide] = useState<'BUY' | 'SELL'>('BUY')
  const [qty, setQty] = useState('')
  const [price, setPrice] = useState('')
  const [stop, setStop] = useState('')
  const [target, setTarget] = useState('')
  const [busy, setBusy] = useState(false)

  // config form
  const [cap, setCap] = useState('')
  const [brok, setBrok] = useState('')
  const [tax, setTax] = useState('')
  const [kill, setKill] = useState('')

  const pnlColor = (v: number | null | undefined) =>
    v == null ? undefined : { color: v >= 0 ? t.deltaUpText : t.deltaDownText }

  const load = useCallback(async () => {
    try {
      const s = await fetchPaper()
      setState(s)
      setError(null)
      if (!cap) setCap(String(s.config.starting_capital))
      if (!brok) setBrok(String(s.config.brokerage_pct))
      if (!tax) setTax(String(s.config.tax_pct))
      if (!kill) setKill(String(s.config.kill_drawdown_pct))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
    try {
      setHistory(await fetchPaperHistory())
    } catch {
      // history optional
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    void load()
    const id = window.setInterval(() => {
      if (!document.hidden) void load()
    }, 15_000) // live mark-to-market while the tab is open
    return () => window.clearInterval(id)
  }, [load])

  const yahooOf = (s: string) => {
    const u = s.trim().toUpperCase()
    if (!u) return null
    return /\.(NS|BO)$/.test(u) ? u : `${u}.NS`
  }

  const onTrade = async (e: FormEvent) => {
    e.preventDefault()
    const q = Number(qty)
    if (!symbol.trim() || !q || q <= 0 || busy) return
    setBusy(true)
    setError(null)
    setFlash(null)
    try {
      const ysym = yahooOf(symbol)
      const r = await paperTrade({
        symbol: symbol.trim().toUpperCase().replace(/\.(NS|BO)$/, ''),
        yahoo_symbol: ysym,
        side,
        qty: q,
        price: price ? Number(price) : null,
        stop: stop ? Number(stop) : null,
        target: target ? Number(target) : null,
      })
      setState(r.state)
      setFlash(
        `${side} ${q} ${symbol.trim().toUpperCase()} @ ₹${fmt.price(r.price)} · charges ₹${fmt.price(r.charges)}` +
          (r.realized_pnl != null ? ` · realized ${r.realized_pnl >= 0 ? '+' : ''}₹${fmt.int(r.realized_pnl)}` : ''),
      )
      setQty(''); setPrice(''); setStop(''); setTarget('')
      setHistory(await fetchPaperHistory())
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const quickSell = async (sym: string, ysym: string | null, q: number) => {
    setError(null)
    try {
      const r = await paperTrade({ symbol: sym, yahoo_symbol: ysym, side: 'SELL', qty: q })
      setState(r.state)
      setFlash(`Closed ${sym} · realized ${(r.realized_pnl ?? 0) >= 0 ? '+' : ''}₹${fmt.int(r.realized_pnl ?? 0)}`)
      setHistory(await fetchPaperHistory())
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const saveConfig = async () => {
    setError(null)
    try {
      const r = await setPaperConfig({
        starting_capital: cap ? Number(cap) : undefined,
        brokerage_pct: brok ? Number(brok) : undefined,
        tax_pct: tax ? Number(tax) : undefined,
        kill_drawdown_pct: kill ? Number(kill) : undefined,
      })
      if (r.capital_locked) setFlash('Costs saved. Starting capital is locked once trades exist — reset to change it.')
      else setFlash('Settings saved.')
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const onReset = async () => {
    if (!window.confirm('Reset the paper account? All trades, positions and history are erased.')) return
    await resetPaper()
    setCap(''); setBrok(''); setTax(''); setKill('')
    await load()
  }

  if (!state) {
    return (
      <main className="results">
        <div className="card">{error ? <div className="error-banner">⚠ {error}</div> : <p className="empty">Loading…</p>}</div>
      </main>
    )
  }

  const s = state
  return (
    <main className="results">
      <div className="card">
        <div className="card-head">
          <h2>
            Paper trading
            {s.market_open && <span className="live-readout" style={{ marginLeft: 10 }}><span className="live-dot" /> live</span>}
          </h2>
          <button type="button" className="btn-ghost" onClick={onReset}>Reset account</button>
        </div>
        <p className="chart-note">
          Forward validation with real live prices and no real money. Equity / ETF / REIT delivery only. Costs below
          are charged on every fill and baked into P&L, so what you see is after-cost. {s.stats.kill_breached && (
            <b style={{ color: t.deltaDownText }}> ⚠ Kill-line breached: drawdown {s.stats.current_drawdown_pct}% ≥ {s.config.kill_drawdown_pct}%.</b>
          )}
        </p>
        <div className="tiles">
          <div className="tile"><div className="tile-label">Equity</div><div className="tile-value">₹{fmt.compact(s.equity)}</div>
            <div className="tile-note" style={pnlColor(s.total_pnl)}>{s.total_pnl >= 0 ? '+' : ''}₹{fmt.compact(s.total_pnl)} ({s.total_return_pct}%)</div></div>
          <div className="tile"><div className="tile-label">Cash</div><div className="tile-value">₹{fmt.compact(s.cash)}</div>
            <div className="tile-note">holdings ₹{fmt.compact(s.holdings_value)}</div></div>
          <div className="tile"><div className="tile-label">Realized P&L</div><div className="tile-value" style={pnlColor(s.realized_pnl)}>{s.realized_pnl >= 0 ? '+' : ''}₹{fmt.compact(s.realized_pnl)}</div>
            <div className="tile-note">unrealized <span style={pnlColor(s.unrealized_pnl)}>{s.unrealized_pnl >= 0 ? '+' : ''}₹{fmt.compact(s.unrealized_pnl)}</span></div></div>
          <div className="tile"><div className="tile-label">Win rate</div><div className="tile-value">{s.stats.win_rate != null ? `${Math.round(s.stats.win_rate * 100)}%` : '—'}</div>
            <div className="tile-note">{s.stats.closed_trades} closed · max DD {s.stats.max_drawdown_pct}%</div></div>
        </div>
      </div>

      <div className="card">
        <h2>Trade</h2>
        {flash && <p className="chart-note" style={{ color: t.deltaUpText }}>{flash}</p>}
        {error && <p className="detail-warn">{error}</p>}
        <form className="controls paper-trade" onSubmit={onTrade}>
          <input value={symbol} onChange={(e) => setSymbol(e.target.value)} placeholder="Symbol e.g. ITC" aria-label="Symbol" spellCheck={false} />
          <select value={side} onChange={(e) => setSide(e.target.value as 'BUY' | 'SELL')} aria-label="Side">
            <option value="BUY">Buy</option><option value="SELL">Sell</option>
          </select>
          <input value={qty} onChange={(e) => setQty(e.target.value)} placeholder="Qty" type="number" min="1" aria-label="Quantity" />
          <input value={price} onChange={(e) => setPrice(e.target.value)} placeholder="Price (blank = live)" type="number" step="0.05" aria-label="Price" />
          {side === 'BUY' && <input value={stop} onChange={(e) => setStop(e.target.value)} placeholder="Stop (opt)" type="number" step="0.05" aria-label="Stop" />}
          {side === 'BUY' && <input value={target} onChange={(e) => setTarget(e.target.value)} placeholder="Target (opt)" type="number" step="0.05" aria-label="Target" />}
          <button type="submit" disabled={busy || !symbol.trim() || !qty}>{busy ? '…' : side === 'BUY' ? 'Buy' : 'Sell'}</button>
        </form>
        <p className="chart-note">Leave price blank to fill at the live Groww price (market hours). Stop/target are for your reference and hit-flags; they don't auto-execute.</p>
      </div>

      <div className="card">
        <h2>Open positions ({s.positions.length})</h2>
        {s.positions.length === 0 ? <p className="empty">No open paper positions.</p> : (
          <div className="table-wrap">
            <table>
              <thead><tr><th>Symbol</th><th className="num">Qty</th><th className="num">Avg ₹</th><th className="num">LTP ₹</th><th className="num">Value ₹</th><th className="num">Unrealized</th><th className="num">Stop / Target</th><th></th></tr></thead>
              <tbody>
                {s.positions.map((p) => (
                  <tr key={p.symbol}>
                    <td><b>{p.symbol}</b></td>
                    <td className="num">{fmt.int(p.qty)}</td>
                    <td className="num">{fmt.price(p.avg_cost)}</td>
                    <td className="num">{fmt.price(p.ltp)}</td>
                    <td className="num">{fmt.compact(p.market_value)}</td>
                    <td className="num" style={pnlColor(p.unrealized_pnl)}>
                      {p.unrealized_pnl == null ? '—' : `${p.unrealized_pnl >= 0 ? '+' : ''}${fmt.int(p.unrealized_pnl)}`}
                      {p.unrealized_pct != null && <span className="cell-sub"> {p.unrealized_pct >= 0 ? '+' : ''}{p.unrealized_pct}%</span>}
                    </td>
                    <td className="num">
                      {p.stop != null ? fmt.price(p.stop) : '—'} / {p.target != null ? fmt.price(p.target) : '—'}
                      {p.stop_hit && <span title="Stop hit" style={{ color: t.deltaDownText }}> ⚠stop</span>}
                      {p.target_hit && <span title="Target hit" style={{ color: t.deltaUpText }}> ✓target</span>}
                    </td>
                    <td><button type="button" className="btn-mini" onClick={() => quickSell(p.symbol, p.yahoo_symbol, p.qty)}>Close</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="card">
        <h2>Performance</h2>
        <EquityCurve data={history?.equity_curve ?? []} theme={t} />
      </div>

      <div className="card">
        <h2>Trade log ({history?.trades.length ?? 0})</h2>
        {history && history.trades.length > 0 ? (
          <div className="table-wrap">
            <table>
              <thead><tr><th>When</th><th>Symbol</th><th>Side</th><th className="num">Qty</th><th className="num">Price ₹</th><th className="num">Charges ₹</th><th className="num">Realized ₹</th></tr></thead>
              <tbody>
                {history.trades.map((tr, i) => (
                  <tr key={i}>
                    <td className="cell-sub">{tr.ts.slice(0, 16).replace('T', ' ')}</td>
                    <td><b>{tr.symbol}</b></td>
                    <td style={{ color: tr.side === 'BUY' ? t.blue : t.orange }}>{tr.side}</td>
                    <td className="num">{fmt.int(tr.qty)}</td>
                    <td className="num">{fmt.price(tr.price)}</td>
                    <td className="num cell-sub">{fmt.price(tr.charges)}</td>
                    <td className="num" style={pnlColor(tr.realized_pnl)}>{tr.realized_pnl == null ? '—' : `${tr.realized_pnl >= 0 ? '+' : ''}${fmt.int(tr.realized_pnl)}`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <p className="empty">No trades yet.</p>}
      </div>

      <div className="card">
        <h2>Settings</h2>
        <p className="chart-note">Brokerage% and tax% are charged on each transaction's value (both applied per fill). Set them to approximate your real all-in cost. Starting capital can only change while the account has no trades.</p>
        <div className="controls">
          <label className="paper-cfg">Starting ₹ <input value={cap} onChange={(e) => setCap(e.target.value)} type="number" /></label>
          <label className="paper-cfg">Brokerage % <input value={brok} onChange={(e) => setBrok(e.target.value)} type="number" step="0.01" /></label>
          <label className="paper-cfg">Tax % <input value={tax} onChange={(e) => setTax(e.target.value)} type="number" step="0.01" /></label>
          <label className="paper-cfg">Kill drawdown % <input value={kill} onChange={(e) => setKill(e.target.value)} type="number" step="0.5" /></label>
          <button type="button" onClick={saveConfig}>Save settings</button>
        </div>
      </div>
    </main>
  )
}
