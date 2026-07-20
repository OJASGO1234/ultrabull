import { useState } from 'react'
import type { Theme } from '../theme'
import type { LiveQuote } from '../live'
import type { Advice } from '../types'
import { fmt } from './chart'
import { paperTrade, placeOrder } from '../paper'

/** Quick paper-trade controls on the Analyze page: market buy/sell at the live
 * price, or a pending limit/stop order the auto-executor fills when hit.
 * Prefills stop/target from the advice plan. */
export function PaperTradeCard({
  symbol, live, advice, lastClose, theme: t,
}: {
  symbol: string
  live: LiveQuote | null
  advice: Advice | null
  lastClose: number | null
  theme: Theme
}) {
  const plan = advice?.plan
  const [qty, setQty] = useState('')
  const [limit, setLimit] = useState('')
  const stop = plan?.stop != null ? String(plan.stop) : ''
  const target = plan?.target1 != null ? String(plan.target1) : ''
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)

  const refPrice = live?.ltp ?? lastClose ?? null
  const ysym = /\.(NS|BO)$/.test(symbol) ? symbol : `${symbol}.NS`
  const bare = symbol.replace(/\.(NS|BO)$/, '')

  const market = async (side: 'BUY' | 'SELL') => {
    const q = Number(qty)
    if (!q || q <= 0) { setErr('Enter a quantity'); return }
    setBusy(true); setErr(null); setMsg(null)
    try {
      const r = await paperTrade({
        symbol: bare, yahoo_symbol: ysym, side, qty: q, price: null,
        stop: side === 'BUY' && stop ? Number(stop) : null,
        target: side === 'BUY' && target ? Number(target) : null,
      })
      setMsg(`${side} ${q} ${bare} @ ₹${fmt.price(r.price)} (paper) · charges ₹${fmt.price(r.charges)}${r.realized_pnl != null ? ` · realized ${r.realized_pnl >= 0 ? '+' : ''}₹${fmt.int(r.realized_pnl)}` : ''}`)
      setQty('')
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const pending = async (side: 'BUY' | 'SELL') => {
    const q = Number(qty); const lp = Number(limit)
    if (!q || q <= 0) { setErr('Enter a quantity'); return }
    if (!lp || lp <= 0) { setErr('Enter a limit price'); return }
    setBusy(true); setErr(null); setMsg(null)
    try {
      const r = await placeOrder({
        symbol: bare, yahoo_symbol: ysym, side, qty: q, limit_price: lp,
        stop: side === 'BUY' && stop ? Number(stop) : null,
        target: side === 'BUY' && target ? Number(target) : null,
      })
      setMsg(`Pending ${r.kind}: ${side} ${q} ${bare} — fills when price ${r.trigger === 'lte' ? '≤' : '≥'} ₹${fmt.price(lp)}. The executor watches it live.`)
      setQty(''); setLimit('')
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="card">
      <div className="card-head">
        <h2>Paper trade</h2>
        <span className="hover-readout">
          {refPrice != null && (
            <>ref ₹{fmt.price(refPrice)}{live && <span className="live-readout"><span className="live-dot" /> live</span>}</>
          )}
        </span>
      </div>
      {msg && <p className="chart-note" style={{ color: t.deltaUpText }}>{msg}</p>}
      {err && <p className="detail-warn">{err}</p>}
      <div className="controls paper-trade">
        <input value={qty} onChange={(e) => setQty(e.target.value)} placeholder="Qty" type="number" min="1" aria-label="Quantity" />
        <button type="button" disabled={busy} onClick={() => market('BUY')}>Buy at market</button>
        <button type="button" className="btn-ghost" disabled={busy} onClick={() => market('SELL')}>Sell at market</button>
        <span className="cell-sub" style={{ alignSelf: 'center' }}>— or a pending order —</span>
        <input value={limit} onChange={(e) => setLimit(e.target.value)} placeholder="at price ₹" type="number" step="0.05" aria-label="Limit price" />
        <button type="button" disabled={busy} onClick={() => pending('BUY')}>Place buy</button>
        <button type="button" className="btn-ghost" disabled={busy} onClick={() => pending('SELL')}>Place sell</button>
      </div>
      <p className="chart-note">
        Market orders fill now at the live price. A pending buy fills when price falls to yours; a pending sell auto-detects
        take-profit (above) vs stop-loss (below) and the executor fills it on its own. Stop ₹{fmt.price(plan?.stop)} / target
        ₹{fmt.price(plan?.target1)} from the plan attach to buys. Paper only.
      </p>
    </div>
  )
}
