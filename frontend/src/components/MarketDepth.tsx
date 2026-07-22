import { useEffect, useRef, useState } from 'react'
import type { Theme } from '../theme'
import { fetchDepth, type DepthBook } from '../live'
import { fmt } from './chart'

/** Level-2 order book from Groww: 5 bids / 5 asks with depth bars, spread, and
 * a buy/sell pressure gauge. Polls ~every 3s while the market is open. */
export function MarketDepth({ symbol, theme: t }: { symbol: string; theme: Theme }) {
  const [book, setBook] = useState<DepthBook | null>(null)
  const [error, setError] = useState<string | null>(null)
  const failures = useRef(0)

  useEffect(() => {
    let stopped = false
    let timer: number | undefined
    const tick = async () => {
      if (stopped) return
      let delay = 3_000
      if (!document.hidden) {
        try {
          const b = await fetchDepth(symbol)
          if (!stopped) {
            setBook(b)
            setError(null)
            failures.current = 0
            delay = b.market_open ? 3_000 : 60_000
          }
        } catch (e) {
          failures.current += 1
          if (!stopped && !book) setError(e instanceof Error ? e.message : String(e))
          delay = failures.current > 2 ? 30_000 : 5_000
        }
      }
      if (!stopped) timer = window.setTimeout(() => void tick(), delay)
    }
    void tick()
    return () => {
      stopped = true
      if (timer != null) window.clearTimeout(timer)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol])

  if (error && !book) {
    return (
      <div className="card">
        <h2>Market depth</h2>
        <p className="empty">Depth unavailable: {error}</p>
      </div>
    )
  }
  if (!book) {
    return (
      <div className="card">
        <h2>Market depth</h2>
        <p className="empty">Loading order book…</p>
      </div>
    )
  }

  const levels = [...book.bids, ...book.asks].map((l) => l.qty ?? 0)
  const maxQty = Math.max(1, ...levels)

  const Row = ({ price, qty, orders, side }: { price: number | null; qty: number | null; orders: number | null; side: 'bid' | 'ask' }) => {
    const w = qty != null ? Math.max(2, (qty / maxQty) * 100) : 0
    const color = side === 'bid' ? t.good : t.critical
    return (
      <div className="depth-row">
        <div className="depth-bar" style={{ width: `${w}%`, background: `color-mix(in srgb, ${color} 22%, transparent)`, [side === 'bid' ? 'right' : 'left']: 0 } as React.CSSProperties} />
        <span className="depth-price" style={{ color }}>{fmt.price(price)}</span>
        <span className="depth-qty">{qty != null ? fmt.int(qty) : '—'}</span>
        <span className="depth-orders">{orders ?? ''}</span>
      </div>
    )
  }

  return (
    <div className="card">
      <div className="card-head">
        <h2>Market depth</h2>
        <span className="hover-readout">
          {book.market_open ? <span className="live-readout"><span className="live-dot" /> live</span> : 'market closed'}
          {book.ltp != null && ` · LTP ₹${fmt.price(book.ltp)}`}
          {book.spread != null && ` · spread ₹${fmt.price(book.spread)} (${book.spread_pct}%)`}
        </span>
      </div>
      <p className="chart-note">
        The live order book — the best 5 buy (bid) and sell (ask) prices with quantities waiting. A thin spread and
        deep book mean easy fills; a wide spread or thin levels mean your order can move the price.
      </p>
      <div className="depth-cols">
        <div className="depth-col">
          <div className="depth-head"><span>Bid ₹</span><span>Qty</span><span>Orders</span></div>
          {book.bids.map((l, i) => <Row key={i} {...l} side="bid" />)}
        </div>
        <div className="depth-col">
          <div className="depth-head"><span>Ask ₹</span><span>Qty</span><span>Orders</span></div>
          {book.asks.map((l, i) => <Row key={i} {...l} side="ask" />)}
        </div>
      </div>
    </div>
  )
}
