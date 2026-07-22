import { useEffect, useRef, useState } from 'react'
import type { Theme } from '../theme'
import { fetchDepth, type DepthBook } from '../live'
import { fmt } from './chart'

/** Prominent live order-flow pressure gauge: buy vs sell quantity queued across
 * the book, as a clear Buy/Sell-heavy verdict with a ratio. Polls ~3s while the
 * market is open. Meant to sit near the outlook. */
export function OrderFlowGauge({ symbol, theme: t }: { symbol: string; theme: Theme }) {
  const [book, setBook] = useState<DepthBook | null>(null)
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
            failures.current = 0
            delay = b.market_open ? 3_000 : 60_000
          }
        } catch {
          failures.current += 1
          delay = failures.current > 2 ? 30_000 : 6_000
        }
      }
      if (!stopped) timer = window.setTimeout(() => void tick(), delay)
    }
    void tick()
    return () => { stopped = true; if (timer != null) window.clearTimeout(timer) }
  }, [symbol])

  if (!book) return null
  const buy = book.total_buy_qty ?? 0
  const sell = book.total_sell_qty ?? 0
  const total = buy + sell
  if (total <= 0) return null

  const buyPct = (buy / total) * 100
  const sellPct = 100 - buyPct
  const heavy = sellPct >= 58 ? 'sell' : buyPct >= 58 ? 'buy' : 'balanced'
  const ratio = heavy === 'sell' ? sell / Math.max(1, buy) : buy / Math.max(1, sell)

  const meta = {
    sell: { label: 'Sellers in control', color: t.deltaDownText, tone: 'bad' as const },
    buy: { label: 'Buyers in control', color: t.deltaUpText, tone: 'good' as const },
    balanced: { label: 'Balanced flow', color: t.textSecondary, tone: 'neutral' as const },
  }[heavy]

  return (
    <div className="card orderflow">
      <div className="card-head">
        <h2>
          Live order flow
          <span className={`verdict verdict-${meta.tone}`} style={{ marginLeft: 10 }}>{meta.label}</span>
        </h2>
        <span className="hover-readout">
          {book.market_open ? <span className="live-readout"><span className="live-dot" /> live</span> : 'market closed'}
          {book.ltp != null && ` · ₹${fmt.price(book.ltp)}`}
        </span>
      </div>

      <div className="of-headline" style={{ color: meta.color }}>
        {heavy === 'balanced'
          ? 'Buy and sell interest roughly matched'
          : `${heavy === 'sell' ? 'Sell' : 'Buy'} orders outweigh the other side ${ratio.toFixed(1)} : 1`}
      </div>

      <div className="of-gauge">
        <div className="of-fill of-buy" style={{ width: `${buyPct}%` }}>
          {buyPct >= 14 && <span>{buyPct.toFixed(0)}%</span>}
        </div>
        <div className="of-fill of-sell" style={{ width: `${sellPct}%` }}>
          {sellPct >= 14 && <span>{sellPct.toFixed(0)}%</span>}
        </div>
        <div className="of-mid" />
      </div>
      <div className="of-legend">
        <span style={{ color: t.deltaUpText }}>▲ Buy {fmt.compact(buy)}</span>
        <span style={{ color: t.deltaDownText }}>Sell {fmt.compact(sell)} ▼</span>
      </div>
      <p className="chart-note">
        Total quantity queued to buy vs. sell across the order book, updating live. A quick read on short-term
        pressure — but it shifts fast and large orders are often placed to be pulled, so treat it as mood, not a
        forecast.
      </p>
    </div>
  )
}
