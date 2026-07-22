import { useEffect, useState } from 'react'
import type { Theme } from '../theme'
import { authFetch } from '../auth'
import { fmt } from './chart'

interface Deal {
  type: 'bulk' | 'block' | 'short'
  symbol: string | null
  name: string | null
  client: string | null
  side: string | null
  qty: number | null
  price: number | null
  date: string | null
  remarks: string | null
}

interface DealsResp {
  symbol: string
  as_on_date: string | null
  ok: boolean
  bulk: Deal[]
  block: Deal[]
  short: Deal[]
}

/** Bulk / block / short-sell deals for the analyzed stock, from NSE (EOD
 * disclosure — same-day-after-close at best, never intraday). */
export function DealsCard({ symbol, theme: t }: { symbol: string; theme: Theme }) {
  const [data, setData] = useState<DealsResp | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    authFetch(`/api/deals?symbol=${encodeURIComponent(symbol)}`)
      .then((r) => r.json())
      .then((d) => { if (!cancelled) setData(d) })
      .catch((e) => { if (!cancelled) setError(e instanceof Error ? e.message : String(e)) })
    return () => { cancelled = true }
  }, [symbol])

  const bulkBlock = data ? [...data.bulk, ...data.block] : []
  const shorts = data?.short ?? []
  const nothing = data && bulkBlock.length === 0 && shorts.length === 0

  return (
    <div className="card">
      <div className="card-head">
        <h2>Large deals — bulk / block / short</h2>
        <span className="hover-readout">{data?.as_on_date ? `NSE, as on ${data.as_on_date}` : ''}</span>
      </div>
      <p className="chart-note">
        SEBI end-of-day disclosures from NSE — <b>not live</b>. Bulk (&gt;0.5% of shares) and block deals name the
        counterparties; short data is aggregate. Today's deals publish after market close, so this is same-day-after-close
        at best. A large institutional buy/sell is context, not a signal.
      </p>
      {error && <p className="empty">Deals unavailable (NSE may be blocking the server): {error}</p>}
      {!data && !error && <p className="empty">Loading deals…</p>}
      {nothing && <p className="empty">No bulk, block, or short-sell disclosures for {data!.symbol} on {data!.as_on_date ?? 'the latest day'}.</p>}
      {bulkBlock.length > 0 && (
        <div className="table-wrap">
          <table>
            <thead><tr><th>Type</th><th>Side</th><th>Client</th><th className="num">Qty</th><th className="num">Price ₹</th><th>Date</th></tr></thead>
            <tbody>
              {bulkBlock.map((d, i) => (
                <tr key={i}>
                  <td>{d.type}</td>
                  <td style={{ color: d.side === 'BUY' ? t.deltaUpText : t.deltaDownText }}>{d.side}</td>
                  <td>{d.client}</td>
                  <td className="num">{d.qty != null ? fmt.int(d.qty) : '—'}</td>
                  <td className="num">{fmt.price(d.price)}</td>
                  <td className="cell-sub">{d.date}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {shorts.length > 0 && (
        <p className="chart-note">
          Short-sell reported: {shorts.map((d) => (d.qty != null ? `${fmt.int(d.qty)} sh` : '')).join(', ')} on {data!.as_on_date}.
        </p>
      )}
    </div>
  )
}
