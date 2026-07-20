import { Fragment, useCallback, useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Theme } from '../theme'
import { fmt } from './chart'
import {
  clearCreds, fetchClaudeTake, fetchGrowwHoldings, fetchServerCreds, fetchSignal, fetchSreHoldings,
  fetchSreNews, fetchStoredSignals, fetchStoredTakes, loadCreds, saveCreds, saveServerCreds, tokenExp,
  type ClaudeTake, type CompactSignal, type HoldingsResponse, type SreHolding, type TakePair,
} from '../sre'
import { fetchHoldDates, fetchXirr, saveHoldDate, type XirrResult } from '../sre'
import { fetchAnalysis, RANGE_PRESETS, type RangeQuery } from '../api'
import type { AnalysisResponse } from '../types'
import { PriceChart } from './PriceChart'
import { KeyIndicators } from './KeyIndicators'
import { useLiveQuotes } from '../live'

type SignalState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'ok'; signal: CompactSignal }
  | { status: 'error'; message: string }

type NewsState =
  | { status: 'loading' }
  | { status: 'ok'; items: Record<string, unknown>[] }
  | { status: 'error'; message: string }

type ChartState =
  | { status: 'loading' }
  | { status: 'ok'; data: AnalysisResponse }
  | { status: 'error'; message: string }

/** History window a snapshot was computed over, as a fetchable range. */
function snapshotRange(s: CompactSignal | null): RangeQuery {
  if (s?.range?.start) return { start: s.range.start, end: s.range.end ?? undefined }
  return { years: s?.range?.years ?? 1 }
}

const VERDICT_META: Record<string, { label: string; tone: 'good' | 'warn' | 'bad' | 'neutral' | 'info' }> = {
  BUY: { label: 'Buy', tone: 'good' },
  BUY_ON_DIP: { label: 'Buy on dip', tone: 'info' },
  WAIT: { label: 'Wait', tone: 'neutral' },
  AVOID: { label: 'Avoid', tone: 'bad' },
  HOLD: { label: 'Hold', tone: 'good' },
  TIGHTEN_STOP: { label: 'Tighten stop', tone: 'warn' },
  CONSIDER_TRIM: { label: 'Consider trim', tone: 'warn' },
  EXIT_OR_REDUCE: { label: 'Exit / reduce', tone: 'bad' },
}

function VerdictBadge({ verdict }: { verdict: string | null | undefined }) {
  if (!verdict) return <span className="empty">—</span>
  const meta = VERDICT_META[verdict] ?? { label: verdict, tone: 'neutral' as const }
  return <span className={`verdict verdict-${meta.tone}`}>{meta.label}</span>
}

const rowKey = (h: SreHolding, i: number) =>
  `${h.broker ?? 'SRE'}:${h.yahoo_symbol ?? h.symbol ?? h.isin ?? String(i)}`

/** The exit levels translated into THIS position's rupees, from qty × avg cost. */
function PositionBlock({ h, s, theme: t }: { h: SreHolding; s: CompactSignal; theme: Theme }) {
  const qty = h.quantity!
  const avg = h.avg_price!
  const color = (v: number) => ({ color: v >= 0 ? t.deltaUpText : t.deltaDownText })
  const rows: { label: string; level: number }[] = []
  if (s.close != null) rows.push({ label: 'Last close', level: s.close })
  if (s.stop != null) rows.push({ label: 'Exit stop', level: s.stop })
  if (s.target != null) rows.push({ label: 'Exit target', level: s.target })
  if (s.target2 != null) rows.push({ label: 'Second target', level: s.target2 })
  return (
    <>
      <p>
        <b>Your position</b> — {fmt.int(qty)} shares · break-even ₹{fmt.price(avg)}
      </p>
      <table className="position-table">
        <thead>
          <tr>
            <th>Sold at…</th>
            <th className="num">Price ₹</th>
            <th className="num">Your P&L ₹</th>
            <th className="num">vs cost</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const rupees = (r.level - avg) * qty
            const pct = (r.level / avg - 1) * 100
            return (
              <tr key={r.label}>
                <td>{r.label}</td>
                <td className="num">{fmt.price(r.level)}</td>
                <td className="num" style={color(rupees)}>
                  {rupees >= 0 ? '+' : '−'}{fmt.int(Math.abs(rupees))}
                </td>
                <td className="num" style={color(pct)}>
                  {pct >= 0 ? '+' : '−'}{Math.abs(pct).toFixed(1)}%
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <p className="chart-note">
        The stop is placed by the chart's support level, not by your purchase price — this table only converts
        each exit level into your position's rupees.
      </p>
    </>
  )
}

/** On-demand Claude judgment for one holding: quick (payload-only) and deep (web research). */
function ClaudeTakeSection({ h, pair, busy, error, onAsk }: {
  h: SreHolding
  pair: TakePair | undefined
  busy: 'quick' | 'deep' | undefined
  error: string | undefined
  onAsk: (deep: boolean) => void
}) {
  const takes = [pair?.deep, pair?.quick].filter((t): t is ClaudeTake => !!t)
  return (
    <div className="claude-take">
      <div className="claude-take-head">
        <b>Claude's take</b>
        <button type="button" className="btn-mini" disabled={!!busy} onClick={() => onAsk(false)}>
          {busy === 'quick' ? 'Thinking…' : pair?.quick ? 'Refresh quick take' : 'Ask Claude'}
        </button>
        <button
          type="button"
          className="btn-mini"
          disabled={!!busy}
          title="Adds live web research (result dates, analyst estimates, news) — can take a minute"
          onClick={() => onAsk(true)}
        >
          {busy === 'deep' ? 'Researching…' : pair?.deep ? 'Refresh deep research' : 'Deep research'}
        </button>
        {busy === 'deep' && <span className="cell-sub">searching the web — this can take a minute</span>}
      </div>
      {error && <p className="detail-warn">Claude take failed: {error}</p>}
      {takes.length === 0 && !busy && !error && (
        <p className="chart-note">
          No stored take yet. Quick take reads the analysis + your position ({h.quantity != null ? fmt.int(h.quantity) : '?'} @
          ₹{fmt.price(h.avg_price)}); deep research also checks the web for result dates, estimates and news.
        </p>
      )}
      {takes.map((t) => (
        <div key={t.tier} className="take-body">
          <p className="cell-sub">
            {t.tier === 'deep' ? 'Deep research' : 'Quick take'} · data through {t.as_of ?? '?'} · asked{' '}
            {t.stored_at.slice(0, 16).replace('T', ' ')}
          </p>
          <div className="chat-md">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{t.text}</ReactMarkdown>
          </div>
        </div>
      ))}
    </div>
  )
}

function rangeLabel(r: CompactSignal['range']): string | null {
  if (!r) return null
  if (r.start) return `${r.start} → ${r.end ?? 'today'}`
  if (r.years != null) return RANGE_PRESETS.find((p) => p.value === String(r.years))?.label ?? `${r.years}y`
  return null
}

// best-effort readers for SRE's undocumented news records
function newsField(item: Record<string, unknown>, names: string[]): string | null {
  const byKey = new Map(Object.keys(item).map((k) => [k.toLowerCase().replace(/[^a-z0-9]/g, ''), k]))
  for (const n of names) {
    const k = byKey.get(n)
    if (k != null && item[k] != null && String(item[k]).trim() !== '') return String(item[k])
  }
  return null
}

export function HoldingsPage({ theme: t, onOpenAnalysis }: { theme: Theme; onOpenAnalysis: (symbol: string) => void }) {
  const saved = loadCreds()
  const [clientId, setClientId] = useState(saved.clientId)
  const [token, setToken] = useState(saved.token)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<HoldingsResponse | null>(null)
  const [growwRows, setGrowwRows] = useState<SreHolding[]>([])
  const [signals, setSignals] = useState<Record<string, SignalState>>({})
  const [news, setNews] = useState<Record<string, NewsState>>({})
  const [charts, setCharts] = useState<Record<string, ChartState>>({})
  const [takes, setTakes] = useState<Record<string, TakePair>>({})
  const [takeBusy, setTakeBusy] = useState<Record<string, 'quick' | 'deep' | undefined>>({})
  const [takeError, setTakeError] = useState<Record<string, string | undefined>>({})
  const [expanded, setExpanded] = useState<string | null>(null)
  const [rangeSel, setRangeSel] = useState('1')
  const [fromDate, setFromDate] = useState('')
  const [toDate, setToDate] = useState('')
  const [tokenDeadline, setTokenDeadline] = useState<number | null>(null)
  const [now, setNow] = useState(() => Date.now())
  const generation = useRef(0)

  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 30_000)
    return () => window.clearInterval(id)
  }, [])

  const setSignal = useCallback((gen: number, sym: string, st: SignalState) => {
    if (generation.current === gen) setSignals((prev) => ({ ...prev, [sym]: st }))
  }, [])

  // fill the table from the local stores — instant, no computation
  const loadStored = useCallback(async (symbols: string[]) => {
    const gen = ++generation.current
    setSignals(Object.fromEntries(symbols.map((s) => [s, { status: 'idle' } as SignalState])))
    setTakes({})
    try {
      const stored = await fetchStoredSignals(symbols)
      if (generation.current !== gen) return
      setSignals((prev) => {
        const next = { ...prev }
        for (const [sym, signal] of Object.entries(stored)) next[sym] = { status: 'ok', signal }
        return next
      })
    } catch {
      // store unreachable — rows stay idle with their Analyse buttons
    }
    try {
      const storedTakes = await fetchStoredTakes(symbols)
      if (generation.current === gen) setTakes(storedTakes)
    } catch {
      // no stored takes — the drawer just shows the Ask Claude buttons
    }
  }, [])

  const askClaude = useCallback(async (h: SreHolding, deep: boolean) => {
    const sym = h.yahoo_symbol
    if (!sym) return
    setTakeBusy((prev) => ({ ...prev, [sym]: deep ? 'deep' : 'quick' }))
    setTakeError((prev) => ({ ...prev, [sym]: undefined }))
    try {
      const take = await fetchClaudeTake(sym, { quantity: h.quantity, avg_price: h.avg_price, deep })
      setTakes((prev) => ({ ...prev, [sym]: { ...prev[sym], [take.tier]: take } }))
    } catch (err) {
      setTakeError((prev) => ({ ...prev, [sym]: err instanceof Error ? err.message : String(err) }))
    } finally {
      setTakeBusy((prev) => ({ ...prev, [sym]: undefined }))
    }
  }, [])

  // history window for Analyse runs — same presets as the analysis tab
  const analyseRange = useCallback((): RangeQuery => {
    if (rangeSel === 'custom') return { start: fromDate, end: toDate || undefined }
    return { years: Number(rangeSel) }
  }, [rangeSel, fromDate, toDate])
  const rangeInvalid = rangeSel === 'custom' && !fromDate

  // full analysis payload for the drawer's candles-and-layers chart; the TTL
  // cache makes this near-instant right after an Analyse run
  const loadChart = useCallback(async (sym: string, range: RangeQuery) => {
    setCharts((prev) => ({ ...prev, [sym]: { status: 'loading' } }))
    try {
      const chartData = await fetchAnalysis(sym, range)
      setCharts((prev) => ({ ...prev, [sym]: { status: 'ok', data: chartData } }))
    } catch (err) {
      setCharts((prev) => ({
        ...prev,
        [sym]: { status: 'error', message: err instanceof Error ? err.message : String(err) },
      }))
    }
  }, [])

  // run the full pipeline for one symbol and persist the snapshot server-side
  const analyseOne = useCallback(async (sym: string) => {
    const gen = generation.current
    const range = analyseRange()
    setSignal(gen, sym, { status: 'loading' })
    try {
      const signal = await fetchSignal(sym, range, true)
      setSignal(gen, sym, { status: 'ok', signal })
      // drop any loaded chart so the drawer refetches the fresh series
      setCharts((prev) => {
        if (!(sym in prev)) return prev
        const next = { ...prev }
        delete next[sym]
        return next
      })
    } catch (err) {
      setSignal(gen, sym, { status: 'error', message: err instanceof Error ? err.message : String(err) })
    }
  }, [setSignal, analyseRange])

  const analyseAll = useCallback((symbols: string[]) => {
    const gen = generation.current
    const range = analyseRange()
    for (const s of symbols) setSignal(gen, s, { status: 'loading' })
    const queue = [...symbols]
    const worker = async () => {
      for (let sym = queue.shift(); sym != null; sym = queue.shift()) {
        const s = sym
        try {
          const signal = await fetchSignal(s, range, true)
          setSignal(gen, s, { status: 'ok', signal })
        } catch (err) {
          setSignal(gen, s, { status: 'error', message: err instanceof Error ? err.message : String(err) })
        }
      }
    }
    for (let i = 0; i < 2; i++) void worker()
  }, [setSignal, analyseRange])

  const connect = useCallback(async (creds: { clientId: string; token: string }) => {
    setLoading(true)
    setError(null)
    try {
      const resp = await fetchSreHoldings(creds)
      saveCreds(creds)
      saveServerCreds(creds) // persist server-side so fields come preset next login
      setData(resp)
      setNews({})
      setCharts({})
      setExpanded(null)
      setTokenDeadline(resp.token?.seconds_left != null ? Date.now() + resp.token.seconds_left * 1000 : null)
    } catch (err) {
      const e = err as Error & { status?: number }
      setError(e.message)
      if (e.status === 401) clearCreds()
    } finally {
      setLoading(false)
    }
  }, [])

  // auto-load when the tab already holds working credentials; otherwise pull
  // the server-stored creds to preset the fields (and auto-connect while the
  // stored token is still alive — SRE tokens die ~30 min after issue)
  useEffect(() => {
    if (saved.clientId && saved.token) {
      if (!data && !loading && !error) void connect(saved)
      return
    }
    let cancelled = false
    void fetchServerCreds().then((server) => {
      if (cancelled || !server) return
      setClientId(server.clientId)
      setToken(server.token)
      const exp = tokenExp(server.token)
      if (exp != null && exp * 1000 > Date.now() + 30_000) void connect(server)
    })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // lazily fetch the chart for whichever row is expanded (and refetch after a
  // re-analysis invalidates it) — window matches the row's stored snapshot
  useEffect(() => {
    if (!expanded || !data) return
    const idx = data.holdings.findIndex((x, i) => rowKey(x, i) === expanded)
    if (idx < 0) return
    const sym = data.holdings[idx].yahoo_symbol
    if (!sym || charts[sym]) return
    const st = signals[sym]
    void loadChart(sym, snapshotRange(st?.status === 'ok' ? st.signal : null))
  }, [expanded, data, charts, signals, loadChart])

  const onSubmit = (e: FormEvent) => {
    e.preventDefault()
    if (!clientId.trim() || !token.trim() || loading) return
    void connect({ clientId: clientId.trim(), token: token.trim() })
  }

  const loadNews = useCallback(async (key: string, symbol: string) => {
    setNews((prev) => ({ ...prev, [key]: { status: 'loading' } }))
    try {
      const resp = await fetchSreNews({ clientId: clientId.trim(), token: token.trim() }, symbol)
      setNews((prev) => ({ ...prev, [key]: { status: 'ok', items: resp.items } }))
    } catch (err) {
      setNews((prev) => ({
        ...prev,
        [key]: { status: 'error', message: err instanceof Error ? err.message : String(err) },
      }))
    }
  }, [clientId, token])

  // Groww rows need no pasted creds — the server authenticates via TOTP keys
  useEffect(() => {
    let cancelled = false
    fetchGrowwHoldings()
      .then((r) => {
        if (!cancelled) setGrowwRows(r.holdings)
      })
      .catch(() => undefined)
    return () => {
      cancelled = true
    }
  }, [])

  const holdings: SreHolding[] = [
    ...(data?.holdings ?? []).map((h) => ({ ...h, broker: 'SRE' as const })),
    ...growwRows,
  ]
  const allSymbols = [...new Set(holdings.map((h) => h.yahoo_symbol).filter((s): s is string => !!s))]
  const symbolsKey = allSymbols.sort().join(',')
  const anyRunning = allSymbols.some((s) => signals[s]?.status === 'loading')
  const liveState = useLiveQuotes(allSymbols)

  // (re)load stored signals + takes whenever the combined symbol set changes
  useEffect(() => {
    if (symbolsKey) void loadStored(symbolsKey.split(','))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbolsKey, loadStored])

  const sig = (h: SreHolding): CompactSignal | null => {
    const st = h.yahoo_symbol ? signals[h.yahoo_symbol] : undefined
    return st?.status === 'ok' ? st.signal : null
  }
  // Price preference: live Groww tick > Yahoo EOD close (from the stored
  // analysis) > SRE's holdings LTP (proved days-stale outside market hours).
  // Qty/avg cost stay broker-side.
  const liveFor = (h: SreHolding) => (h.yahoo_symbol ? liveState.quotes[h.yahoo_symbol] : undefined)
  const effPrice = (h: SreHolding) => liveFor(h)?.ltp ?? sig(h)?.close ?? h.ltp ?? null

  const totals = holdings.reduce(
    (acc, h) => {
      const price = effPrice(h)
      const invested = h.invested ?? (h.quantity != null && h.avg_price != null ? h.quantity * h.avg_price : null)
      const current = h.quantity != null && price != null ? h.quantity * price : h.current_value
      if (invested != null) acc.invested += invested
      if (current != null) acc.current += current
      return acc
    },
    { invested: 0, current: 0 },
  )
  const totalPnl = totals.current - totals.invested

  // buy-dates + portfolio XIRR (positions priced with the same live>eod logic)
  const [holdDates, setHoldDates] = useState<Record<string, string>>({})
  const [xirr, setXirr] = useState<XirrResult | null>(null)
  useEffect(() => {
    fetchHoldDates().then((r) => setHoldDates(r.dates)).catch(() => undefined)
  }, [])
  const positionsKey = holdings
    .map((h, i) => {
      const price = effPrice(h)
      const invested = h.invested ?? (h.quantity != null && h.avg_price != null ? h.quantity * h.avg_price : null)
      const cur = h.quantity != null && price != null ? h.quantity * price : h.current_value
      return `${rowKey(h, i)}|${invested ?? ''}|${cur ?? ''}|${holdDates[rowKey(h, i)] ?? ''}`
    })
    .join(';')
  useEffect(() => {
    if (!holdings.length) return
    const positions = holdings
      .map((h, i) => {
        const price = effPrice(h)
        const invested = h.invested ?? (h.quantity != null && h.avg_price != null ? h.quantity * h.avg_price : null)
        const cur = h.quantity != null && price != null ? h.quantity * price : h.current_value
        return invested != null && cur != null
          ? { key: rowKey(h, i), invested, current_value: cur, buy_date: holdDates[rowKey(h, i)] ?? null }
          : null
      })
      .filter((p): p is NonNullable<typeof p> => p != null)
    if (positions.length) fetchXirr(positions).then(setXirr).catch(() => undefined)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [positionsKey])

  const setBuyDate = useCallback((key: string, date: string) => {
    setHoldDates((prev) => ({ ...prev, [key]: date }))
    void saveHoldDate(key, date).catch(() => undefined)
  }, [])

  const secondsLeft = tokenDeadline != null ? Math.max(0, Math.round((tokenDeadline - now) / 1000)) : null
  const pnlColor = (v: number | null) =>
    v == null ? undefined : { color: v >= 0 ? t.deltaUpText : t.deltaDownText }

  return (
    <main className="results">
      <div className="card">
        <h2>SRE account</h2>
        <p className="chart-note">
          Paste your SRE client ID and the bearer token from the trading terminal (DevTools → any request →
          Authorization header, with or without the “Bearer ” prefix). Both stay in this browser tab only and are
          sent to our own backend per request — SRE tokens expire ~30 minutes after login, so expect to re-paste.
        </p>
        <form className="controls sre-connect" onSubmit={onSubmit}>
          <input
            value={clientId}
            onChange={(e) => setClientId(e.target.value)}
            placeholder="Client ID, e.g. A790195"
            aria-label="SRE client ID"
            spellCheck={false}
          />
          <input
            value={token}
            onChange={(e) => setToken(e.target.value)}
            placeholder="Bearer token (eyJ…)"
            aria-label="SRE bearer token"
            type="password"
            spellCheck={false}
            className="sre-token"
          />
          <button type="submit" disabled={loading || !clientId.trim() || !token.trim()}>
            {loading ? 'Loading…' : data ? 'Refresh holdings' : 'Load holdings'}
          </button>
          {data && (
            <button
              type="button"
              className="btn-ghost"
              onClick={() => {
                clearCreds()
                setToken('')
                setData(null)
                setSignals({})
                setCharts({})
                setTokenDeadline(null)
              }}
            >
              Disconnect
            </button>
          )}
        </form>
        {data?.token && (
          <p className="chart-note">
            Token for <b>{data.token.client ?? clientId}</b>
            {secondsLeft != null &&
              (secondsLeft > 0
                ? ` · expires in ~${Math.ceil(secondsLeft / 60)} min`
                : ' · has EXPIRED — paste a fresh one before refreshing')}
          </p>
        )}
      </div>

      {error && <div className="error-banner">⚠ {error}</div>}

      {data && (data.holdings?.length ?? 0) === 0 && growwRows.length === 0 && (
        <div className="card">
          <p className="empty">
            SRE returned no holdings rows. If the account does hold stock, the response shape may be one we don't
            recognize yet — check the raw payload in the network tab (<code>/api/sre/holdings</code>) and we can
            extend the field mapping.
          </p>
        </div>
      )}

      {holdings.length > 0 && (
        <div className="card">
          <div className="card-head">
            <h2>
              Holdings ({holdings.length})
              {liveState.live && (
                <span className="live-readout" style={{ marginLeft: 10 }}>
                  <span className="live-dot" /> {liveState.marketOpen ? 'live' : 'market closed'}
                </span>
              )}
            </h2>
            <span className="hover-readout">
              invested ₹{fmt.compact(totals.invested)} · now ₹{fmt.compact(totals.current)} ·{' '}
              <span style={pnlColor(totalPnl)}>
                {totalPnl >= 0 ? '+' : ''}₹{fmt.compact(totalPnl)}
                {totals.invested > 0 && ` (${((100 * totalPnl) / totals.invested).toFixed(1)}%)`}
              </span>
              {xirr?.portfolio_xirr_pct != null && (
                <span title="Money-weighted annualized return (XIRR) over the holdings with a buy date set. Set dates in each row's drawer.">
                  {' · '}XIRR{' '}
                  <span style={pnlColor(xirr.portfolio_xirr_pct)}>
                    {xirr.portfolio_xirr_pct >= 0 ? '+' : ''}{xirr.portfolio_xirr_pct.toFixed(1)}%
                  </span>
                </span>
              )}
            </span>
            <span className="analyse-controls">
              <select value={rangeSel} onChange={(e) => setRangeSel(e.target.value)} aria-label="Analysis window">
                {RANGE_PRESETS.map((o) => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
              {rangeSel === 'custom' && (
                <>
                  <input type="date" value={fromDate} onChange={(e) => setFromDate(e.target.value)} aria-label="From date" />
                  <input type="date" value={toDate} onChange={(e) => setToDate(e.target.value)} aria-label="To date (blank = today)" min={fromDate || undefined} />
                </>
              )}
              <button
                type="button"
                className="btn-mini"
                disabled={anyRunning || allSymbols.length === 0 || rangeInvalid}
                onClick={() => analyseAll(allSymbols)}
              >
                {anyRunning ? 'Analysing…' : 'Analyse all'}
              </button>
            </span>
          </div>
          <p className="chart-note">
            Qty and avg cost come from the broker; <b>Price ₹ and P&L use Yahoo's latest close</b> (SRE's holdings
            quotes proved stale outside market hours — rows marked "broker" have no analysis yet and fall back to
            SRE's number). Signal columns show the <b>last stored analysis</b> (note the as-of date) — press
            Analyse to compute a fresh one. <b>Our call</b> is the existing-holding verdict — what to do with stock
            you already own; <b>fresh money</b> is whether we'd buy it today. The exit stop sits 0.5×ATR below the
            nearest mapped support (a close below it breaks the setup); the target is the next resistance. Click a
            row for the reasoning, entry zone and news.
          </p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Stock</th>
                  <th>Broker</th>
                  <th className="num">Qty</th>
                  <th className="num">Avg ₹</th>
                  <th className="num">Price ₹</th>
                  <th className="num">P&L ₹</th>
                  <th className="num">P&L %</th>
                  <th>Our call</th>
                  <th>Fresh money</th>
                  <th className="num">Stop ₹</th>
                  <th className="num">Exit target ₹</th>
                  <th className="num">Support</th>
                  <th className="num">Resistance</th>
                  <th>Breakout</th>
                </tr>
              </thead>
              <tbody>
                {holdings.map((h, i) => {
                  const key = rowKey(h, i)
                  const st = h.yahoo_symbol ? signals[h.yahoo_symbol] : undefined
                  const s = st?.status === 'ok' ? st.signal : null
                  const lq = liveFor(h)
                  const usingYahoo = s?.close != null
                  const price = effPrice(h)
                  const invested = h.invested ?? (h.quantity != null && h.avg_price != null ? h.quantity * h.avg_price : null)
                  // P&L from the best price we have (live > yahoo > broker)
                  const pnl = h.quantity != null && invested != null && price != null
                    ? h.quantity * price - invested
                    : h.pnl ?? null
                  const pnlPct = pnl != null && invested ? (100 * pnl) / invested : h.pnl_pct
                  const isOpen = expanded === key
                  const newsState = news[key]
                  const breakout = s?.breakout.above_all_resistance
                    ? 'Blue sky (above every mapped resistance)'
                    : s?.breakout.recent_breakout
                      ? `${s.breakout.recent_breakout.pattern} on ${s.breakout.recent_breakout.date}`
                      : null
                  // broker LTP vs the analysis' EOD close — brokers often serve
                  // stale holdings quotes outside market hours
                  const gapPct = h.ltp != null && s?.close != null ? (100 * (h.ltp - s.close)) / s.close : null
                  const priceConflict = gapPct != null && Math.abs(gapPct) > 2
                  const stopBreached = s?.stop != null && price != null && price <= s.stop
                  return (
                    <Fragment key={key}>
                      <tr className="holding-row" onClick={() => setExpanded(isOpen ? null : key)}>
                        <td>
                          <b>{h.symbol ?? h.isin ?? '?'}</b>
                          <span className="cell-sub">
                            {' '}
                            {[h.exchange, h.symbol ? h.isin : null].filter(Boolean).join(' · ')}
                          </span>
                        </td>
                        <td>{h.broker ?? 'SRE'}</td>
                        <td className="num">{h.quantity == null ? '—' : fmt.int(h.quantity)}</td>
                        <td className="num">{fmt.price(h.avg_price)}</td>
                        <td className="num">
                          {fmt.price(price)}
                          {lq?.day_change_pct != null && (
                            <span
                              className="cell-sub"
                              style={{ color: lq.day_change_pct >= 0 ? t.deltaUpText : t.deltaDownText }}
                            >
                              {' '}{lq.day_change_pct >= 0 ? '+' : ''}{lq.day_change_pct.toFixed(1)}%
                            </span>
                          )}
                          {!lq && !usingYahoo && h.ltp != null && <span className="cell-sub"> broker</span>}
                          {!lq && usingYahoo && priceConflict && (
                            <span
                              title={`Broker (SRE) reports ₹${fmt.price(h.ltp)} — ${Math.abs(gapPct!).toFixed(1)}% away, likely a stale quote; showing Yahoo's ${s?.last_date} close instead`}
                            >
                              {' '}⚠
                            </span>
                          )}
                        </td>
                        <td className="num" style={pnlColor(pnl)}>
                          {pnl == null ? '—' : `${pnl >= 0 ? '+' : ''}${fmt.int(pnl)}`}
                        </td>
                        <td className="num" style={pnlColor(pnlPct)}>
                          {pnlPct == null ? '—' : `${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(1)}%`}
                        </td>
                        {!h.yahoo_symbol ? (
                          <td colSpan={7} className="signal-pending">no NSE/BSE symbol in the SRE row</td>
                        ) : st?.status === 'loading' ? (
                          <td colSpan={7} className="signal-pending">analysing (full pipeline)…</td>
                        ) : st?.status === 'error' ? (
                          <td colSpan={7} className="signal-pending">
                            failed: {st.message}{' '}
                            <button
                              type="button"
                              className="btn-mini"
                              onClick={(e) => {
                                e.stopPropagation()
                                void analyseOne(h.yahoo_symbol!)
                              }}
                            >
                              Retry
                            </button>
                          </td>
                        ) : s ? (
                          <>
                            <td>
                              <VerdictBadge verdict={s.holding?.verdict} />
                              <span className="cell-sub"> {s.last_date}</span>
                            </td>
                            <td><VerdictBadge verdict={s.fresh?.verdict} /></td>
                            <td className="num">{fmt.price(s.stop)}</td>
                            <td className="num">{fmt.price(s.target)}</td>
                            <td className="num">
                              {s.support ? (
                                <>
                                  {fmt.price(s.support.price)}
                                  <span className="cell-sub"> {s.support.distance_pct != null ? `${s.support.distance_pct.toFixed(1)}%` : ''}</span>
                                </>
                              ) : '—'}
                            </td>
                            <td className="num">
                              {s.resistance ? (
                                <>
                                  {fmt.price(s.resistance.price)}
                                  <span className="cell-sub"> {s.resistance.distance_pct != null ? `+${s.resistance.distance_pct.toFixed(1)}%` : ''}</span>
                                </>
                              ) : '—'}
                            </td>
                            <td>{breakout ?? <span className="empty">—</span>}</td>
                          </>
                        ) : (
                          <td colSpan={7} className="signal-pending">
                            <button
                              type="button"
                              className="btn-mini"
                              onClick={(e) => {
                                e.stopPropagation()
                                void analyseOne(h.yahoo_symbol!)
                              }}
                            >
                              Analyse
                            </button>
                            <span className="cell-sub"> no stored analysis yet</span>
                          </td>
                        )}
                      </tr>
                      {isOpen && (
                        <tr className="holding-detail">
                          <td colSpan={14}>
                            {s && priceConflict && (
                              <p className="detail-warn">
                                ⚠ Price sources disagree: the broker (SRE) reports ₹{fmt.price(h.ltp)} while Yahoo's
                                close for {s.last_date} is ₹{fmt.price(s.close)} ({gapPct! >= 0 ? '+' : ''}
                                {gapPct!.toFixed(1)}% apart) — broker holdings endpoints often serve stale quotes
                                outside market hours. Price, P&L and all plan levels here use the Yahoo series; the
                                broker figure is shown only for reference.
                              </p>
                            )}
                            {s && stopBreached && (
                              <p className="detail-warn">
                                ⚠ The current price ₹{fmt.price(price)} is at/below the exit stop ₹{fmt.price(s.stop)}{' '}
                                — the exit condition has already triggered; verify the live quote and decide.
                              </p>
                            )}
                            {s ? (
                              <div className="detail-grid">
                                <div>
                                  <p>
                                    <b>Our call — {VERDICT_META[s.holding?.verdict ?? '']?.label ?? s.holding?.verdict}:</b>{' '}
                                    {s.holding?.reason}
                                  </p>
                                  <p>
                                    <b>Fresh money — {VERDICT_META[s.fresh?.verdict ?? '']?.label ?? s.fresh?.verdict}:</b>{' '}
                                    {s.fresh?.reason}
                                  </p>
                                  {s.in_earnings_blackout && (
                                    <p className="detail-warn">⚠ Results imminent — gap risk; the engine blocks fresh entries here.</p>
                                  )}
                                  <ul className="factor-list">
                                    {s.factors.map((f, j) => (
                                      <li key={j}>
                                        <span className="factor-mark" style={{ color: f.score > 0 ? t.deltaUpText : t.deltaDownText }}>
                                          {f.score > 0 ? '+' : '−'}
                                        </span>
                                        <b>{f.name}</b>: {f.detail}
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                                <div>
                                  <p>
                                    <b>Plan</b> — data through {s.last_date}, close ₹{fmt.price(s.close)} · analysed{' '}
                                    {s.stored_at.slice(0, 16).replace('T', ' ')}
                                    {rangeLabel(s.range) && ` · ${rangeLabel(s.range)} window`}
                                  </p>
                                  <ul className="plan-list">
                                    <li>Exit stop: ₹{fmt.price(s.stop)} — {s.stop_basis}</li>
                                    <li>
                                      Exit target: ₹{fmt.price(s.target)}
                                      {s.target2 != null && ` (then ₹${fmt.price(s.target2)})`}
                                      {s.reward_risk != null && ` · reward:risk ${s.reward_risk.toFixed(1)}:1`}
                                    </li>
                                    <li>
                                      Add-on / re-entry zone: ₹{fmt.price(s.entry_zone[0])}–₹{fmt.price(s.entry_zone[1])}
                                    </li>
                                    {s.if_breakout?.note && (
                                      <li><b>If it breaks:</b> {s.if_breakout.note}</li>
                                    )}
                                    {s.outlook_1w.p_up != null && (
                                      <li>
                                        1-week base rate: {Math.round(s.outlook_1w.p_up * 100)} of 100 similar days closed
                                        higher a week later (n={s.outlook_1w.n ?? '?'}) — a lean, not a forecast
                                      </li>
                                    )}
                                    {s.next_earnings && <li>Next results (announced): {s.next_earnings}</li>}
                                    {!s.next_earnings && s.expected_results_window && (
                                      <li>
                                        Results expected ~{s.expected_results_window.start} to {s.expected_results_window.end}{' '}
                                        <span className="cell-sub">({s.expected_results_window.basis})</span>
                                      </li>
                                    )}
                                    {s.results_overdue && <li className="detail-warn">Results look overdue vs the usual cadence — announcement (and gap risk) could come any day</li>}
                                    {s.market_regime && <li>Market regime: NIFTY {s.market_regime}</li>}
                                    {s.breakout.recent_breakout?.english && <li>{s.breakout.recent_breakout.english}</li>}
                                  </ul>
                                  {h.quantity != null && h.avg_price != null && <PositionBlock h={h} s={s} theme={t} />}
                                  <p className="buydate-row">
                                    <label>
                                      Bought on{' '}
                                      <input
                                        type="date"
                                        value={holdDates[key] ?? ''}
                                        max={new Date().toISOString().slice(0, 10)}
                                        onChange={(e) => e.target.value && setBuyDate(key, e.target.value)}
                                      />
                                    </label>
                                    {xirr?.positions[key]?.annualized_pct != null && (
                                      <span style={pnlColor(xirr.positions[key]!.annualized_pct)}>
                                        {' '}· annualized {xirr.positions[key]!.annualized_pct! >= 0 ? '+' : ''}
                                        {xirr.positions[key]!.annualized_pct!.toFixed(1)}%
                                        {xirr.positions[key]?.days_held != null && (
                                          <span className="cell-sub"> ({xirr.positions[key]!.days_held}d held)</span>
                                        )}
                                      </span>
                                    )}
                                    {!holdDates[key] && (
                                      <span className="cell-sub"> — set a buy date to get annualized return &amp; portfolio XIRR</span>
                                    )}
                                  </p>
                                  <div className="detail-actions">
                                    {h.yahoo_symbol && (
                                      <>
                                        <button type="button" onClick={() => onOpenAnalysis(h.yahoo_symbol!)}>
                                          Full analysis →
                                        </button>
                                        <button type="button" className="btn-ghost" onClick={() => void analyseOne(h.yahoo_symbol!)}>
                                          Re-analyse
                                        </button>
                                      </>
                                    )}
                                    {h.symbol && (
                                      <button
                                        type="button"
                                        className="btn-ghost"
                                        disabled={newsState?.status === 'loading'}
                                        onClick={() => void loadNews(key, h.symbol!)}
                                      >
                                        {newsState?.status === 'loading' ? 'Fetching news…' : 'Company news (SRE)'}
                                      </button>
                                    )}
                                  </div>
                                </div>
                              </div>
                            ) : (
                              <p className="empty">
                                {h.yahoo_symbol
                                  ? 'No stored analysis yet — press Analyse on this row.'
                                  : 'Could not map this SRE row to an NSE/BSE Yahoo symbol.'}
                              </p>
                            )}
                            {newsState?.status === 'error' && <p className="detail-warn">News failed: {newsState.message}</p>}
                            {newsState?.status === 'ok' && (
                              <div className="news-list">
                                {newsState.items.length === 0 && <p className="empty">No news returned for this scrip.</p>}
                                {newsState.items.slice(0, 8).map((item, j) => {
                                  const head = newsField(item, ['headline', 'heading', 'title', 'newsheadline', 'subject', 'caption', 'news'])
                                  const date = newsField(item, ['newsdate', 'date', 'dt', 'time', 'newsdatetime'])
                                  return (
                                    <p key={j} className="news-item">
                                      {date && <span className="cell-sub">{date} · </span>}
                                      {head ?? JSON.stringify(item)}
                                    </p>
                                  )
                                })}
                              </div>
                            )}
                            {h.yahoo_symbol && (
                              <ClaudeTakeSection
                                h={h}
                                pair={takes[h.yahoo_symbol]}
                                busy={takeBusy[h.yahoo_symbol]}
                                error={takeError[h.yahoo_symbol]}
                                onAsk={(deep) => void askClaude(h, deep)}
                              />
                            )}
                            {s?.key_indicators && <KeyIndicators ki={s.key_indicators} theme={t} compact />}
                            {h.yahoo_symbol && charts[h.yahoo_symbol] && (
                              <div className="drawer-chart">
                                {(() => {
                                  const ch = charts[h.yahoo_symbol!]
                                  if (ch.status === 'loading')
                                    return <p className="signal-pending">loading price chart…</p>
                                  if (ch.status === 'error')
                                    return <p className="detail-warn">Chart failed: {ch.message}</p>
                                  return (
                                    <PriceChart
                                      key={`${ch.data.symbol}:${ch.data.bars}:${ch.data.last_date}`}
                                      data={ch.data}
                                      theme={t}
                                      live={lq ?? null}
                                    />
                                  )
                                })()}
                              </div>
                            )}
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {!data && !loading && !error && (
        <div className="empty-state">Connect your SRE account above to see holdings with our signals attached.</div>
      )}
    </main>
  )
}
