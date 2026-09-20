import { useCallback, useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import { fetchAnalysis, RANGE_PRESETS, type RangeQuery } from './api'
import type { AnalysisResponse, ViewName } from './types'
import { useTheme } from './theme'
import { fmt } from './components/chart'
import { PriceChart } from './components/PriceChart'
import { MarketDepth } from './components/MarketDepth'
import { OrderFlowGauge } from './components/OrderFlowGauge'
import { DealsCard } from './components/DealsCard'
import { RsiChart } from './components/RsiChart'
import { MacdChart } from './components/MacdChart'
import { EquityChart } from './components/EquityChart'
import { MetricsCards } from './components/MetricsCards'
import { AuditBadge } from './components/AuditBadge'
import { ByPatternTable, DetectionsList, EventsTable, LevelsTable, SignalsTable, TradesTable } from './components/Tables'
import { ContextCard } from './components/ContextCard'
import { OutlookCard } from './components/OutlookCard'
import { AdviceCard } from './components/AdviceCard'
import { PaperTradeCard } from './components/PaperTradeCard'
import { ChatSidebar } from './components/ChatSidebar'
import { FundamentalsCard } from './components/FundamentalsCard'
import { KeyIndicators } from './components/KeyIndicators'
import { MomentumCard } from './components/MomentumCard'
import { MarketContextToggle } from './components/MarketContextToggle'
import { HoldingsPage } from './components/HoldingsPage'
import { HealthPage } from './components/HealthPage'
import { PaperPage } from './components/PaperPage'
import { Login } from './components/Login'
import { clearToken, fetchMe, UNAUTHORIZED_EVENT } from './auth'
import { useLiveQuotes } from './live'

type Page = 'analyze' | 'holdings' | 'health' | 'paper'

export default function App() {
  const theme = useTheme()
  const [page, setPage] = useState<Page>(() => {
    const p = new URLSearchParams(window.location.search).get('page')
    return p === 'holdings' || p === 'health' || p === 'paper' ? p : 'analyze'
  })
  const [wide, setWide] = useState(() => localStorage.getItem('ui.wide') === '1')
  const [authState, setAuthState] = useState<'checking' | 'login' | 'ok'>('checking')
  const [authRequired, setAuthRequired] = useState(false)

  useEffect(() => {
    let cancelled = false
    fetchMe()
      .then((me) => {
        if (cancelled) return
        setAuthRequired(me.auth_required)
        setAuthState(me.auth_required && !me.username ? 'login' : 'ok')
      })
      .catch(() => {
        // API unreachable — let the app render; requests will surface errors
        if (!cancelled) setAuthState('ok')
      })
    const onUnauthorized = () => setAuthState('login')
    window.addEventListener(UNAUTHORIZED_EVENT, onUnauthorized)
    return () => {
      cancelled = true
      window.removeEventListener(UNAUTHORIZED_EVENT, onUnauthorized)
    }
  }, [])
  const [symbol, setSymbol] = useState('HINDCOPPER.NS')
  const [rangeSel, setRangeSel] = useState('0.5')
  const [interval, setInterval] = useState('1d')
  const [fromDate, setFromDate] = useState('')
  const [toDate, setToDate] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<AnalysisResponse | null>(null)
  // Which analysis view is on screen — shared by the Analyze and Holdings
  // tabs so switching "market context" applies everywhere at once.
  // Deliberately NOT persisted to localStorage: the stock-only view ignores
  // validated filters, so it should be a conscious choice each session
  // rather than a setting that quietly sticks.
  const [view, setView] = useState<ViewName>('market_aware')
  const [active, setActive] = useState<{ symbol: string; range: RangeQuery } | null>(null)
  const liveState = useLiveQuotes(active && page === 'analyze' ? [active.symbol] : [])
  const liveQuote = active ? liveState.quotes[active.symbol] ?? null : null

  // Both views ship in one payload, so switching is instant and needs no
  // refetch. Fall back to the top-level fields for intraday runs (no views).
  const activeView = data?.views?.[view] ?? null
  const advice = activeView?.advice ?? data?.advice ?? null
  const outlook = activeView?.outlook ?? data?.outlook ?? null
  const stockOnly = view === 'stock_only' && activeView != null
  const keyIndicators = activeView?.key_indicators ?? data?.key_indicators ?? null
  // The backtest belongs to the view too — a stock-only verdict sitting above a
  // market-aware equity curve would be telling two stories at once. Charts take
  // the whole payload, so overlay the view's series onto it.
  const viewData = data && activeView
    ? {
        ...data,
        metrics: activeView.metrics ?? data.metrics,
        by_pattern: activeView.by_pattern ?? data.by_pattern,
        signals: activeView.signals ?? data.signals,
        trades: activeView.trades ?? data.trades,
        equity: activeView.equity ?? data.equity,
        detections: activeView.detections ?? data.detections,
      }
    : data

  const run = useCallback(async (s: string, range: RangeQuery, iv = '1d') => {
    setLoading(true)
    setError(null)
    try {
      setData(await fetchAnalysis(s, range, iv))
      setActive({ symbol: s, range })
      const url = new URL(window.location.href)
      url.searchParams.set('symbol', s)
      if (range.start) {
        url.searchParams.set('start', range.start)
        if (range.end) url.searchParams.set('end', range.end)
        else url.searchParams.delete('end')
        url.searchParams.delete('years')
      } else {
        url.searchParams.set('years', String(range.years ?? 0.5))
        url.searchParams.delete('start')
        url.searchParams.delete('end')
      }
      window.history.replaceState(null, '', url)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }, [])

  // Deep link: /?symbol=ATUL.NS&years=1 or &start=2024-01-01&end=2025-01-01
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const s = params.get('symbol')?.trim().toUpperCase()
    if (!s) return
    setSymbol(s)
    const start = params.get('start')
    if (start) {
      const end = params.get('end') ?? ''
      setRangeSel('custom')
      setFromDate(start)
      setToDate(end)
      void run(s, { start, end: end || undefined })
    } else {
      const y = Number(params.get('years')) || 0.5
      setRangeSel(String(y))
      void run(s, { years: y })
    }
  }, [run])

  const onSubmit = (e: FormEvent) => {
    e.preventDefault()
    const s = symbol.trim().toUpperCase()
    if (!s || loading) return
    if (rangeSel === 'custom') {
      if (!fromDate) return
      void run(s, { start: fromDate, end: toDate || undefined }, interval)
    } else {
      void run(s, { years: Number(rangeSel) }, interval)
    }
  }

  const switchPage = useCallback((p: Page) => {
    setPage(p)
    const url = new URL(window.location.href)
    if (p !== 'analyze') url.searchParams.set('page', p)
    else url.searchParams.delete('page')
    window.history.replaceState(null, '', url)
  }, [])

  const openAnalysis = useCallback(
    (s: string) => {
      switchPage('analyze')
      setSymbol(s)
      setRangeSel('1')
      setInterval('1d')
      void run(s, { years: 1 })
    },
    [run, switchPage],
  )

  const toggleWide = () => {
    setWide((w) => {
      localStorage.setItem('ui.wide', w ? '0' : '1')
      return !w
    })
  }

  if (authState === 'checking') {
    return <div className="page"><div className="loading">Loading…</div></div>
  }
  if (authState === 'login') {
    return <Login onSuccess={() => setAuthState('ok')} />
  }

  return (
    <div className={wide ? 'page page-wide' : 'page'}>
      <header className="page-head">
        <h1>Ultrabull</h1>
        <p className="subtitle">
          Pattern detection + backtest for one NSE/BSE ticker (Yahoo Finance symbols, e.g.{' '}
          <code>ATUL.NS</code>)
        </p>
        <nav className="tabs" aria-label="Pages">
          <button
            type="button"
            className={page === 'analyze' ? 'tab tab-on' : 'tab'}
            onClick={() => switchPage('analyze')}
          >
            Analyze
          </button>
          <button
            type="button"
            className={page === 'holdings' ? 'tab tab-on' : 'tab'}
            onClick={() => switchPage('holdings')}
          >
            Holdings
          </button>
          <button
            type="button"
            className={page === 'paper' ? 'tab tab-on' : 'tab'}
            onClick={() => switchPage('paper')}
          >
            Paper
          </button>
          <button
            type="button"
            className={page === 'health' ? 'tab tab-on' : 'tab'}
            onClick={() => switchPage('health')}
          >
            Health
          </button>
          <button
            type="button"
            className="tab tab-end"
            onClick={toggleWide}
            title={wide ? 'Constrain to a centered column' : 'Use the full window width'}
          >
            {wide ? '⇥⇤ Normal width' : '⇤⇥ Full width'}
          </button>
          {authRequired && (
            <button
              type="button"
              className="tab"
              onClick={() => {
                clearToken()
                setAuthState('login')
              }}
            >
              Sign out
            </button>
          )}
        </nav>
      </header>

      {page === 'holdings' && (
        <HoldingsPage theme={theme} onOpenAnalysis={openAnalysis} view={view} onViewChange={setView} />
      )}

      {page === 'health' && <HealthPage theme={theme} />}

      {page === 'paper' && <PaperPage theme={theme} />}

      {page === 'analyze' && (<>
      <form className="controls" onSubmit={onSubmit}>
        <input
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
          placeholder="Ticker, e.g. HINDCOPPER.NS"
          aria-label="Ticker symbol"
          spellCheck={false}
        />
        <select value={interval} onChange={(e) => setInterval(e.target.value)} aria-label="Granularity">
          <option value="1d">Daily</option>
          <option value="60m">1 hour</option>
          <option value="30m">30 min</option>
          <option value="15m">15 min</option>
          <option value="5m">5 min</option>
        </select>
        <select value={rangeSel} onChange={(e) => setRangeSel(e.target.value)} aria-label="History window">
          {RANGE_PRESETS.map((o) => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>
        {rangeSel === 'custom' && (
          <>
            <input
              type="date"
              value={fromDate}
              onChange={(e) => setFromDate(e.target.value)}
              aria-label="From date"
              required
            />
            <input
              type="date"
              value={toDate}
              onChange={(e) => setToDate(e.target.value)}
              aria-label="To date (blank = today)"
              min={fromDate || undefined}
            />
          </>
        )}
        <button type="submit" disabled={loading || !symbol.trim() || (rangeSel === 'custom' && !fromDate)}>
          {loading ? 'Analyzing…' : 'Analyze'}
        </button>
      </form>

      {/* Chosen BEFORE the run, alongside ticker and window — it is an analysis
          setting, not a result filter. Both views are computed server-side in
          one pass, so changing it after a run re-renders instantly instead of
          refetching; the choice made here is what the results land on. */}
      <MarketContextToggle view={view} onChange={setView} />
      {interval !== '1d' && (
        <p className="chart-note">
          Market context does not apply to intraday runs — the gates and base rates it switches were fit
          on daily bars, so intraday results are unchanged either way.
        </p>
      )}

      {error && <div className="error-banner">⚠ {error}</div>}
      {loading && <div className="loading">Downloading data, detecting patterns, running backtest…</div>}

      {data && !loading && (
        <main className="results">
          <section className="run-head">
            <h2>
              {data.symbol}{' '}
              {liveQuote && (
                <span
                  className="live-readout live-big"
                  style={{ color: (liveQuote.day_change_pct ?? 0) >= 0 ? theme.deltaUpText : theme.deltaDownText }}
                >
                  <span className="live-dot" /> ₹
                  {liveQuote.ltp.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  {liveQuote.day_change_pct != null &&
                    ` ${liveQuote.day_change_pct >= 0 ? '+' : ''}${liveQuote.day_change_pct.toFixed(2)}%`}
                  {liveState.mode === 'ws' && <span className="cell-sub"> realtime</span>}
                </span>
              )}{' '}
              <span className="run-meta">
                · {data.bars} bars · through {data.last_date}
                {data.cached && ' · cached'}
              </span>
            </h2>
            <AuditBadge audit={data.audit} />
          </section>

          {data.intraday && (
            <div className="card intraday-banner">
              <b>Intraday view ({data.interval}).</b> Showing price structure, indicators, support/resistance and
              pattern sightings on {data.interval} bars. The base-rate outlook, calibrated probabilities, the backtest
              and the validated-edge verdicts are <b>hidden here</b> — they were fit on daily bars and don't transfer to
              intraday. Live order flow, market depth, deals and paper trading below are timeframe-independent and still
              apply. Switch to <b>Daily</b> for the full analysis.
              {data.cap_note && <><br /><b>Note:</b> {data.cap_note}</>}
            </div>
          )}


          {stockOnly && activeView && (
            <div className="card stock-only-banner">
              <b>Stock-only view — market context is switched off.</b>{' '}
              {activeView.warning}
              {activeView.lifted_score_factors && (
                <>
                  <br />
                  <b>Dropped from the advice score:</b>{' '}
                  {activeView.lifted_score_factors.join(', ')}.{' '}
                  <b>Lifted as backtest gates:</b> {activeView.lifted_gates?.join(', ')}.{' '}
                  <b>Still applied:</b> {activeView.kept_filters?.join(', ')}.
                  {activeView.already_off?.map((s) => (
                    <span key={s}> ({s}.)</span>
                  ))}
                </>
              )}
              {activeView.metrics && data.views?.market_aware.metrics && (
                <>
                  <br />
                  <b>On this symbol's own backtest:</b> profit factor{' '}
                  {fmt.ratio(data.views.market_aware.metrics.profit_factor)} (market-aware) →{' '}
                  {fmt.ratio(activeView.metrics.profit_factor)} (stock-only) over{' '}
                  {data.views.market_aware.trades.length} → {activeView.trades.length} trades.
                  The backtest, equity curve and signal list below follow this view.
                </>
              )}
            </div>
          )}

          {advice && <AdviceCard advice={advice} outlook={outlook} theme={theme} />}

          {keyIndicators && <KeyIndicators ki={keyIndicators} theme={theme} />}

          {data.momentum && <MomentumCard momentum={data.momentum} theme={theme} />}

          <PaperTradeCard
            symbol={data.symbol}
            live={liveQuote}
            advice={advice}
            lastClose={data.ohlcv.length ? data.ohlcv[data.ohlcv.length - 1].close : null}
            theme={theme}
          />

          {outlook && <OutlookCard outlook={outlook} theme={theme} />}

          <OrderFlowGauge symbol={data.symbol} theme={theme} />

          {data.context && data.timing && (
            <ContextCard context={data.context} timing={data.timing} theme={theme} stockOnly={stockOnly} />
          )}

          {data.fundamentals && <FundamentalsCard fundamentals={data.fundamentals} theme={theme} />}

          <PriceChart
            key={`${data.symbol}:${data.bars}:${data.last_date}`}
            data={viewData ?? data}
            theme={theme}
            live={liveQuote}
          />

          <MarketDepth symbol={data.symbol} theme={theme} />

          <DealsCard symbol={data.symbol} theme={theme} />

          <div className="card">
            <h2>Support / resistance levels</h2>
            <LevelsTable levels={data.levels ?? []} theme={theme} />
          </div>

          <div className="chart-row">
            <RsiChart data={data} theme={theme} />
            <MacdChart data={data} theme={theme} />
          </div>

          {!data.intraday && (
            <>
              <EquityChart data={viewData ?? data} theme={theme} />

              <div className="card">
                <h2>Per-pattern performance</h2>
                <ByPatternTable rows={(viewData ?? data).by_pattern} theme={theme} />
              </div>

              <MetricsCards metrics={(viewData ?? data).metrics} theme={theme} />

              <div className="card">
                <h2>Trades ({(viewData ?? data).trades.length})</h2>
                <TradesTable trades={(viewData ?? data).trades} theme={theme} />
              </div>

              <div className="card">
                <h2>Compiled signals ({(viewData ?? data).signals.length})</h2>
                <p className="chart-note">
                  A compiled signal is a detection that survived every filter (qualification,{' '}
                  {stockOnly ? '' : 'market/sector regime, '}relative strength, earnings blackout,
                  liquidity) and was converted into a concrete order: when to enter, and a stop and
                  target in ₹ built from the stock's own volatility (×ATR). These are exactly what the
                  backtest above traded. Hover a pattern name for what the setup means.
                  {stockOnly && ' Market-timing filters are lifted in this view, so more detections survive.'}
                </p>
                <SignalsTable signals={(viewData ?? data).signals} />
              </div>

              <div className="card">
                <h2>Corporate events in window ({data.events?.length ?? 0})</h2>
                <p className="chart-note">
                  Why these matter for timing: <b>results</b> gap prices unpredictably — the engine refuses
                  fresh entries 3 days before and 1 day after each announcement; <b>ex-dividend dates</b>{' '}
                  mechanically drop the price by roughly the payout (prices here are dividend-adjusted, so
                  signals are unaffected — but a short held through ex-date owes the dividend);{' '}
                  <b>splits/bonuses</b> are cosmetic but often bring a burst of volume and retail attention.
                </p>
                <EventsTable events={data.events ?? []} />
              </div>
            </>
          )}

          <div className="card">
            <h2>Recent qualified detections</h2>
            <p className="chart-note">
              Detections are raw pattern sightings — the watchlist stage, <i>before</i> the market-regime
              and earnings gates. "Qualified" means the pattern agreed with the stock's own trend (or is
              a legitimate counter-trend reversal). Each entry below explains what the pattern means and
              the ₹ plan the system would attach. A detection that never became a compiled signal was
              blocked by a gate — usually market regime, relative strength, or earnings blackout.
            </p>
            <DetectionsList detections={(viewData ?? data).detections} />
            {data.signal_summary?.total_hits != null && (
              <p className="chart-note">
                {data.signal_summary.total_hits} detections total ·{' '}
                {data.signal_summary.qualified_hits} qualified ·{' '}
                {data.signal_summary.per_week_qualified?.toFixed(1)} qualified/week
                {data.signal_summary.by_family &&
                  ' · ' +
                    Object.entries(data.signal_summary.by_family)
                      .map(([k, v]) => `${k}: ${v}`)
                      .join(', ')}
              </p>
            )}
          </div>

          {!data.intraday && (
            <div className="card">
              <h2>Latest session summary</h2>
              <p className="chart-note">
                Narrative digest of the most recent session: what fired, in which regime, with the key
                stats spelled out — meant to be read as prose after the market closes.
              </p>
              <pre className="summary">{data.summary_md.replace(/[#*_]/g, '')}</pre>
            </div>
          )}
        </main>
      )}

      {!data && !loading && !error && (
        <div className="empty-state">Enter a ticker above to run the full pipeline.</div>
      )}

      <ChatSidebar
        symbol={active?.symbol ?? ''}
        range={active?.range ?? {}}
        hasData={active != null && data != null}
        view={view}
      />
      </>)}
    </div>
  )
}
