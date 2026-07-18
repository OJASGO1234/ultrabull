import { useCallback, useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import { fetchAnalysis, type RangeQuery } from './api'
import type { AnalysisResponse } from './types'
import { useTheme } from './theme'
import { PriceChart } from './components/PriceChart'
import { RsiChart } from './components/RsiChart'
import { MacdChart } from './components/MacdChart'
import { EquityChart } from './components/EquityChart'
import { MetricsCards } from './components/MetricsCards'
import { AuditBadge } from './components/AuditBadge'
import { ByPatternTable, DetectionsList, EventsTable, LevelsTable, SignalsTable, TradesTable } from './components/Tables'
import { ContextCard } from './components/ContextCard'
import { OutlookCard } from './components/OutlookCard'
import { AdviceCard } from './components/AdviceCard'
import { ChatSidebar } from './components/ChatSidebar'
import { FundamentalsCard } from './components/FundamentalsCard'

const PRESETS = [
  { value: '0.25', label: '3 months' },
  { value: '0.5', label: '6 months' },
  { value: '1', label: '1 year' },
  { value: '2', label: '2 years' },
  { value: '3', label: '3 years' },
  { value: '5', label: '5 years' },
  { value: 'custom', label: 'Custom range…' },
]

export default function App() {
  const theme = useTheme()
  const [symbol, setSymbol] = useState('HINDCOPPER.NS')
  const [rangeSel, setRangeSel] = useState('0.5')
  const [fromDate, setFromDate] = useState('')
  const [toDate, setToDate] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<AnalysisResponse | null>(null)
  const [active, setActive] = useState<{ symbol: string; range: RangeQuery } | null>(null)

  const run = useCallback(async (s: string, range: RangeQuery) => {
    setLoading(true)
    setError(null)
    try {
      setData(await fetchAnalysis(s, range))
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
      void run(s, { start: fromDate, end: toDate || undefined })
    } else {
      void run(s, { years: Number(rangeSel) })
    }
  }

  return (
    <div className="page">
      <header className="page-head">
        <h1>Ultrabull</h1>
        <p className="subtitle">
          Pattern detection + backtest for one NSE/BSE ticker (Yahoo Finance symbols, e.g.{' '}
          <code>ATUL.NS</code>)
        </p>
      </header>

      <form className="controls" onSubmit={onSubmit}>
        <input
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
          placeholder="Ticker, e.g. HINDCOPPER.NS"
          aria-label="Ticker symbol"
          spellCheck={false}
        />
        <select value={rangeSel} onChange={(e) => setRangeSel(e.target.value)} aria-label="History window">
          {PRESETS.map((o) => (
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

      {error && <div className="error-banner">⚠ {error}</div>}
      {loading && <div className="loading">Downloading data, detecting patterns, running backtest…</div>}

      {data && !loading && (
        <main className="results">
          <section className="run-head">
            <h2>
              {data.symbol}{' '}
              <span className="run-meta">
                · {data.bars} bars · through {data.last_date}
                {data.cached && ' · cached'}
              </span>
            </h2>
            <AuditBadge audit={data.audit} />
          </section>

          {data.advice && <AdviceCard advice={data.advice} outlook={data.outlook} theme={theme} />}

          {data.outlook && <OutlookCard outlook={data.outlook} theme={theme} />}

          {data.context && data.timing && (
            <ContextCard context={data.context} timing={data.timing} theme={theme} />
          )}

          {data.fundamentals && <FundamentalsCard fundamentals={data.fundamentals} theme={theme} />}

          <PriceChart
            key={`${data.symbol}:${data.bars}:${data.last_date}`}
            data={data}
            theme={theme}
          />

          <div className="card">
            <h2>Support / resistance levels</h2>
            <LevelsTable levels={data.levels ?? []} theme={theme} />
          </div>

          <div className="chart-row">
            <RsiChart data={data} theme={theme} />
            <MacdChart data={data} theme={theme} />
          </div>

          <EquityChart data={data} theme={theme} />

          <div className="card">
            <h2>Per-pattern performance</h2>
            <ByPatternTable rows={data.by_pattern} theme={theme} />
          </div>

          <MetricsCards metrics={data.metrics} theme={theme} />

          <div className="card">
            <h2>Trades ({data.trades.length})</h2>
            <TradesTable trades={data.trades} theme={theme} />
          </div>

          <div className="card">
            <h2>Compiled signals ({data.signals.length})</h2>
            <p className="chart-note">
              A compiled signal is a detection that survived every filter (qualification, market/sector
              regime, relative strength, earnings blackout, liquidity) and was converted into a concrete
              order: when to enter, and a stop and target in ₹ built from the stock's own volatility
              (×ATR). These are exactly what the backtest above traded. Hover a pattern name for what
              the setup means.
            </p>
            <SignalsTable signals={data.signals} />
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

          <div className="card">
            <h2>Recent qualified detections</h2>
            <p className="chart-note">
              Detections are raw pattern sightings — the watchlist stage, <i>before</i> the market-regime
              and earnings gates. "Qualified" means the pattern agreed with the stock's own trend (or is
              a legitimate counter-trend reversal). Each entry below explains what the pattern means and
              the ₹ plan the system would attach. A detection that never became a compiled signal was
              blocked by a gate — usually market regime, relative strength, or earnings blackout.
            </p>
            <DetectionsList detections={data.detections} />
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

          <div className="card">
            <h2>Latest session summary</h2>
            <p className="chart-note">
              Narrative digest of the most recent session: what fired, in which regime, with the key
              stats spelled out — meant to be read as prose after the market closes.
            </p>
            <pre className="summary">{data.summary_md.replace(/[#*_]/g, '')}</pre>
          </div>
        </main>
      )}

      {!data && !loading && !error && (
        <div className="empty-state">Enter a ticker above to run the full pipeline.</div>
      )}

      <ChatSidebar
        symbol={active?.symbol ?? ''}
        range={active?.range ?? {}}
        hasData={active != null && data != null}
      />
    </div>
  )
}
