import { useCallback, useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import { fetchAnalysis } from './api'
import type { AnalysisResponse } from './types'
import { useTheme } from './theme'
import { PriceChart } from './components/PriceChart'
import { RsiChart } from './components/RsiChart'
import { MacdChart } from './components/MacdChart'
import { EquityChart } from './components/EquityChart'
import { MetricsCards } from './components/MetricsCards'
import { AuditBadge } from './components/AuditBadge'
import { ByPatternTable, DetectionsList, SignalsTable, TradesTable } from './components/Tables'

const YEAR_OPTIONS = [
  { value: 0.5, label: '6 months' },
  { value: 1, label: '1 year' },
  { value: 2, label: '2 years' },
]

export default function App() {
  const theme = useTheme()
  const [symbol, setSymbol] = useState('HINDCOPPER.NS')
  const [years, setYears] = useState(0.5)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<AnalysisResponse | null>(null)

  const run = useCallback(async (s: string, y: number) => {
    setLoading(true)
    setError(null)
    try {
      setData(await fetchAnalysis(s, y))
      const url = new URL(window.location.href)
      url.searchParams.set('symbol', s)
      url.searchParams.set('years', String(y))
      window.history.replaceState(null, '', url)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }, [])

  // Deep link: /?symbol=ATUL.NS&years=1 auto-runs on load
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const s = params.get('symbol')?.trim().toUpperCase()
    if (!s) return
    const y = Number(params.get('years')) || 0.5
    setSymbol(s)
    setYears(y)
    void run(s, y)
  }, [run])

  const onSubmit = (e: FormEvent) => {
    e.preventDefault()
    const s = symbol.trim().toUpperCase()
    if (!s || loading) return
    void run(s, years)
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
        <select value={years} onChange={(e) => setYears(Number(e.target.value))} aria-label="History window">
          {YEAR_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>
        <button type="submit" disabled={loading || !symbol.trim()}>
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

          <MetricsCards metrics={data.metrics} theme={theme} />

          <PriceChart data={data} theme={theme} />

          <div className="chart-row">
            <RsiChart data={data} theme={theme} />
            <MacdChart data={data} theme={theme} />
          </div>

          <EquityChart data={data} theme={theme} />

          <div className="card">
            <h2>Per-pattern performance</h2>
            <ByPatternTable rows={data.by_pattern} theme={theme} />
          </div>

          <div className="card">
            <h2>Trades ({data.trades.length})</h2>
            <TradesTable trades={data.trades} theme={theme} />
          </div>

          <div className="card">
            <h2>Compiled signals ({data.signals.length})</h2>
            <SignalsTable signals={data.signals} />
          </div>

          <div className="card">
            <h2>Recent qualified detections</h2>
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
            <pre className="summary">{data.summary_md.replace(/[#*_]/g, '')}</pre>
          </div>
        </main>
      )}

      {!data && !loading && !error && (
        <div className="empty-state">Enter a ticker above to run the full pipeline.</div>
      )}
    </div>
  )
}
