import { useCallback, useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { AreaSeries, type IChartApi, type ISeriesApi, type Time } from 'lightweight-charts'
import { useTheme, type Theme } from '../theme'
import '../paper.css'
import { fmt } from './chart'
import { makeChart } from './chart'
import {
  cancelOrder, fetchPaper, fetchPaperHistory, listOrders, paperTrade, resetPaper, setPaperConfig,
  type PaperHistory, type PaperState, type PendingOrder,
} from '../paper'
import { fetchMonitorStatus, runMonitorCycle, type MonitorCycleResult, type MonitorStatus } from '../momentumMonitor'
import { fetchIntradayStatus, runIntradayOnce, startIntradayMonitor, stopIntradayMonitor, type IntradayStatus } from '../intradayMonitor'
import { Button, EmptyState, MetricCard, MeterBar, Panel, Skeleton, StatusPill, type Tone } from './ui'

const signed = (n: number, text: string) => `${n >= 0 ? '+' : ''}${text}`
const toneOf = (v: number | null | undefined): Tone => (v == null ? 'neutral' : v >= 0 ? 'up' : 'down')

/** Continuous-monitoring prototype (api/momentum_monitor.py): a full
 * screen -> recommend -> paper-execute -> monitor -> close cycle runs
 * automatically once a day (api/main.py starts the scheduler on boot; see
 * that module's docstring for the schedule/timezone and idempotency/restart
 * story) -- "Run monitoring cycle" below just triggers the SAME cycle
 * on demand for development/demo/testing. See core/MOMENTUM_STRATEGY.md's
 * "Continuous Monitoring Prototype" section for what's real vs demo. Open
 * positions it creates show up in the "Open positions" table below like
 * any other paper trade -- this card only adds the recommendation/outcome
 * layer on top. */
function MonitorCard({ theme: t }: { theme: Theme }) {
  const [status, setStatus] = useState<MonitorStatus | null>(null)
  const [lastResult, setLastResult] = useState<MonitorCycleResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      setStatus(await fetchMonitorStatus())
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => { void load() }, [load])

  const onRun = async () => {
    setBusy(true)
    setError(null)
    try {
      setLastResult(await runMonitorCycle({ lookback: '6M', top_n: 5 }))
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const pnlColor = (v: number | null | undefined) =>
    v == null ? undefined : { color: v >= 0 ? t.deltaUpText : t.deltaDownText }
  const latest = status?.latest_recommendations[0] ?? null

  return (
    <Panel
      className="span-6"
      title="UltraBull monitor"
      actions={<Button onClick={onRun} loading={busy}>{busy ? 'Running…' : 'Run monitoring cycle'}</Button>}
    >
      <p className="chart-note">
        Screens the momentum universe, records an immutable recommendation, paper-executes it, then watches open
        positions for target/stop/expiry.
      </p>
      {status && (
        <div className="pill-row">
          {/* The scheduler is a once-a-day job, not a live feed, so its pill never pulses. */}
          {status.scheduler && (
            <StatusPill
              tone={status.scheduler.enabled ? 'good' : 'warn'}
              label={status.scheduler.enabled ? 'Automatic monitoring active' : 'Automatic monitoring disabled'}
            />
          )}
          <StatusPill tone={status.market_open ? 'good' : 'neutral'} label={`Market ${status.market_open ? 'open' : 'closed'}`} />
        </div>
      )}
      {status?.scheduler && (status.scheduler.enabled || status.scheduler.last_error) && (
        <p className="chart-note">
          {status.scheduler.enabled && (
            <>Daily at {status.scheduler.schedule_time} IST
              {status.scheduler.next_run_at && <>, next run {status.scheduler.next_run_at.slice(0, 16).replace('T', ' ')} IST</>}
            </>
          )}
          {status.scheduler.last_error && (
            <span style={{ color: t.deltaDownText }}> — last scheduled run failed: {status.scheduler.last_error}</span>
          )}
        </p>
      )}
      {!status && !error && <Skeleton lines={3} />}
      {error && <p className="notice notice-bad">{error}</p>}
      {lastResult && <p className="notice notice-good">{lastResult.summary}</p>}
      {!lastResult && status?.last_run && (
        <p className="chart-note">
          Last run: {status.last_run.run_at.slice(0, 16).replace('T', ' ')} — {status.last_run.summary}
          {' '}({status.last_run.ok ? 'SUCCESS' : 'ERROR'})
        </p>
      )}

      {status && (
        <div className="tiles">
          <MetricCard label="Recommendations today" value={status.recommendations_today} />
          <MetricCard label="Open paper positions" value={status.open_positions.length} />
          {latest && (
            <MetricCard
              className="tile-wide tile-text"
              label={`Latest recommendation (${latest.status})`}
              display={latest.symbol}
              note={
                <span className="tile-facts">
                  <span>Entry ₹{fmt.price(latest.suggested_entry)}</span>
                  <span>SL ₹{fmt.price(latest.sl_est)}</span>
                  <span>TP ₹{fmt.price(latest.tp_est)}</span>
                  {latest.actual_entry != null && <span>Filled ₹{fmt.price(latest.actual_entry)}</span>}
                </span>
              }
            />
          )}
        </div>
      )}

      {status && status.open_positions.length > 0 && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Symbol</th><th className="num">Entry ₹</th><th className="num">Current ₹</th><th className="num">Unrealized</th><th className="num">Stop / Target</th></tr>
            </thead>
            <tbody>
              {status.open_positions.map((p) => (
                <tr key={p.recommendation_id}>
                  <td><b>{p.symbol}</b></td>
                  <td className="num">{fmt.price(p.actual_entry)}</td>
                  <td className="num">{fmt.price(p.current_price)}</td>
                  <td className="num" style={pnlColor(p.unrealized_pnl)}>
                    {p.unrealized_pnl == null ? '—' : `${p.unrealized_pnl >= 0 ? '+' : ''}₹${fmt.int(p.unrealized_pnl)}`}
                  </td>
                  <td className="num">{fmt.price(p.stop)} / {fmt.price(p.target)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {status && status.recent_outcomes.length > 0 && (
        <ul className="factor-list">
          {status.recent_outcomes.map((o) => (
            <li key={o.id}>
              <span className="factor-mark" style={pnlColor(o.close_realized_pnl)}>
                {o.close_realized_pnl != null && o.close_realized_pnl >= 0 ? '✓' : '✗'}
              </span>
              <b>{o.symbol}</b> — {o.exit_reason}
              {o.close_realized_pnl != null && (
                <span style={pnlColor(o.close_realized_pnl)}> {o.close_realized_pnl >= 0 ? '+' : ''}₹{fmt.int(o.close_realized_pnl)}</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </Panel>
  )
}

/** Intraday paper-monitoring PROTOTYPE (api/intraday_monitor.py) -- a
 * SEPARATE experiment from the daily UltraBull monitor above. Its signal
 * engine is an explicitly unvalidated test/demo-grade placeholder rule
 * (three consecutive higher-close 1-minute bars), never claimed to be a
 * profitable or production-ready strategy -- see that module's docstring.
 * Trades here land in a dedicated "intraday-monitor"/"intraday-demo" paper
 * account, fully isolated from the daily monitor's account above. */
function IntradayMonitorCard({ theme: t }: { theme: Theme }) {
  const [status, setStatus] = useState<IntradayStatus | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      setStatus(await fetchIntradayStatus())
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => {
    void load()
    const id = window.setInterval(() => void load(), 5000)
    return () => window.clearInterval(id)
  }, [load])

  const toggleMonitoring = async () => {
    setBusy(true)
    setError(null)
    try {
      if (status?.monitoring_active) await stopIntradayMonitor()
      else await startIntradayMonitor()
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const onRunOnce = async () => {
    setBusy(true)
    setError(null)
    try {
      await runIntradayOnce({})
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const pnlColor = (v: number | null | undefined) =>
    v == null ? undefined : { color: v >= 0 ? t.deltaUpText : t.deltaDownText }

  return (
    <Panel
      className="span-6"
      title="Intraday monitor (prototype)"
      actions={<>
        <Button onClick={onRunOnce} disabled={busy}>Run once</Button>
        <Button onClick={toggleMonitoring} disabled={busy}>
          {status?.monitoring_active ? 'Stop monitor' : 'Start monitor'}
        </Button>
      </>}
    >
      <p className="chart-note">
        1-minute paper-trading prototype: samples live prices into 1-minute bars, evaluates a test/demo-grade signal
        rule (NOT a validated strategy), paper-executes, and force-squares-off before {status?.square_off_time ?? '—'} IST.
        Fully separate from the daily UltraBull monitor above — its own account, its own signal engine.
      </p>
      {!status && !error && <Skeleton lines={4} />}
      {error && <p className="notice notice-bad">{error}</p>}
      {status && (
        <>
          {/* Three separate questions, three separate pills: is the MONITOR on, is the MARKET
              open, and is the DATA fresh. None of them implies another. */}
          <div className="pill-row">
            <StatusPill
              tone={status.monitoring_active ? 'good' : 'warn'}
              label={status.monitoring_active ? 'Monitoring active' : 'Monitoring stopped'}
              pulse={status.monitoring_active && status.in_session && !status.data_stale}
              title="Pulses only while monitoring is on, inside the trading session, with fresh data"
            />
            <StatusPill tone={status.market_open ? 'good' : 'neutral'} label={`Market ${status.market_open ? 'open' : 'closed'}`} />
            <StatusPill
              tone={status.data_stale ? 'bad' : status.data_source ? 'good' : 'neutral'}
              label={
                status.data_stale
                  ? (status.data_source === 'none' ? 'Data stale · no prices' : 'Data stale')
                  : status.data_source ? `Data: ${status.data_source.toUpperCase()}` : 'Data n/a'
              }
              title={
                (status.data_stale && status.data_source === 'none'
                  ? 'The last monitoring cycle received no prices from the live feed or the REST poll, so entries are blocked. '
                  : '') + (status.last_data_ts ? `Reflects the last cycle, data as of ${status.last_data_ts}` : 'No data timestamp reported')
              }
            />
            <StatusPill tone={status.mode === 'demo' ? 'warn' : 'info'} label={status.mode === 'demo' ? 'DEMO MODE' : 'PAPER MODE'} />
            {status.kill_switch && <StatusPill tone="bad" label="KILL SWITCH TRIPPED (daily loss limit)" />}
          </div>
          <div className="tiles">
            <MetricCard
              label="Timeframe"
              display={status.timeframe}
              note={status.in_session ? (status.in_entry_window ? 'in entry window' : 'monitor-only (past entry cutoff)') : 'outside session'}
            />
            <MetricCard
              label="Last bar"
              display={status.last_bar ? status.last_bar.slice(11, 19) : '—'}
              note={<>{status.last_bar ? '' : 'no bar finalized · '}data {status.data_stale ? 'STALE' : status.data_source ? status.data_source.toUpperCase() : 'n/a'}</>}
            />
            <MetricCard label="Symbols monitored" value={status.symbols_monitored} />
            <MetricCard label="Signals today" value={status.signals_today} />
            <MetricCard label="Trades today" value={status.trades_today} note={`/ ${status.max_trades_per_day} max`} />
            <MetricCard label="Open positions" value={status.open_positions.length} note={`/ ${status.max_open_positions} max`} />
            <MetricCard
              label="Daily P&L"
              value={status.daily_pnl}
              format={(n) => signed(n, `₹${fmt.int(n)}`)}
              tone={toneOf(status.daily_pnl)}
            />
          </div>

          {status.open_positions.length > 0 && (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Symbol</th><th>Dir</th><th className="num">Entry ₹</th><th className="num">Current ₹</th>
                    <th className="num">SL / Target</th><th className="num">P&amp;L</th><th className="num">Holding</th>
                  </tr>
                </thead>
                <tbody>
                  {status.open_positions.map((p) => (
                    <tr key={p.execution_id}>
                      <td><b>{p.symbol}</b></td>
                      <td>{p.direction}</td>
                      <td className="num">{fmt.price(p.actual_entry)}</td>
                      <td className="num">{fmt.price(p.current_price)}</td>
                      <td className="num">{fmt.price(p.stop_loss)} / {fmt.price(p.target)}</td>
                      <td className="num" style={pnlColor(p.unrealized_pnl)}>
                        {p.unrealized_pnl == null ? '—' : `${p.unrealized_pnl >= 0 ? '+' : ''}₹${fmt.int(p.unrealized_pnl)}`}
                      </td>
                      <td className="num">{p.holding_minutes} min</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {status.recent_signals.length > 0 && (
            <ul className="factor-list">
              {status.recent_signals.map((s) => (
                <li key={s.id} title={s.rejection_reason ?? undefined}>
                  <span className="factor-mark">{s.status === 'executed' ? '●' : s.status === 'closed' ? '✓' : s.status === 'rejected' ? '✕' : '○'}</span>
                  {s.signal_timestamp.slice(11, 16)} <b>{s.symbol}</b> {s.direction} — {s.status.toUpperCase()}
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </Panel>
  )
}

/** Equity curve. The chart is created once (rebuilt only if the theme changes);
 * each poll just calls setData on the existing series, so the chart doesn't
 * flash every 15 seconds when fresh history arrives. */
function EquityCurve({ data, theme: t }: { data: { date: string; equity: number }[]; theme: Theme }) {
  const ref = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<ISeriesApi<'Area'> | null>(null)
  const hasChart = data.length >= 2

  useEffect(() => {
    const el = ref.current
    if (!el || !hasChart) return
    const chart = makeChart(el, t, 240)
    seriesRef.current = chart.addSeries(AreaSeries, {
      lineColor: t.blue, topColor: t.areaWash, bottomColor: 'transparent',
      lineWidth: 2, lastValueVisible: true, priceLineVisible: false,
    })
    chartRef.current = chart
    return () => {
      chart.remove()
      chartRef.current = null
      seriesRef.current = null
    }
  }, [hasChart, t])

  useEffect(() => {
    if (!hasChart || !seriesRef.current) return
    seriesRef.current.setData(data.map((p) => ({ time: p.date as Time, value: p.equity })))
    chartRef.current?.timeScale().fitContent()
  }, [data, hasChart, t])

  if (!hasChart) {
    return <EmptyState title="Equity curve appears once there are a few daily snapshots." hint="Nothing is plotted until real history exists." />
  }
  return <div ref={ref} className="chart chart-reveal" />
}

export function PaperPage({ theme: t }: { theme?: Theme }) {
  const fallbackTheme = useTheme()
  const th = t ?? fallbackTheme
  const [state, setState] = useState<PaperState | null>(null)
  const [history, setHistory] = useState<PaperHistory | null>(null)
  const [orders, setOrders] = useState<PendingOrder[]>([])
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
    v == null ? undefined : { color: v >= 0 ? th.deltaUpText : th.deltaDownText }

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
    try {
      setOrders((await listOrders()).orders)
    } catch {
      // orders optional
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
      <main className="grid-12 paper">
        <Panel>{error ? <div className="error-banner">⚠ {error}</div> : <Skeleton lines={4} />}</Panel>
      </main>
    )
  }

  const s = state
  const exposurePct = s.equity > 0 ? (s.holdings_value / s.equity) * 100 : null
  return (
    <main className="grid-12 paper">
      <MonitorCard theme={th} />
      <IntradayMonitorCard theme={th} />

      <Panel
        title="Paper trading"
        actions={<Button variant="ghost" onClick={onReset}>Reset account</Button>}
      >
        <div className="pill-row">
          {/* Pulses only while the market is actually open. */}
          <StatusPill
            tone={s.market_open ? 'good' : 'neutral'}
            pulse={s.market_open}
            label={s.market_open ? 'Market open · fills live' : 'Market closed · fills resume at open'}
          />
          <StatusPill tone={s.executor.running ? 'good' : 'bad'} label={`Auto-executor ${s.executor.running ? 'running' : 'stopped'}`} />
          <StatusPill
            tone={s.executor.pending_orders > 0 ? 'info' : 'neutral'}
            label={`${s.executor.pending_orders} pending order${s.executor.pending_orders === 1 ? '' : 's'}`}
          />
          {s.stats.kill_breached && (
            <StatusPill tone="bad" label={`Kill-line breached: drawdown ${s.stats.current_drawdown_pct}% ≥ ${s.config.kill_drawdown_pct}%`} />
          )}
        </div>
        <p className="chart-note">
          Forward validation with real live prices and no real money. Equity / ETF / REIT delivery only. Costs below
          are charged on every fill and baked into P&L, so what you see is after-cost.
        </p>
        <div className="tiles paper-metrics">
          <MetricCard
            label="Equity"
            value={s.equity}
            format={(n) => `₹${fmt.compact(n)}`}
            note={`${signed(s.total_pnl, `₹${fmt.compact(s.total_pnl)}`)} (${s.total_return_pct}%)`}
            noteTone={toneOf(s.total_pnl)}
          />
          <MetricCard
            label="Cash"
            value={s.cash}
            format={(n) => `₹${fmt.compact(n)}`}
            note={`holdings ₹${fmt.compact(s.holdings_value)}`}
          />
          <MetricCard
            label="Realized P&L"
            value={s.realized_pnl}
            format={(n) => signed(n, `₹${fmt.compact(n)}`)}
            tone={toneOf(s.realized_pnl)}
            note={<>unrealized <span style={pnlColor(s.unrealized_pnl)}>{signed(s.unrealized_pnl, `₹${fmt.compact(s.unrealized_pnl)}`)}</span></>}
          />
          <MetricCard
            label="Win rate"
            value={s.stats.win_rate != null ? s.stats.win_rate * 100 : null}
            format={(n) => `${Math.round(n)}%`}
            note={`${s.stats.closed_trades} closed · max DD ${s.stats.max_drawdown_pct}%`}
          />
        </div>
      </Panel>

      <Panel className="span-8" title="Performance">
        <EquityCurve data={history?.equity_curve ?? []} theme={th} />
      </Panel>

      <Panel className="span-4" title="Risk">
        <div className="paper-risk-head">
          <span className="tile-label">Drawdown vs kill line</span>
          <span className="paper-risk-value">{s.stats.current_drawdown_pct}% <span className="cell-sub">of {s.config.kill_drawdown_pct}%</span></span>
        </div>
        <MeterBar value={s.stats.current_drawdown_pct} max={s.config.kill_drawdown_pct} label="Current drawdown against kill line" />
        <dl className="kv">
          <div><dt>Max drawdown</dt><dd>{s.stats.max_drawdown_pct}%</dd></div>
          <div><dt>Exposure</dt><dd>{exposurePct == null ? '—' : `${exposurePct.toFixed(1)}%`}</dd></div>
          <div><dt>Avg win</dt><dd style={pnlColor(s.stats.avg_win)}>{s.stats.avg_win == null ? '—' : `₹${fmt.int(s.stats.avg_win)}`}</dd></div>
          <div><dt>Avg loss</dt><dd style={pnlColor(s.stats.avg_loss)}>{s.stats.avg_loss == null ? '—' : `₹${fmt.int(s.stats.avg_loss)}`}</dd></div>
          <div><dt>Kill line</dt><dd>{s.stats.kill_breached ? <StatusPill tone="bad" label="Breached" /> : <StatusPill tone="good" label="Within limit" />}</dd></div>
        </dl>
      </Panel>

      <Panel title="Trade">
        {flash && <p className="notice notice-good">{flash}</p>}
        {error && <p className="notice notice-bad">{error}</p>}
        <form className="controls paper-trade" onSubmit={onTrade}>
          <input value={symbol} onChange={(e) => setSymbol(e.target.value)} placeholder="Symbol e.g. ITC" aria-label="Symbol" spellCheck={false} />
          <select value={side} onChange={(e) => setSide(e.target.value as 'BUY' | 'SELL')} aria-label="Side">
            <option value="BUY">Buy</option><option value="SELL">Sell</option>
          </select>
          <input value={qty} onChange={(e) => setQty(e.target.value)} placeholder="Qty" type="number" min="1" aria-label="Quantity" />
          <input value={price} onChange={(e) => setPrice(e.target.value)} placeholder="Price (blank = live)" type="number" step="0.05" aria-label="Price" />
          {side === 'BUY' && <input value={stop} onChange={(e) => setStop(e.target.value)} placeholder="Stop (opt)" type="number" step="0.05" aria-label="Stop" />}
          {side === 'BUY' && <input value={target} onChange={(e) => setTarget(e.target.value)} placeholder="Target (opt)" type="number" step="0.05" aria-label="Target" />}
          <Button type="submit" loading={busy} disabled={!symbol.trim() || !qty} className={side === 'SELL' ? 'is-sell' : ''}>
            {side === 'BUY' ? 'Buy' : 'Sell'}
          </Button>
        </form>
        <p className="chart-note">Leave price blank to fill at the live Groww price (market hours). Stop/target are for your reference and hit-flags; they don't auto-execute.</p>
      </Panel>

      <Panel title={`Open positions (${s.positions.length})`}>
        {s.positions.length === 0 ? <EmptyState title="No open paper positions." hint="Positions appear here after a paper buy fills." /> : (
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
                      {p.stop_hit && <span title="Stop hit" style={{ color: th.deltaDownText }}> ⚠stop</span>}
                      {p.target_hit && <span title="Target hit" style={{ color: th.deltaUpText }}> ✓target</span>}
                    </td>
                    <td><Button variant="mini" onClick={() => quickSell(p.symbol, p.yahoo_symbol, p.qty)}>Close</Button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      {orders.length > 0 && (
        <Panel title={`Pending orders (${orders.length})`}>
          <p className="chart-note">
            The auto-executor polls live prices ~every 12s during market hours and fills these when the price is hit.
          </p>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Symbol</th><th>Type</th><th className="num">Qty</th><th className="num">Fills at ₹</th><th>Placed</th><th></th></tr></thead>
              <tbody>
                {orders.map((o) => (
                  <tr key={o.id}>
                    <td><b>{o.symbol}</b></td>
                    <td>
                      {o.side === 'BUY' ? 'Limit buy (≤)' : o.trigger === 'gte' ? 'Take-profit (≥)' : 'Stop-loss (≤)'}
                    </td>
                    <td className="num">{fmt.int(o.qty)}</td>
                    <td className="num">{fmt.price(o.limit_price)}</td>
                    <td className="cell-sub">{o.created_at.slice(0, 16).replace('T', ' ')}</td>
                    <td>
                      <Button variant="mini" onClick={async () => { await cancelOrder(o.id); await load() }}>
                        Cancel
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}

      <Panel title={`Trade log (${history?.trades.length ?? 0})`}>
        {history && history.trades.length > 0 ? (
          <div className="table-wrap">
            <table>
              <thead><tr><th>When</th><th>Symbol</th><th>Side</th><th className="num">Qty</th><th className="num">Price ₹</th><th className="num">Charges ₹</th><th className="num">Realized ₹</th></tr></thead>
              <tbody>
                {history.trades.map((tr, i) => (
                  <tr key={i}>
                    <td className="cell-sub">{tr.ts.slice(0, 16).replace('T', ' ')}</td>
                    <td><b>{tr.symbol}</b></td>
                    <td style={{ color: tr.side === 'BUY' ? th.blue : th.orange }}>{tr.side}</td>
                    <td className="num">{fmt.int(tr.qty)}</td>
                    <td className="num">{fmt.price(tr.price)}</td>
                    <td className="num cell-sub">{fmt.price(tr.charges)}</td>
                    <td className="num" style={pnlColor(tr.realized_pnl)}>{tr.realized_pnl == null ? '—' : `${tr.realized_pnl >= 0 ? '+' : ''}${fmt.int(tr.realized_pnl)}`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <EmptyState title="No trades yet." hint="Executed paper fills are listed here." />}
      </Panel>

      <Panel title="Settings">
        <p className="chart-note">Brokerage% and tax% are charged on each transaction's value (both applied per fill). Set them to approximate your real all-in cost. Starting capital can only change while the account has no trades.</p>
        <div className="controls">
          <label className="paper-cfg"><span>Starting ₹</span><input value={cap} onChange={(e) => setCap(e.target.value)} type="number" /></label>
          <label className="paper-cfg"><span>Brokerage %</span><input value={brok} onChange={(e) => setBrok(e.target.value)} type="number" step="0.01" /></label>
          <label className="paper-cfg"><span>Tax %</span><input value={tax} onChange={(e) => setTax(e.target.value)} type="number" step="0.01" /></label>
          <label className="paper-cfg"><span>Kill drawdown %</span><input value={kill} onChange={(e) => setKill(e.target.value)} type="number" step="0.5" /></label>
          <Button onClick={saveConfig}>Save settings</Button>
        </div>
      </Panel>
    </main>
  )
}
