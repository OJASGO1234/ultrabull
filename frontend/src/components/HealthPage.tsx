import { useCallback, useEffect, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Theme } from '../theme'
import {
  applyOverride, fetchAudit, fetchSteward, runAudit, runCalibrate, runSteward,
  type AuditReport, type CalibrationHorizon, type PatternRow, type PatternStatus, type Stat,
  type StewardAssessment,
} from '../health'

const HZ_LABEL: Record<string, string> = { '1d': '1 day', '1w': '1 week', '1m': '1 month' }

const STATUS_META: Record<PatternStatus, { label: string; tone: 'good' | 'warn' | 'bad' | 'info' | 'neutral' }> = {
  decayed: { label: 'Decayed', tone: 'bad' },
  negative: { label: 'Negative', tone: 'bad' },
  unconfirmed: { label: 'Unconfirmed', tone: 'warn' },
  emerging: { label: 'Emerging', tone: 'info' },
  holding: { label: 'Holding', tone: 'good' },
  no_edge: { label: 'No edge', tone: 'neutral' },
  insufficient: { label: 'Too few', tone: 'neutral' },
}

function tstat(s: Stat): string {
  if (s.t_stat == null) return `—`
  return `t=${s.t_stat} · ${s.mean_excess_bps != null ? (s.mean_excess_bps >= 0 ? '+' : '') + s.mean_excess_bps + 'bps' : ''} · n=${s.n}`
}

function CalibrationCard({ hz, name, theme: t }: { hz: CalibrationHorizon; name: string; theme: Theme }) {
  const skill = hz.brier < hz.brier_baseline - 0.002
  return (
    <div className="calib-block">
      <div className="calib-head">
        <b>{HZ_LABEL[name] ?? name}</b>
        <span className="cell-sub">
          ECE {(hz.ece * 100).toFixed(1)}% · Brier {hz.brier.toFixed(3)} vs {hz.brier_baseline} coin-flip ·{' '}
          <span style={{ color: skill ? t.deltaUpText : t.muted }}>{skill ? 'some skill' : '≈ coin flip'}</span>
        </span>
      </div>
      <div className="table-wrap">
        <table className="calib-table">
          <thead>
            <tr>
              <th>App says p(up)</th>
              <th className="num">Days</th>
              <th className="num">Predicted</th>
              <th className="num">Actually happened</th>
              <th className="num">Gap</th>
            </tr>
          </thead>
          <tbody>
            {hz.reliability.map((b) => {
              const gap = b.predicted - b.realized
              const over = gap > 0.02
              return (
                <tr key={b.bucket}>
                  <td>{b.bucket}</td>
                  <td className="num">{b.n.toLocaleString('en-IN')}</td>
                  <td className="num">{(b.predicted * 100).toFixed(1)}%</td>
                  <td className="num" style={{ color: over ? t.deltaDownText : t.textPrimary }}>
                    {(b.realized * 100).toFixed(1)}%
                  </td>
                  <td className="num" style={{ color: over ? t.deltaDownText : t.muted }}>
                    {gap >= 0 ? '+' : ''}{(gap * 100).toFixed(1)}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

const OVERALL_META: Record<string, { label: string; tone: 'good' | 'warn' | 'bad' }> = {
  stable: { label: 'Stable', tone: 'good' },
  watch: { label: 'Watch', tone: 'warn' },
  action_needed: { label: 'Action needed', tone: 'bad' },
}

function StewardCard({
  assessment, overrides, busy, error, onRun, onApply, theme: t,
}: {
  assessment: StewardAssessment | null
  overrides: string[]
  busy: boolean
  error: string | null
  onRun: () => void
  onApply: (pattern: string, disabled: boolean) => void
  theme: Theme
}) {
  const ov = OVERALL_META[assessment?.overall ?? 'watch']
  return (
    <div className="card">
      <div className="card-head">
        <h2>
          Claude's read
          {assessment && (
            <span className={`verdict verdict-${ov.tone}`} style={{ marginLeft: 10 }}>{ov.label}</span>
          )}
        </h2>
        <span className="hover-readout">
          {assessment && `${assessment.generated_at.slice(0, 16).replace('T', ' ')} · ${assessment.model}`}
          {'  '}
          <button type="button" className="btn-mini" disabled={busy} onClick={onRun}>
            {busy ? 'Thinking…' : assessment ? 'Re-run steward' : 'Run steward'}
          </button>
        </span>
      </div>
      <p className="chart-note">
        A Claude agent reads the audit below and proposes bounded actions — it judges decay-vs-hostile-market,
        never recomputes the numbers, and nothing takes effect until you click Apply. It can only disable an
        existing pattern (or revert a disable it made); the evidence-based base prunes stay untouchable.
      </p>
      {error && <p className="detail-warn">{error}</p>}
      {!assessment && !busy && !error && <p className="empty">No steward assessment yet — run it.</p>}
      {assessment && (
        <>
          <div className="chat-md steward-brief">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{assessment.briefing}</ReactMarkdown>
          </div>
          {assessment.proposals.length > 0 && (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Pattern</th><th>Proposal</th><th>Confidence</th><th>Rationale</th><th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {assessment.proposals.map((p) => {
                    const isDisabled = overrides.includes(p.pattern)
                    const canApply = p.action === 'disable' || p.action === 'reenable'
                    return (
                      <tr key={p.pattern}>
                        <td><b>{p.pattern}</b></td>
                        <td>
                          <span className={`verdict verdict-${p.action === 'disable' ? 'bad' : p.action === 'reenable' ? 'info' : 'neutral'}`}>
                            {p.action}
                          </span>
                          {p.needs_human_review && <span title="Adds risk — review carefully"> ⚠</span>}
                        </td>
                        <td className="cell-sub">{p.confidence}</td>
                        <td className="cell-sub" style={{ whiteSpace: 'normal', maxWidth: 460 }}>{p.rationale}</td>
                        <td>
                          {isDisabled ? (
                            <button type="button" className="btn-mini" onClick={() => onApply(p.pattern, false)}>
                              Re-enable
                            </button>
                          ) : canApply && p.action === 'disable' ? (
                            <button type="button" className="btn-mini" onClick={() => onApply(p.pattern, true)}>
                              Apply disable
                            </button>
                          ) : (
                            <span className="cell-sub">—</span>
                          )}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
          {overrides.length > 0 && (
            <p className="chart-note" style={{ color: t.deltaDownText }}>
              Currently disabled by steward: <b>{overrides.join(', ')}</b> — these are excluded from live verdicts.
            </p>
          )}
        </>
      )}
    </div>
  )
}

export function HealthPage({ theme: t }: { theme: Theme }) {
  const [report, setReport] = useState<AuditReport | null>(null)
  const [detail, setDetail] = useState<string | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [steward, setSteward] = useState<StewardAssessment | null>(null)
  const [overrides, setOverrides] = useState<string[]>([])
  const [stewardBusy, setStewardBusy] = useState(false)
  const [stewardError, setStewardError] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      const r = await fetchAudit()
      if (r.available && r.report) {
        setReport(r.report)
        setDetail(null)
      } else {
        setReport(null)
        setDetail(r.detail ?? 'No audit yet.')
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
    try {
      const s = await fetchSteward()
      setSteward(s.assessment)
      setOverrides(s.overrides)
    } catch {
      // steward optional — audit still shows
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  const onRunSteward = async () => {
    setStewardBusy(true)
    setStewardError(null)
    try {
      const r = await runSteward()
      setSteward(r.assessment)
    } catch (e) {
      setStewardError(e instanceof Error ? e.message : String(e))
    } finally {
      setStewardBusy(false)
    }
  }

  const onApply = async (pattern: string, disabled: boolean) => {
    try {
      const r = await applyOverride(pattern, disabled)
      setOverrides(r.disabled)
    } catch (e) {
      setStewardError(e instanceof Error ? e.message : String(e))
    }
  }

  const onRun = async () => {
    setRunning(true)
    setError(null)
    try {
      await runAudit()
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setRunning(false)
    }
  }

  const onRecalibrate = async () => {
    setRunning(true)
    setError(null)
    try {
      await runCalibrate()
      await runAudit() // re-audit so the before/after reflects the new fit
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setRunning(false)
    }
  }

  const decayed = report?.pattern_decay.patterns.filter((p) => p.status === 'decayed' || p.status === 'negative') ?? []

  return (
    <main className="results">
      <div className="card">
        <div className="card-head">
          <h2>Validation health</h2>
          <span className="hover-readout">
            {report && `audited ${report.generated_at.slice(0, 16).replace('T', ' ')} · ${report.symbols_used} symbols`}
            {'  '}
            <button type="button" className="btn-mini" disabled={running} onClick={onRun}>
              {running ? 'Running…' : 'Re-run audit'}
            </button>
            {'  '}
            <button type="button" className="btn-mini" disabled={running} onClick={onRecalibrate}
              title="Rebuild base rates on current data + refit the calibration, then re-audit">
              {running ? '…' : 'Recalibrate'}
            </button>
          </span>
        </div>
        <p className="chart-note">
          Does the system still work? Everything here is measured on <b>held-out time</b> — the recent 12 months are
          kept out of the fit, then the fit's predictions are checked against what actually happened. Deterministic;
          these are facts, not opinions. (A Claude steward that reads this and proposes gate changes is the next phase.)
        </p>
        {error && <div className="error-banner">⚠ {error}</div>}
        {!report && !error && <p className="empty">{detail ?? 'Loading…'}</p>}
      </div>

      <StewardCard
        assessment={steward}
        overrides={overrides}
        busy={stewardBusy}
        error={stewardError}
        onRun={onRunSteward}
        onApply={onApply}
        theme={t}
      />

      {report && (
        <>
          {decayed.length > 0 && (
            <div className="card">
              <h2>⚠ Needs attention</h2>
              <p className="chart-note">
                {decayed.map((p) => p.pattern).join(', ')} — {decayed.length === 1 ? 'this pattern' : 'these patterns'}{' '}
                had (or were validated with) an edge that has weakened or turned negative in the last year. Candidates to
                down-weight or disable.
              </p>
            </div>
          )}

          {report.calibration_fit && (
            <div className="card">
              <h2>Applied calibration — the fix</h2>
              <p className="chart-note">
                The raw states are over-confident (below), so the outlook shrinks each probability toward 50% by a
                fitted factor <b>k</b> (k=1 unchanged, k=0 pure coin-flip). Fit on held-out time to minimise error —
                this is what the app now shows. Fit {report.calibration_fit.fit_at.slice(0, 10)}.
              </p>
              <div className="table-wrap">
                <table className="calib-table">
                  <thead>
                    <tr>
                      <th>Horizon</th><th className="num">Shrink k</th>
                      <th className="num">ECE before → after</th><th className="num">Brier before → after</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(report.calibration_fit.horizons).map(([name, d]) => (
                      <tr key={name}>
                        <td>{HZ_LABEL[name] ?? name}</td>
                        <td className="num">{d.k.toFixed(2)}{d.k <= 0.1 ? ' (≈ no edge)' : ''}</td>
                        <td className="num">
                          {d.raw_ece != null ? `${(d.raw_ece * 100).toFixed(1)}% → ${(d.calibrated_ece! * 100).toFixed(1)}%` : '—'}
                        </td>
                        <td className="num">
                          {d.raw_brier != null ? `${d.raw_brier} → ${d.calibrated_brier}` : '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="chart-note">
                k near 0 at the 1-day and 1-week horizons is the honest finding, not a bug: day- and week-ahead
                direction is a coin flip in this data, so the outlook now says so.
              </p>
            </div>
          )}

          <div className="card">
            <h2>Calibration — raw states (the diagnostic)</h2>
            <p className="chart-note">
              When the app says "57 of 100 similar days closed higher," does that hold out-of-sample? Each row groups
              recent days by what the app <i>predicted</i> and shows what <i>actually</i> happened. A big positive gap =
              the app was over-confident in that band. ECE is the average gap; Brier near 0.25 means coin-flip skill.
            </p>
            {report.calibration.note ? (
              <p className="empty">{report.calibration.note}</p>
            ) : (
              Object.entries(report.calibration.horizons).map(([name, hz]) => (
                <CalibrationCard key={name} name={name} hz={hz} theme={t} />
              ))
            )}
          </div>

          <div className="card">
            <h2>Pattern edge — then vs. now</h2>
            <p className="chart-note">
              Forward-return excess (vs. each stock's own drift) at {report.pattern_decay.headline_horizon_bars} bars,
              split at {report.pattern_decay.cutoff}: the earlier window vs. the recent 12 months. ★ = validated in the
              original 2026 study. |t| ≥ 2 is the significance bar.
            </p>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Pattern</th>
                    <th>Status</th>
                    <th>Earlier window</th>
                    <th>Recent 12 months</th>
                  </tr>
                </thead>
                <tbody>
                  {report.pattern_decay.patterns.map((p: PatternRow) => {
                    const meta = STATUS_META[p.status]
                    return (
                      <tr key={p.pattern}>
                        <td>
                          {p.validated_2026 && <span title="Validated in the 2026 study">★ </span>}
                          <b>{p.pattern}</b>
                        </td>
                        <td><span className={`verdict verdict-${meta.tone}`}>{meta.label}</span></td>
                        <td className="cell-sub">{tstat(p.early)}</td>
                        <td className="cell-sub">{tstat(p.recent_12m)}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card">
            <h2>Market-regime gate</h2>
            <p className="chart-note">
              The validated gate keeps long trades only when NIFTY is in an uptrend. Its value = how much filtering to
              up-markets improves the long pool (at {report.regime_gate.horizon_bars} bars). Positive means it still helps.
            </p>
            <p>
              Historically <b>{report.regime_gate.full?.gate_gain_bps ?? '—'} bps</b>, recent 12 months{' '}
              <b style={{ color: (report.regime_gate.recent_12m?.gate_gain_bps ?? 0) >= 0 ? t.deltaUpText : t.deltaDownText }}>
                {report.regime_gate.recent_12m?.gate_gain_bps ?? '—'} bps
              </b>{' '}
              — <span className={`verdict verdict-${report.regime_gate.status === 'holding' ? 'good' : 'warn'}`}>
                {report.regime_gate.status === 'holding' ? 'holding' : 'watch'}
              </span>
            </p>
          </div>

          <div className="card">
            <h2>Holdings coverage</h2>
            <p className="chart-note">
              The verdicts and base rates are fit on {report.coverage.universe_size} NIFTY heavyweights. Holdings outside
              that set are scored by statistics that don't describe them — the case for expanding the universe.
            </p>
            <p>
              {report.coverage.coverage_pct != null
                ? `${report.coverage.coverage_pct}% of your holdings are in the study universe.`
                : 'No holdings detected.'}
              {report.coverage.not_covered.length > 0 && (
                <>
                  {' '}
                  <span style={{ color: t.deltaDownText }}>Not covered:</span>{' '}
                  {report.coverage.not_covered.join(', ')}.
                </>
              )}
            </p>
          </div>
        </>
      )}
    </main>
  )
}
