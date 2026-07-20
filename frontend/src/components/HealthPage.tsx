import { useCallback, useEffect, useState } from 'react'
import type { Theme } from '../theme'
import {
  fetchAudit, runAudit,
  type AuditReport, type CalibrationHorizon, type PatternRow, type PatternStatus, type Stat,
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

export function HealthPage({ theme: t }: { theme: Theme }) {
  const [report, setReport] = useState<AuditReport | null>(null)
  const [detail, setDetail] = useState<string | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)

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
  }, [])

  useEffect(() => {
    void load()
  }, [load])

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
              {running ? 'Running (~1 min)…' : 'Re-run audit'}
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

          <div className="card">
            <h2>Calibration — is the outlook honest?</h2>
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
