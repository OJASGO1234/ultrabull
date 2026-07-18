import type { Advice, Outlook } from '../types'
import type { Theme } from '../theme'
import { fmt } from './chart'

interface Props {
  advice: Advice
  outlook?: Outlook | null
  theme: Theme
}

const VERDICT_TEXT: Record<string, string> = {
  BUY: 'Buy',
  BUY_ON_DIP: 'Buy the dip',
  WAIT: 'Wait',
  AVOID: 'Avoid',
  HOLD: 'Hold',
  CONSIDER_TRIM: 'Consider trimming',
  TIGHTEN_STOP: 'Tighten stop',
  EXIT_OR_REDUCE: 'Exit / reduce',
}

function verdictColor(t: Theme, v: string): string {
  if (v === 'BUY' || v === 'HOLD') return t.deltaUpText
  if (v === 'AVOID' || v === 'EXIT_OR_REDUCE') return t.deltaDownText
  if (v === 'BUY_ON_DIP' || v === 'CONSIDER_TRIM' || v === 'TIGHTEN_STOP') return t.warning
  return t.textSecondary
}

/** Level-aware buy/sell synthesis: two verdicts (fresh money vs existing
 * holding), the factor checklist behind them, and a plan anchored on
 * support/resistance rather than arbitrary percentages. */
export function AdviceCard({ advice: a, outlook, theme: t }: Props) {
  const mark = (s: number) => (s > 0 ? '✓' : s < 0 ? '✗' : '·')
  const markColor = (s: number) =>
    s > 0 ? t.deltaUpText : s < 0 ? t.deltaDownText : t.muted

  // Reconcile with the Outlook card when the two point different ways —
  // a statistical tendency and a good entry are different claims.
  const biasUp =
    outlook != null &&
    (outlook.horizons['1m']?.verdict === 'up' || outlook.horizons['3m']?.verdict === 'up')
  const biasDown =
    outlook != null &&
    (outlook.horizons['1m']?.verdict === 'down' || outlook.horizons['3m']?.verdict === 'down')
  const cautious = ['WAIT', 'AVOID', 'BUY_ON_DIP'].includes(a.fresh.verdict)
  const reconcile =
    biasUp && cautious
      ? 'Why "Up bias" below but caution here? The Outlook is a base rate — stocks in today\'s ' +
        'state drifted up over 1–3 months slightly more often than not, averaged across every ' +
        'similar historical situation. This card judges the specific trade available at today\'s ' +
        'price, and the ✗ factors above (entry quality, reward:risk, market health) say the ' +
        'tendency isn\'t worth paying this price for. A mild tailwind is not an entry signal — ' +
        'the plan\'s entry zone is where the two align.'
      : biasDown && a.fresh.verdict === 'BUY'
        ? 'Note: the Outlook base rates lean down for this state — this buy call rests on the ' +
          'entry-quality factors above, so keep the stop honest.'
        : null

  return (
    <div className="card">
      <div className="card-head">
        <h2>Buy / sell signal — level-aware synthesis</h2>
        <span className="hover-readout">as of {a.as_of} · close ₹{fmt.price(a.close)} · net score {a.score > 0 ? '+' : ''}{a.score}</span>
      </div>

      <div className="tiles">
        <div className="tile">
          <div className="tile-label">Fresh position</div>
          <div className="tile-value" style={{ color: verdictColor(t, a.fresh.verdict) }}>
            {VERDICT_TEXT[a.fresh.verdict] ?? a.fresh.verdict}
          </div>
          <div className="tile-note">{a.fresh.reason}</div>
        </div>
        <div className="tile">
          <div className="tile-label">If already holding</div>
          <div className="tile-value" style={{ color: verdictColor(t, a.holding.verdict) }}>
            {VERDICT_TEXT[a.holding.verdict] ?? a.holding.verdict}
          </div>
          <div className="tile-note">{a.holding.reason}</div>
        </div>
        <div className="tile">
          <div className="tile-label">Plan (level-anchored)</div>
          <div className="tile-value" style={{ fontSize: 14, lineHeight: 1.5 }}>
            {a.plan.entry_zone_low != null && (
              <>entry ₹{fmt.price(a.plan.entry_zone_low)}–{fmt.price(a.plan.entry_zone_high)}<br /></>
            )}
            stop ₹{fmt.price(a.plan.stop)} · target ₹{fmt.price(a.plan.target1)}
            {a.plan.target2 != null && <> → ₹{fmt.price(a.plan.target2)}</>}
            {a.plan.reward_risk != null && <> · {a.plan.reward_risk}:1</>}
          </div>
          <div className="tile-note">{a.plan.stop_basis}</div>
        </div>
      </div>

      <ul className="factor-list">
        {a.factors.map((f) => (
          <li key={f.name}>
            <span className="factor-mark" style={{ color: markColor(f.score) }}>{mark(f.score)}</span>
            <b>{f.name}</b> — {f.detail}
          </li>
        ))}
      </ul>
      {reconcile && <p className="chart-note"><b>{reconcile}</b></p>}
      <p className="chart-note">{a.note}</p>
    </div>
  )
}
