import type { KeyIndicators as KI } from '../types'
import type { Theme } from '../theme'

const STATE_MARK: Record<string, string> = { bullish: '▲', bearish: '▼', neutral: '•' }

function stateColor(t: Theme, state: string): string {
  return state === 'bullish' ? t.deltaUpText : state === 'bearish' ? t.deltaDownText : t.muted
}

const LEAN_LABEL: Record<string, string> = {
  bullish: 'Bullish lean',
  bearish: 'Bearish lean',
  mixed: 'Mixed signals',
}

/** Compact, explained technical posture. Used as a card on the analyze view and
 * inside the holdings drawer (pass `compact` there to drop the wrapping card). */
export function KeyIndicators({ ki, theme: t, compact }: { ki: KI; theme: Theme; compact?: boolean }) {
  const leanColor =
    ki.lean === 'bullish' ? t.deltaUpText : ki.lean === 'bearish' ? t.deltaDownText : t.textSecondary
  const body = (
    <>
      <div className="ki-head">
        {!compact && <h2>Key indicators</h2>}
        <span className="ki-lean" style={{ color: leanColor }}>
          {compact && <b>Key indicators — </b>}
          {LEAN_LABEL[ki.lean]} · {ki.bullish_count}▲ / {ki.bearish_count}▼
        </span>
      </div>
      <ul className="ki-list">
        {ki.flags.map((f, i) => (
          <li key={i} className={f.ignored ? 'factor-ignored' : undefined}>
            <span className="ki-mark" style={{ color: f.ignored ? t.muted : stateColor(t, f.state) }}>
              {f.ignored ? '—' : STATE_MARK[f.state]}
            </span>
            <span className="ki-text">
              <b>{f.label}</b>
              {f.value && <span className="ki-value"> {f.value}</span>} — {f.detail}
              {f.ignored && <span className="factor-tag"> (not counted in this view)</span>}
            </span>
          </li>
        ))}
      </ul>
      {!compact && <p className="chart-note">{ki.note}</p>}
    </>
  )
  return compact ? <div className="ki-compact">{body}</div> : <div className="card">{body}</div>
}
