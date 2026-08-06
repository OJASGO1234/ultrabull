import type { ViewName } from '../types'

/** Shared "market context" on/off switch — used on the Analyze tab (per-run)
 * and the Holdings tab (per-table). Deliberately not persisted to
 * localStorage; see the `view` state comment in App.tsx. */
export function MarketContextToggle({ view, onChange }: { view: ViewName; onChange: (v: ViewName) => void }) {
  const stockOnly = view === 'stock_only'
  return (
    <div className="view-switch">
      <span className="view-switch-label">Market context</span>
      <div className="view-tabs" role="group" aria-label="Analysis view">
        <button
          type="button"
          className={`view-tab${view === 'market_aware' ? ' view-tab-on' : ''}`}
          aria-pressed={view === 'market_aware'}
          onClick={() => onChange('market_aware')}
          title="The validated default: NIFTY regime and the distribution-day ceiling gate the backtest, and all four market factors count toward the advice score"
        >
          On — market-aware
        </button>
        <button
          type="button"
          className={`view-tab${stockOnly ? ' view-tab-on' : ''}`}
          aria-pressed={stockOnly}
          onClick={() => onChange('stock_only')}
          title="Judge the stock in isolation: market-timing gates lifted and market factors dropped from the score"
        >
          Off — stock only
        </button>
      </div>
      <span className="view-switch-note">
        {stockOnly ? 'Judging the stock in isolation.' : 'The validated default.'}
      </span>
    </div>
  )
}
