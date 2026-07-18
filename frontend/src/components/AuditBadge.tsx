import type { Audit } from '../types'

/** Simulator self-check result. Status colors never travel alone: icon + label. */
export function AuditBadge({ audit }: { audit: Audit }) {
  if (audit.ok) {
    return (
      <span className="badge badge-good" title="Every SL/TP exit price matched its theoretical level (incl. slippage) within tolerance">
        ✓ Audit clean — {audit.checked} SL/TP exits verified
      </span>
    )
  }
  return (
    <span className="badge badge-critical" title="Some exit fills disagree with the SL/TP + slippage model — treat metrics from this run as suspect">
      ✕ Audit FAILED — {audit.mismatch_count} of {audit.checked} exits mismatched
    </span>
  )
}
