# core/audit_walkforward_validation.py
"""
Chronological out-of-sample / walk-forward validation, without any tuning
on the test ("valid") period.

Thin wrapper around the EXISTING walkforward.py module (core/walkforward.py),
which already implements exactly this design (see its own docstring):
  tune  = first 70% of bars per symbol, used only to RANK presets
  valid = last 30% of bars per symbol, an out-of-sample check
Every preset (including "baseline", the production-equivalent config) is a
FIXED config applied identically to both periods — nothing is refit or
re-optimized on the valid period. This script does not change that design,
does not add a new preset, and does not select/tune anything based on the
valid-period results; it only runs walkforward.py's own run()/summarize_wf()
functions (imported, not modified) and adds an explicit PASS/INVESTIGATE
verdict per preset based on whether tune-period performance holds up
out-of-sample.

Does NOT modify walkforward.py or any other production file.

Run:  python audit_walkforward_validation.py <cache_dir>   (from core/ directory)
Output: core/audit_walkforward_validation_results.csv (+ printed report)
NOTE: walkforward.py's own run(), if invoked separately, writes
walkforward_trades.csv — this script writes its own differently-named
summary CSV and does not overwrite that file.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

import walkforward as wf


def verdict_for(row: pd.Series) -> tuple[str, str]:
    """Simple, pre-declared (not fit-to-data) rule: PASS if the preset's
    valid-period profit factor is still >1 AND its valid-period expectancy
    keeps the same sign as tune. INVESTIGATE otherwise. This rule is fixed
    before looking at results and is not adjusted per-preset."""
    pf_tune, pf_valid = row.get("pf_tune", np.nan), row.get("pf_valid", np.nan)
    exp_tune, exp_valid = row.get("expectancy_tune", np.nan), row.get("expectancy_valid", np.nan)
    if pd.isna(pf_valid) or pd.isna(exp_valid):
        return "INVESTIGATE", "missing valid-period data (too few out-of-sample trades)"
    same_sign = np.sign(exp_tune) == np.sign(exp_valid) if pd.notna(exp_tune) else False
    if pf_valid > 1.0 and same_sign:
        return "PASS", f"pf_valid={pf_valid:.3f} > 1.0 and expectancy sign held (tune={exp_tune:.2f}, valid={exp_valid:.2f})"
    return "INVESTIGATE", f"pf_valid={pf_valid:.3f}, expectancy tune={exp_tune:.2f} valid={exp_valid:.2f} (sign held: {same_sign})"


def main():
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    print("Running walkforward.py::run() unmodified — presets are FIXED configs, "
          "nothing is fit or tuned on the valid (out-of-sample) period.\n")
    print(f"Presets evaluated (from walkforward.py, unmodified): {wf.PRESETS}")
    print(f"Split: {int(0.7*100)}% tune / {int(0.3*100)}% valid, per-symbol chronological split\n")

    trades = wf.run(cache_dir=cache)
    print(f"{len(trades)} trades across {trades['symbol'].nunique()} symbols, "
          f"{trades['preset'].nunique()} presets")

    summary = wf.summarize_wf(trades)
    pd.set_option("display.width", 220)
    print("\n=== Walk-forward summary (from walkforward.py::summarize_wf, unmodified) ===")
    print(summary.round(3).to_string())

    rows = []
    for preset, row in summary.iterrows():
        v, reason = verdict_for(row)
        rows.append({
            "preset": preset,
            "trades_tune": row.get("trades_tune", np.nan), "trades_valid": row.get("trades_valid", np.nan),
            "win_rate_tune": row.get("win_rate_tune", np.nan), "win_rate_valid": row.get("win_rate_valid", np.nan),
            "pf_tune": row.get("pf_tune", np.nan), "pf_valid": row.get("pf_valid", np.nan),
            "expectancy_tune": row.get("expectancy_tune", np.nan), "expectancy_valid": row.get("expectancy_valid", np.nan),
            "delta_pf_valid_minus_tune": row.get("pf_valid", np.nan) - row.get("pf_tune", np.nan),
            "delta_expectancy_valid_minus_tune": row.get("expectancy_valid", np.nan) - row.get("expectancy_tune", np.nan),
            "verdict": v, "verdict_reason": reason,
        })
    out = pd.DataFrame(rows)

    print("\n=== Per-preset PASS/INVESTIGATE verdict (rule fixed in advance: "
          "pf_valid>1.0 AND expectancy sign matches tune) ===")
    print(out.round(3).to_string(index=False))

    out.to_csv("audit_walkforward_validation_results.csv", index=False)

    baseline_row = out[out["preset"] == "baseline"]
    print("\n=== Focus: 'baseline' preset (production-equivalent config) ===")
    if len(baseline_row):
        print(baseline_row.to_string(index=False))
    else:
        print("  'baseline' not present in results (check walkforward.py's PRESETS list / data availability).")

    print("\nLIMITATIONS:")
    print("  - This validates PRESET RANKING stability (does a fixed config's edge survive into an\n"
          "    unseen later period), not a live/forward-tested result — both periods are still\n"
          "    historical, just split chronologically per symbol.")
    print("  - Same overlapping-forward-return and no-shared-capital-portfolio caveats as every\n"
          "    other backtest audit in this series apply here too.")
    print("  - The verdict rule above (pf_valid>1.0, sign match) is a simple pre-declared threshold,\n"
          "    not a statistical significance test; treat it as directional, not proof of robustness.")
    print("\nSaved: audit_walkforward_validation_results.csv")


if __name__ == "__main__":
    main()
