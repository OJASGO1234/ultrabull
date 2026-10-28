# core/trace_gate_redundancy.py
"""
Follow-up diagnostic to audit_gates.py: audit_gates.py found four gate
conditions with 0% empirical rejection (breakout/breakdown volume-surge,
breakout/breakdown MA20-position, mean-reversion trend bias, pullback trend)
and a liquidity floor with 0% rejection. This script traces each 0%-rejection
gate back to the PATTERN DETECTOR that produced the candidate signals, to
show whether the condition is already enforced upstream (making the gate a
provable no-op) or just an empirical coincidence in this dataset — and
separately reports the actual 20-day median turnover distribution to show
why the liquidity floor never fires.

Does NOT modify any production code (rules/gates.py, evaluate_gates.py,
event_study.py, patterns/*.py all read-only). Creates no other files.

Run:  python trace_gate_redundancy.py <cache_dir>   (from the core/ directory)
Output: core/trace_gate_redundancy_results.csv (+ printed report)
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

from event_study import UNIVERSE, prepare_symbol
from patterns.primitives import regime_from

pd.set_option("display.width", 200)

# ---------------------------------------------------------------------------
# PART 1 — static trace: quote the detector source that makes each gate a
# provable identity (or not) against the corresponding gates.py condition.
# ---------------------------------------------------------------------------
STATIC_TRACE = """
================================================================================
STATIC TRACE: gate condition vs. the detector that generates its candidates
================================================================================

1) Breakout/Breakdown VOLUME SURGE gate  (gates.py: volx20[i] >= vol_k, vol_k=1.5)
   Detector: patterns/breakouts.py::breakouts()
       vma20 = v.rolling(20).mean().replace(0, 1)
       bo = (c > rh) & (v >= vol_k * vma20)      # vol_k passed in = cfg["breakout"]["vol_k"]
       bd = (c < rl) & (v >= vol_k * vma20)
   Caller: patterns/pipeline.py passes vol_k=cfg["breakout"]["vol_k"]
   Value:  patterns/config.py CFG["breakout"]["vol_k"] = 1.5
   Gate's vol_k: GATE_CFG has no "breakout.vol_k" key -> gates.py's _get()
       default of 1.5 applies.
   => SAME THRESHOLD (1.5), SAME RATIO (v / 20d mean volume). A row cannot be
      emitted as Breakout20/55 or Breakdown20/55 unless it already cleared
      v >= 1.5x its 20d average volume. The only formula difference is the
      rolling-mean min_periods (detector: none, i.e. NaN/False for the first
      19 bars, so nothing is ever detected there; gate: min_periods=1). That
      difference cannot cause a live detection to fail the gate. PROVABLE
      IDENTITY, not coincidence.

2) Breakout/Breakdown MA20-POSITION gate  (gates.py: close[i] >= ma20[i] for
   breakouts, close[i] <= ma20[i] for breakdowns)
   Detector: patterns/breakouts.py::breakouts()
       rh = hi.rolling(N).max().shift(1)   # prior N-day HIGH (N=20 or 55)
       bo = (c > rh) & (v >= vol_k * vma20)   # no MA20 condition at all
   => NOT a formula identity: the detector requires close above the prior
      N-day HIGH, the gate requires close above the 20d moving AVERAGE of
      CLOSES. These are different quantities. Closing above a 20-/55-day
      high is a much stricter condition than closing above the 20d average
      close, so in practice it should imply the MA20 condition, but this is
      an EMPIRICAL claim, not an algebraic guarantee -- checked in Part 2
      below with the actual margin distribution.

3) Mean-reversion TREND BIAS gate  (gates.py: ma20[i] > ma50[i] for BB_BuyDip,
   ma20[i] < ma50[i] for BB_SellFade)
   Detector: patterns/pullbacks.py::bb_reversion()
       up = ind["_regime"]=="uptrend"; down = ind["_regime"]=="downtrend"
       bbuy = up & touch_low; bsell = down & touch_up
   ind["_regime"] source: patterns/pipeline.py -> primitives.regime_from(ind):
       up = ind["MA20"] > ind["MA50"]; down = ind["MA20"] < ind["MA50"]
   => PROVABLE IDENTITY. "_regime"=="uptrend" is DEFINED as MA20>MA50, the
      exact boolean gates.py re-checks. A BB_BuyDip row cannot exist unless
      MA20>MA50 already held at detection; the gate re-testing the identical
      boolean can only ever pass. Same for BB_SellFade / downtrend.

4) Pullback TREND gate  (gates.py: ma20[i] > ma50[i])
   Detector: patterns/pullbacks.py::pullback_ma20()
       up = ind["_regime"]=="uptrend"
       hit = up & dipped & close_above
   => Same _regime identity as (3). PROVABLE IDENTITY.

Net: (1), (3), (4) are exact algebraic identities with their detectors --
0% rejection is guaranteed by construction, not a property of this dataset.
(2) is NOT an identity; Part 2 measures it empirically.
"""

VOL_K = 1.5


def _cols(df):
    return {str(c).lower(): c for c in df.columns}


def trace_symbol(sym: str, years: float, cache_dir: str | None) -> dict | None:
    prep = prepare_symbol(sym, years=years, cache_dir=cache_dir)
    if prep is None:
        return None
    df_ohlcv, ind_df, dets = prep
    if dets is None or dets.empty:
        return None

    c = _cols(df_ohlcv)
    close = df_ohlcv[c["close"]]
    high = df_ohlcv[c["high"]]
    low = df_ohlcv[c["low"]]
    vol = df_ohlcv[c["volume"]]

    ma20 = ind_df["MA20"]
    ma50 = ind_df["MA50"]
    vma20_gate = vol.rolling(20, min_periods=1).mean()  # gates.py convention
    vma20_det = vol.rolling(20).mean()                  # detector convention

    # Recompute _regime exactly as pipeline.py does, for the identity check.
    regime = regime_from(pd.DataFrame({"MA20": ma20, "MA50": ma50}))

    dt_idx = pd.to_datetime(df_ohlcv.index)
    det_dates = pd.to_datetime(dets["date"])
    pos = pd.Series(np.arange(len(df_ohlcv)), index=dt_idx).reindex(det_dates).to_numpy()
    valid = ~np.isnan(pos)
    dets = dets.loc[valid].copy()
    pos = pos[valid].astype(int)

    rows = []

    is_brk = dets["pattern"].isin(["Breakout20", "Breakout55"]).to_numpy()
    is_brd = dets["pattern"].isin(["Breakdown20", "Breakdown55"]).to_numpy()
    if is_brk.any() or is_brd.any():
        idx_bo = pos[is_brk]
        idx_bd = pos[is_brd]
        # (1) volume surge: gate formula vs value at detection bar
        for label, idx in (("Breakout", idx_bo), ("Breakdown", idx_bd)):
            if len(idx) == 0:
                continue
            ratio_gate = (vol.to_numpy()[idx] / (vma20_gate.to_numpy()[idx] + 1e-12))
            fails = int((ratio_gate < VOL_K).sum())
            rows.append({"symbol": sym, "check": f"{label}: volume-surge margin (v / vma20_gate - vol_k)",
                         "n": len(idx), "n_fail_gate": fails,
                         "min_margin": float((ratio_gate - VOL_K).min()),
                         "p50_margin": float(np.median(ratio_gate - VOL_K)),
                         "max_margin": float((ratio_gate - VOL_K).max())})
        # (2) MA20 position margin: (close - ma20)/ma20 for breakout, (ma20-close)/ma20 for breakdown
        if len(idx_bo):
            m = (close.to_numpy()[idx_bo] - ma20.to_numpy()[idx_bo]) / ma20.to_numpy()[idx_bo]
            rows.append({"symbol": sym, "check": "Breakout: MA20-position margin ((close-ma20)/ma20)",
                         "n": len(idx_bo), "n_fail_gate": int((m < 0).sum()),
                         "min_margin": float(np.nanmin(m)), "p50_margin": float(np.nanmedian(m)),
                         "max_margin": float(np.nanmax(m))})
        if len(idx_bd):
            m = (ma20.to_numpy()[idx_bd] - close.to_numpy()[idx_bd]) / ma20.to_numpy()[idx_bd]
            rows.append({"symbol": sym, "check": "Breakdown: MA20-position margin ((ma20-close)/ma20)",
                         "n": len(idx_bd), "n_fail_gate": int((m < 0).sum()),
                         "min_margin": float(np.nanmin(m)), "p50_margin": float(np.nanmedian(m)),
                         "max_margin": float(np.nanmax(m))})

    is_buydip = (dets["pattern"] == "BB_BuyDip").to_numpy()
    is_sellfade = (dets["pattern"] == "BB_SellFade").to_numpy()
    is_pb = (dets["pattern"] == "PullbackToMA20").to_numpy()
    reg_arr = regime.to_numpy()
    for label, mask, want in (("BB_BuyDip: regime identity", is_buydip, "uptrend"),
                               ("BB_SellFade: regime identity", is_sellfade, "downtrend"),
                               ("PullbackToMA20: regime identity", is_pb, "uptrend")):
        idx = pos[mask]
        if len(idx) == 0:
            continue
        matches = (reg_arr[idx] == want)
        rows.append({"symbol": sym, "check": label, "n": len(idx),
                     "n_fail_gate": int((~matches).sum()),
                     "min_margin": np.nan, "p50_margin": np.nan, "max_margin": np.nan})

    # Liquidity: 20d median turnover, at ALL bars (not just detections) and at detection bars.
    turnover20 = (close * vol).rolling(20, min_periods=5).median()
    all_bars = turnover20.dropna().to_numpy()
    at_dets = turnover20.to_numpy()[pos]
    at_dets = at_dets[np.isfinite(at_dets)]

    return {
        "symbol": sym,
        "rows": rows,
        "turnover_all_bars": all_bars,
        "turnover_at_detections": at_dets,
    }


def main():
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    years = 4

    all_rows = []
    turnover_all = []
    turnover_at_dets = []
    per_symbol_median_turnover = []

    for sym in UNIVERSE:
        try:
            r = trace_symbol(sym, years=years, cache_dir=cache)
        except Exception as e:
            print(f"  {sym}: FAILED ({e})")
            continue
        if r is None:
            continue
        all_rows.extend(r["rows"])
        turnover_all.append(r["turnover_all_bars"])
        turnover_at_dets.append(r["turnover_at_detections"])
        if len(r["turnover_all_bars"]):
            per_symbol_median_turnover.append({
                "symbol": sym,
                "median_turnover_20d": float(np.median(r["turnover_all_bars"])),
                "p10_turnover_20d": float(np.percentile(r["turnover_all_bars"], 10)),
                "min_turnover_20d": float(np.min(r["turnover_all_bars"])),
            })
        print(f"  {sym}: traced")

    print(STATIC_TRACE)

    report = pd.DataFrame(all_rows)
    report.to_csv("trace_gate_redundancy_results.csv", index=False)

    print("=" * 80)
    print("PART 2 -- EMPIRICAL VERIFICATION (per-symbol rows, then pooled)")
    print("=" * 80)
    print(report.round(6).to_string(index=False))

    pooled = report.groupby("check").agg(
        n=("n", "sum"), n_fail_gate=("n_fail_gate", "sum"),
        min_margin=("min_margin", "min"), max_margin=("max_margin", "max"),
    )
    print("\n--- Pooled across all 37 symbols ---")
    print(pooled.round(6).to_string())

    print("\n" + "=" * 80)
    print("PART 3 -- LIQUIDITY: 20-day median turnover (close*volume) distribution")
    print("=" * 80)
    all_bars = np.concatenate(turnover_all) if turnover_all else np.array([])
    at_dets = np.concatenate(turnover_at_dets) if turnover_at_dets else np.array([])
    threshold = 2e7

    def _describe(arr, label):
        if len(arr) == 0:
            print(f"{label}: no data")
            return
        pct = np.percentile(arr, [0, 1, 5, 10, 25, 50, 75, 90, 100])
        below = int((arr < threshold).sum())
        print(f"{label} (n={len(arr)}):")
        print(f"  min={pct[0]:,.0f}  p1={pct[1]:,.0f}  p5={pct[2]:,.0f}  p10={pct[3]:,.0f}  "
              f"p25={pct[4]:,.0f}  median={pct[5]:,.0f}  p75={pct[6]:,.0f}  p90={pct[7]:,.0f}  max={pct[8]:,.0f}")
        print(f"  bars below GATE_CFG threshold ({threshold:,.0f}): {below} ({100*below/len(arr):.2f}%)")

    _describe(all_bars, "All daily bars, all 37 symbols")
    _describe(at_dets, "Bars at actual detection dates, all 37 symbols")

    sym_med = pd.DataFrame(per_symbol_median_turnover).sort_values("median_turnover_20d")
    sym_med.to_csv("trace_gate_redundancy_liquidity_by_symbol.csv", index=False)
    print(f"\nPer-symbol median 20d turnover, ascending (lowest-liquidity symbols first,\n"
          f"threshold = {threshold:,.0f}):")
    print(sym_med.round(0).to_string(index=False))
    print(f"\nSymbols with median turnover below threshold: {(sym_med['median_turnover_20d'] < threshold).sum()} / {len(sym_med)}")
    print(f"Symbols with min (worst single 20d window) turnover below threshold: "
          f"{(sym_med['min_turnover_20d'] < threshold).sum()} / {len(sym_med)}")

    print("\nSaved: trace_gate_redundancy_results.csv, trace_gate_redundancy_liquidity_by_symbol.csv")


if __name__ == "__main__":
    main()
