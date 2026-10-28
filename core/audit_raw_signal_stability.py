# core/audit_raw_signal_stability.py
"""
Chronological RAW-SIGNAL predictive-edge stability audit, using the
existing event_study.py framework (compute_event_rows: 1/3/5/10-bar
signed forward returns and excess-vs-unconditional-drift) -- BEFORE any
gate, qualification filter, or the sim engine's SL/TP/sizing/costs. This
isolates whether the pattern DEFINITIONS themselves still have directional
edge in raw detections, independent of everything downstream that the
prior trade-level audits already covered.

For every currently-traded directional pattern (the patterns NOT in
build_production_cfg()'s disable_patterns list, determined programmatically
-- not hardcoded), raw detections (run_all_patterns output, no gates
applied) are split into the SAME 8 equal chronological blocks per symbol
used by audit_rolling_pattern_stability.py, so the raw-signal timeline can
be directly compared block-for-block against that script's trade-level
findings -- i.e. this script answers "does the raw edge flip sign in the
same block where realized trading performance broke down, an earlier
block, or a later one?"

MFE/MAE are computed here independently (diagnostic only) as a fixed
10-bar-forward excursion relative to the detection bar's own close --
NOT the same thing as a realized trade's MAE/MFE (which is relative to an
actual delayed entry price and an actual SL/TP-bounded exit window); this
is a raw-signal proxy, labeled as such.

Reuses event_study.UNIVERSE/prepare_symbol/compute_event_rows/HORIZONS and
patterns.pipeline.run_all_patterns/patterns.config.CFG and
patterns.postprocess.direction_of (all imported, not retyped) plus
audit_baseline_backtest.build_production_cfg() (for the disable_patterns
list only -- not used to gate/trade anything here). Does NOT modify any
production file. Does NOT optimize or search any threshold. Diagnosis only.

Run:  python audit_raw_signal_stability.py <cache_dir>
Output: core/audit_raw_signal_stability_results.csv (+ printed report)
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

from event_study import UNIVERSE, prepare_symbol, compute_event_rows, HORIZONS
from patterns.postprocess import direction_of
from audit_baseline_backtest import build_production_cfg

N_BLOCKS = 8  # matches audit_rolling_pattern_stability.py for direct comparability
TRAIN_BLOCKS = list(range(0, 6))   # blocks 0-5 == first 75% (close to the 70% split used elsewhere)
VALID_BLOCKS = list(range(6, 8))   # blocks 6-7 == last 25%
OUT_ROWS = []


def log(section, rows):
    for r in rows:
        d = dict(r); d["section"] = section
        OUT_ROWS.append(d)


def collect(cache):
    active_disabled = set(build_production_cfg()["disable_patterns"])
    all_events = []
    all_mae_mfe = []
    bad = []
    for sym in UNIVERSE:
        prep = prepare_symbol(sym, years=4, cache_dir=cache)
        if prep is None:
            bad.append(sym)
            continue
        df_ohlcv, ind_df, dets = prep
        if dets is None or dets.empty:
            continue
        dets = dets[~dets["pattern"].isin(active_disabled)].copy()
        dets = dets[dets["pattern"].map(direction_of) != "neutral"]
        if dets.empty:
            continue

        n = len(df_ohlcv)
        dt_to_pos = pd.Series(np.arange(n), index=pd.to_datetime(df_ohlcv.index))
        pos = dt_to_pos.reindex(pd.to_datetime(dets["date"])).to_numpy()
        valid_mask = ~np.isnan(pos)
        dets = dets.loc[valid_mask].copy()
        pos = pos[valid_mask].astype(int)
        dets["bar_idx"] = pos
        dets["block"] = np.minimum(N_BLOCKS - 1, (pos / n * N_BLOCKS).astype(int))

        ev = compute_event_rows(dets, df_ohlcv, sym)
        if ev is not None and len(ev):
            ev = ev.merge(dets[["date", "pattern", "block", "bar_idx"]].drop_duplicates(["date", "pattern"]),
                           on=["date", "pattern"], how="left")
            ev["symbol"] = sym
            all_events.append(ev)

        # raw-signal MAE/MFE: fixed 10-bar-forward excursion from the detection
        # bar's own close (a proxy -- NOT a realized trade's MAE/MFE)
        high, low, close = df_ohlcv["High"].to_numpy(), df_ohlcv["Low"].to_numpy(), df_ohlcv["Close"].to_numpy()
        max_h = max(HORIZONS)
        for _, r in dets.iterrows():
            b = int(r["bar_idx"])
            lo, hi = b, min(n - 1, b + max_h)
            if hi <= lo:
                continue
            entry_px = close[b]
            window_high, window_low = high[lo + 1:hi + 1], low[lo + 1:hi + 1]
            if len(window_high) == 0:
                continue
            d = direction_of(r["pattern"])
            if d == "bullish":
                mfe = (window_high.max() - entry_px) / entry_px * 100.0
                mae = (window_low.min() - entry_px) / entry_px * 100.0
            else:
                mfe = (entry_px - window_low.min()) / entry_px * 100.0
                mae = (entry_px - window_high.max()) / entry_px * 100.0
            all_mae_mfe.append({"symbol": sym, "pattern": r["pattern"], "block": int(r["block"]),
                                 "mae_pct": mae, "mfe_pct": mfe})

    events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    mae_mfe = pd.DataFrame(all_mae_mfe)
    return events, mae_mfe, bad, active_disabled


def block_table(ev: pd.DataFrame, pattern: str, horizon: int) -> pd.DataFrame:
    sub = ev[(ev["pattern"] == pattern) & (ev["horizon"] == horizon)]
    rows = []
    for b in range(N_BLOCKS):
        g = sub[sub["block"] == b]
        rows.append({
            "block": b, "n": len(g),
            "mean_signed_ret_bps": float(g["signed_ret"].mean() * 1e4) if len(g) else np.nan,
            "mean_excess_bps": float(g["excess"].mean() * 1e4) if len(g) else np.nan,
        })
    return pd.DataFrame(rows)


def first_sign_flip(bt: pd.DataFrame, col="mean_excess_bps"):
    """First block where sign flips from positive to negative (scanning in
    chronological order), among blocks with n>0."""
    vals = bt[bt["n"] > 0][["block", col]].dropna()
    if len(vals) < 2:
        return None
    prev_sign = None
    for _, row in vals.iterrows():
        sign = row[col] > 0
        if prev_sign is True and sign is False:
            return int(row["block"])
        prev_sign = sign
    return None


def main():
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    print("Collecting RAW detections (no gates applied) for every currently-traded directional "
          "pattern, computing 1/3/5/10-bar signed forward returns via event_study.compute_event_rows "
          "(unmodified), split into the same 8 chronological blocks used by "
          "audit_rolling_pattern_stability.py for direct comparison.\n")

    events, mae_mfe, bad, active_disabled = collect(cache)
    active_patterns = sorted(events["pattern"].unique()) if len(events) else []
    print(f"Active (currently-traded) directional patterns found: {active_patterns}")
    print(f"Disabled patterns excluded (per build_production_cfg): {sorted(active_disabled)}")
    print(f"Total raw-signal event rows: {len(events)} ({len(bad)} symbols with data issues)\n")

    pd.set_option("display.width", 240)
    flip_summary_rows = []

    for pat in active_patterns:
        print("\n" + "=" * 100)
        print(f"PATTERN: {pat}")
        print("=" * 100)
        for h in HORIZONS:
            bt = block_table(events, pat, h)
            print(f"\n  horizon = {h} bar(s):")
            print("  " + bt.round(2).to_string(index=False).replace("\n", "\n  "))
            log("BLOCK_TABLE", [{"pattern": pat, "horizon": h, **r} for r in bt.to_dict("records")])

            flip_block = first_sign_flip(bt)
            train_mean = bt[bt["block"].isin(TRAIN_BLOCKS)]["mean_excess_bps"].mean()
            valid_mean = bt[bt["block"].isin(VALID_BLOCKS)]["mean_excess_bps"].mean()
            train_n = bt[bt["block"].isin(TRAIN_BLOCKS)]["n"].sum()
            valid_n = bt[bt["block"].isin(VALID_BLOCKS)]["n"].sum()
            print(f"  -> train(blocks0-5) mean excess: {train_mean:.2f} bps (n={train_n}), "
                  f"valid(blocks6-7) mean excess: {valid_mean:.2f} bps (n={valid_n}); "
                  f"first positive->negative flip at block: {flip_block}")
            flip_summary_rows.append({
                "pattern": pat, "horizon": h, "train_mean_excess_bps": train_mean, "valid_mean_excess_bps": valid_mean,
                "train_n": int(train_n), "valid_n": int(valid_n),
                "edge_sign_train": "positive" if train_mean > 0 else "negative",
                "edge_sign_valid": "positive" if valid_mean > 0 else "negative",
                "first_sign_flip_block": flip_block,
            })

        # raw-signal MAE/MFE, train vs valid
        mm = mae_mfe[mae_mfe["pattern"] == pat]
        mm_train = mm[mm["block"].isin(TRAIN_BLOCKS)]
        mm_valid = mm[mm["block"].isin(VALID_BLOCKS)]
        print(f"\n  Raw-signal MAE/MFE (10-bar forward proxy, NOT a realized trade's MAE/MFE):")
        print(f"    train: n={len(mm_train)}, avg_mae={mm_train['mae_pct'].mean():.3f}%, avg_mfe={mm_train['mfe_pct'].mean():.3f}%")
        print(f"    valid: n={len(mm_valid)}, avg_mae={mm_valid['mae_pct'].mean():.3f}%, avg_mfe={mm_valid['mfe_pct'].mean():.3f}%")
        log("MAE_MFE", [{"pattern": pat, "period": "train", "n": len(mm_train),
                           "avg_mae_pct": mm_train["mae_pct"].mean(), "avg_mfe_pct": mm_train["mfe_pct"].mean()},
                          {"pattern": pat, "period": "valid", "n": len(mm_valid),
                           "avg_mae_pct": mm_valid["mae_pct"].mean(), "avg_mfe_pct": mm_valid["mfe_pct"].mean()}])

    flip_df = pd.DataFrame(flip_summary_rows)
    print("\n" + "=" * 100)
    print("SUMMARY: raw-signal edge sign, train vs valid, and WHEN it flips (horizon=5 is primary)")
    print("=" * 100)
    print(flip_df[flip_df["horizon"] == 5].round(2).to_string(index=False))

    print("\nComparison to trade-level deterioration block (from audit_rolling_pattern_stability.py, "
          "same 8-block scheme): Hammer's trade-level collapse was concentrated in block 7; "
          "BearishEngulfing's broke specifically in the block6->7 transition; PullbackToMA20 was "
          "noisy throughout with its worst block also at block 7.")
    for pat in ["Hammer", "BearishEngulfing", "PullbackToMA20"]:
        row = flip_df[(flip_df["pattern"] == pat) & (flip_df["horizon"] == 5)]
        if len(row):
            r = row.iloc[0]
            timing = ("UNKNOWN (no flip found)" if r["first_sign_flip_block"] is None else
                      f"block {r['first_sign_flip_block']}")
            print(f"  {pat}: raw-signal edge sign flip at {timing}; train_excess={r['train_mean_excess_bps']:.2f}bps, "
                  f"valid_excess={r['valid_mean_excess_bps']:.2f}bps")

    log("FLIP_SUMMARY", flip_summary_rows)
    out_df = pd.DataFrame(OUT_ROWS)
    out_df.to_csv("audit_raw_signal_stability_results.csv", index=False)
    print(f"\nSaved: audit_raw_signal_stability_results.csv ({len(out_df)} rows)")
    print("\nNOTE: diagnosis only. No threshold searched/optimized, no new pattern searched for, "
          "no production file/parameter changed.")


if __name__ == "__main__":
    main()
