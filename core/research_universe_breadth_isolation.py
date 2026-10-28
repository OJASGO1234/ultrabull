# core/research_universe_breadth_isolation.py
"""
Follow-up to core/MOMENTUM_STRATEGY.md section 14's conclusion: section 14
changed TWO variables at once relative to the original baseline (section 8)
-- universe size (37 -> 111 symbols) AND history length (4y -> 20y) -- and
found the improvement held even in the sub-window matching the original
test's exact dates, pointing at universe concentration rather than history
length as the likely explanation. This script isolates that claim: hold
history length FIXED at the original 4 years, hold the config FIXED at the
original baseline (trail_atr=1.5, untouched), and change ONLY the universe
(37 -> the same 111-symbol union used in section 14).

FROZEN, UNCHANGED FROM THE ORIGINAL BASELINE:
  - trail_atr=1.5, sl_atr=2.0, tp_atr=4.0 (and every other MomentumConfig
    field) -- reproduced by importing and calling
    core/research_momentum_experiment.py's own build_cfg(), not retyped.
  - years=4.0 -- the SAME window core/research_momentum_experiment.py used.
  - The chronological 70/30 per-symbol split, cost/slippage/position-sizing
    defaults, and simulator -- all unchanged.

THE ONLY THING THAT CHANGES: the universe. core/momentum_experiment_
results.csv / _trades.csv (the ORIGINAL 37-symbol/4y baseline) are READ for
comparison, never re-run or overwritten -- there is no need to re-run them
since nothing about that config/window changed.

Universe: the SAME 111-symbol union defined in
core/research_hypothesis1_robustness.py (event_study.UNIVERSE union
market_context.SYMBOL_SECTOR keys) -- reused, not redefined, so this
experiment's "wider universe" is identical to section 14's, not a new
ad hoc list.

Run: python core/research_universe_breadth_isolation.py
Output:
    - printed report + direct comparison against the existing 37-symbol baseline
    - core/momentum_universe_breadth_results.csv
    - core/momentum_universe_breadth_trades.csv
    - core/momentum_universe_breadth_signal_accounting.csv
"""
from __future__ import annotations
import os
import sys

CORE = os.path.dirname(os.path.abspath(__file__))
if CORE not in sys.path:
    sys.path.insert(0, CORE)

import numpy as np
import pandas as pd

from market_context import build_market_context
from momentum import LOOKBACK_BARS
from strategies.momentum_strategy import (
    build_universe_frames,
    load_benchmark_close,
    compute_momentum_panels,
    eligibility_mask,
    cross_sectional_rank,
    select_candidates,
    generate_symbol_detections,
    run_symbol_backtest,
    summarize_trades_only,
)

# Reuse the FROZEN baseline config builder (not reimplemented) and the SAME
# 111-symbol universe section 14 already defined and ran with.
import research_momentum_experiment as baseline_runner
from research_hypothesis1_robustness import ROBUSTNESS_UNIVERSE, _BASELINE_ARGS

CACHE_DIR = os.path.join(CORE, ".momentum_cache")
YEARS = 4.0  # the ORIGINAL baseline's window -- held fixed on purpose
TRAIN_FRAC = 0.7

ORIGINAL_RESULTS_PATH = os.path.join(CORE, "momentum_experiment_results.csv")
ORIGINAL_TRADES_PATH = os.path.join(CORE, "momentum_experiment_trades.csv")


def run_wide_universe_baseline():
    bench = load_benchmark_close(YEARS, cache_dir=CACHE_DIR)
    summary_rows = []
    all_trades = []
    all_accounting = []

    for label in LOOKBACK_BARS:
        cfg = baseline_runner.build_cfg(label, _BASELINE_ARGS)  # trail_atr=1.5, untouched
        print(f"\n{'='*70}\n{label} -- {len(ROBUSTNESS_UNIVERSE)}-symbol universe, years={YEARS} "
              f"(trail_atr={cfg.trail_atr}, unchanged)\n{'='*70}")

        frames = build_universe_frames(ROBUSTNESS_UNIVERSE, YEARS, cache_dir=CACHE_DIR)
        abs_df, rel_df = compute_momentum_panels(frames, bench, cfg.lookback_bars)
        eligible = eligibility_mask(abs_df, rel_df, cfg)
        rank_df = cross_sectional_rank(rel_df, eligible)
        selected = select_candidates(rank_df, eligible, cfg)

        trades_this_lookback = []
        for sym, (df_ohlcv, ind_df) in frames.items():
            if sym not in selected.columns:
                continue
            try:
                ctx = build_market_context(sym, df_ohlcv["Close"], years=YEARS, cache_dir=CACHE_DIR)
            except Exception:
                ctx = None
            dets = generate_symbol_detections(
                sym, df_ohlcv, ind_df,
                selected_col=selected[sym], abs_mom_col=abs_df[sym], rel_mom_col=rel_df[sym],
                rank_col=rank_df[sym], ctx=ctx, cfg=cfg,
            )
            n_candidates = len(dets)
            result = run_symbol_backtest(sym, df_ohlcv, ind_df, dets, cfg)
            trades = result["trades"]
            n_trades = 0 if trades is None else len(trades)
            n_eos = int((dets["bar_idx"] + 1 >= len(df_ohlcv)).sum()) if n_candidates else 0
            all_accounting.append({
                "lookback": label, "symbol": sym, "n_candidates": n_candidates,
                "n_trades": n_trades, "n_end_of_series": n_eos,
                "n_blocked_or_other": max(0, n_candidates - n_trades - n_eos),
            })
            if n_trades:
                trades = trades.copy()
                trades["symbol"] = sym
                trades["lookback"] = label
                split_bar = int(len(df_ohlcv) * TRAIN_FRAC)
                trades["period"] = np.where(trades["entry_bar"] < split_bar, "train", "valid")
                trades_this_lookback.append(trades)

        trades_all = pd.concat(trades_this_lookback, ignore_index=True) if trades_this_lookback else pd.DataFrame()
        if len(trades_all):
            all_trades.append(trades_all)

        overall = summarize_trades_only(trades_all) if len(trades_all) else {"trades": 0}
        train = summarize_trades_only(trades_all[trades_all["period"] == "train"]) if len(trades_all) else {"trades": 0}
        valid = summarize_trades_only(trades_all[trades_all["period"] == "valid"]) if len(trades_all) else {"trades": 0}
        print(f"symbols loaded: {len(frames)} | OVERALL: {overall}")
        print(f"TRAIN: {train}")
        print(f"VALID: {valid}")

        row = {"lookback": label, "n_symbols": len(frames)}
        row.update({f"overall_{k}": v for k, v in overall.items()})
        row.update({f"train_{k}": v for k, v in train.items()})
        row.update({f"valid_{k}": v for k, v in valid.items()})
        summary_rows.append(row)

    return pd.DataFrame(summary_rows), (pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()), pd.DataFrame(all_accounting)


def main():
    print(f"Universe breadth isolation: FROZEN baseline config (trail_atr=1.5, unchanged), "
          f"FROZEN years={YEARS} (the ORIGINAL baseline's window, unchanged).")
    print(f"ONLY the universe changes: 37 symbols (original) -> {len(ROBUSTNESS_UNIVERSE)} symbols (this run).")

    out_df, trades_df, acc_df = run_wide_universe_baseline()

    out_path = os.path.join(CORE, "momentum_universe_breadth_results.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote summary: {out_path}")
    if len(trades_df):
        trades_path = os.path.join(CORE, "momentum_universe_breadth_trades.csv")
        trades_df.to_csv(trades_path, index=False)
        print(f"Wrote trade log: {trades_path}")
    acc_path = os.path.join(CORE, "momentum_universe_breadth_signal_accounting.csv")
    acc_df.to_csv(acc_path, index=False)
    print(f"Wrote signal accounting: {acc_path}")

    # --- Direct comparison against the EXISTING 37-symbol/4y baseline (read-only) ---
    print("\n" + "=" * 78)
    print("37-symbol (original) vs 111-symbol (this run) -- SAME years=4.0, SAME trail_atr=1.5")
    print("=" * 78)
    if os.path.exists(ORIGINAL_RESULTS_PATH):
        orig = pd.read_csv(ORIGINAL_RESULTS_PATH)
        cols = ["lookback", "overall_trades", "overall_win_rate", "overall_profit_factor",
                "train_profit_factor", "valid_profit_factor", "overall_total_pnl"]
        merged = out_df[[c for c in cols if c in out_df.columns]].merge(
            orig[[c for c in cols if c in orig.columns]], on="lookback", suffixes=("_111sym", "_37sym"))
        print(merged.to_string(index=False))
    else:
        print(f"(original results file not found at {ORIGINAL_RESULTS_PATH})")

    if os.path.exists(ORIGINAL_TRADES_PATH) and len(trades_df):
        orig_trades = pd.read_csv(ORIGINAL_TRADES_PATH)
        # pooled, all lookbacks
        pooled_new = summarize_trades_only(trades_df[trades_df.period == "valid"])
        pooled_orig = summarize_trades_only(orig_trades[orig_trades.period == "valid"])
        print(f"\nPooled VALID, 37-symbol (original): {pooled_orig}")
        print(f"Pooled VALID, 111-symbol (this run): {pooled_new}")

        # concentration check, same metric as section 12.4 / section 14
        def top5_share(df):
            c = df["symbol"].value_counts()
            return c.head(5).sum() / c.sum() if len(c) else float("nan"), c.head(5).to_dict()

        share_orig, top_orig = top5_share(orig_trades)
        share_new, top_new = top5_share(trades_df)
        print(f"\nTop-5 concentration, 37-symbol (original): {share_orig:.1%} -- {top_orig}")
        print(f"Top-5 concentration, 111-symbol (this run): {share_new:.1%} -- {top_new}")


if __name__ == "__main__":
    main()
