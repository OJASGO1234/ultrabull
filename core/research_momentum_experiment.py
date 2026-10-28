# core/research_momentum_experiment.py
"""
Phase-7 research runner for the momentum strategy: backtest the SAME
strategy independently at 3M / 6M / 9M / 12M lookbacks over the shared
UNIVERSE used throughout the rest of the audit suite (core/event_study.py),
and report metrics side by side.

This deliberately does NOT pick a winner. Per the research brief: collect
evidence at each lookback, report train vs out-of-sample (chronological
70/30 split, same convention as core/audit_walkforward_market_condition_diagnosis.py),
and let a human decide what (if anything) is worth pursuing further.

Run (from the repo root or from core/):
    python core/research_momentum_experiment.py
    python core/research_momentum_experiment.py --years 4 --top-n 5
    python core/research_momentum_experiment.py --selection top_percentile --top-percentile 0.15

Output:
    - printed report per lookback
    - core/momentum_experiment_results.csv (one row per lookback, all metrics)
    - core/momentum_experiment_trades.csv (every simulated trade, all lookbacks, with a `lookback` column)
"""
from __future__ import annotations
import argparse
import os
import sys

CORE = os.path.dirname(os.path.abspath(__file__))
if CORE not in sys.path:
    sys.path.insert(0, CORE)

import pandas as pd

from event_study import UNIVERSE
from momentum import LOOKBACK_BARS
from strategies.momentum_strategy import MomentumConfig, run_momentum_experiment

CACHE_DIR = os.path.join(CORE, ".momentum_cache")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--years", type=float, default=4.0, help="History window per symbol (default: 4.0, matches the rest of the audit suite)")
    p.add_argument("--selection", choices=["top_n", "top_percentile"], default="top_n")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--top-percentile", type=float, default=0.1)
    p.add_argument("--no-technical", action="store_true", help="Disable technical confirmation (MA20/MA50/ADX)")
    p.add_argument("--no-market-regime", action="store_true", help="Disable the NIFTY regime filter")
    p.add_argument("--no-relative", action="store_true", help="Disable the relative-momentum requirement/ranking (rank by absolute momentum instead)")
    p.add_argument("--use-adx", action="store_true", help="Also require ADX >= --adx-min")
    p.add_argument("--adx-min", type=float, default=20.0)
    p.add_argument("--rebalance-bars", type=int, default=21)
    p.add_argument("--symbols", type=str, default=None, help="Comma-separated override of the UNIVERSE (for a quick smoke test)")
    return p.parse_args()


def build_cfg(label: str, args) -> MomentumConfig:
    return MomentumConfig(
        lookback_label=label,
        use_relative_momentum=not args.no_relative,
        selection_method=args.selection,
        top_n=args.top_n,
        top_percentile=args.top_percentile,
        use_technical_confirmation=not args.no_technical,
        use_market_regime=not args.no_market_regime,
        use_adx=args.use_adx,
        adx_min=args.adx_min,
        rebalance_bars=args.rebalance_bars,
    )


def main():
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else UNIVERSE

    print(f"Universe: {len(symbols)} symbols, years={args.years}")
    print(f"Selection: {args.selection} (top_n={args.top_n}, top_percentile={args.top_percentile})")
    print(f"Relative momentum: {not args.no_relative} | Technical confirmation: {not args.no_technical} "
          f"| Market regime: {not args.no_market_regime} | ADX filter: {args.use_adx} (min={args.adx_min})")
    print(f"Rebalance cooldown: {args.rebalance_bars} bars")

    summary_rows = []
    all_trades = []

    for label in LOOKBACK_BARS:
        cfg = build_cfg(label, args)
        print(f"\n{'=' * 70}\nMomentum lookback {label} ({cfg.lookback_bars} bars)\n{'=' * 70}")
        report = run_momentum_experiment(symbols, args.years, cfg, cache_dir=CACHE_DIR)

        overall, train, valid = report["overall"], report["train"], report["valid"]
        print(f"symbols loaded: {report['n_symbols']} | symbols with >=1 trade: {report['n_symbols_with_trades']}")
        print(f"OVERALL: {overall}")
        print(f"TRAIN:   {train}")
        print(f"VALID:   {valid}")

        trades_all = report["trades_all"]
        if len(trades_all):
            trades_all = trades_all.copy()
            trades_all["lookback"] = label
            all_trades.append(trades_all)

        row = {"lookback": label, "lookback_bars": cfg.lookback_bars,
               "n_symbols": report["n_symbols"], "n_symbols_with_trades": report["n_symbols_with_trades"]}
        row.update({f"overall_{k}": v for k, v in overall.items()})
        row.update({f"train_{k}": v for k, v in train.items()})
        row.update({f"valid_{k}": v for k, v in valid.items()})
        summary_rows.append(row)

    out_df = pd.DataFrame(summary_rows)
    out_path = os.path.join(CORE, "momentum_experiment_results.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote summary: {out_path}")

    if all_trades:
        trades_df = pd.concat(all_trades, ignore_index=True)
        trades_path = os.path.join(CORE, "momentum_experiment_trades.csv")
        trades_df.to_csv(trades_path, index=False)
        print(f"Wrote trade log: {trades_path}")
    else:
        print("\nNo trades were generated at any lookback with this configuration -- "
              "see the per-lookback report above. This is a real (not fabricated) result; "
              "it may mean the filters are too strict for this universe/period, not that "
              "the implementation is broken (see core/strategies/test_momentum_strategy.py "
              "for coverage that the plumbing itself works on synthetic data).")

    print("\n" + "=" * 70)
    print("COMPARISON (lookback x key metrics) -- no winner is being selected;")
    print("this is evidence for a human to evaluate.")
    print("=" * 70)
    cols = ["lookback", "n_symbols_with_trades", "overall_trades", "overall_win_rate",
            "overall_profit_factor", "overall_expectancy", "overall_total_pnl",
            "valid_trades", "valid_win_rate", "valid_profit_factor", "valid_expectancy"]
    present = [c for c in cols if c in out_df.columns]
    print(out_df[present].to_string(index=False))


if __name__ == "__main__":
    main()
