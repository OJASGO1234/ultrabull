# core/research_hypothesis1_no_trailing_stop.py
"""
Hypothesis Experiment 1 (core/MOMENTUM_STRATEGY.md §12.9, first bullet):

    "The trailing stop (trail_atr=1.5) may be too tight relative to the
    target (tp_atr=4.0) for this holding-period/volatility combination;
    ... removing it and relying on the fixed target/stop ... would change
    how much of the designed 2.0 reward:risk is actually realized."

Diagnostic evidence that motivated this (core/diagnose_momentum_experiment.py,
§12.1): only ~10% of trades ever reach the full take-profit; ~25-29% exit
via the trailing stop while still profitable (cut short of the target);
realized avg_win/avg_loss is ~1.4-1.5 against a designed 2.0.

THIS IS A SEPARATE, ADDITIVE EXPERIMENT. It does not modify
core/momentum.py, core/strategies/momentum_strategy.py, the baseline
core/research_momentum_experiment.py, or any of their output files
(core/momentum_experiment_results.csv / core/momentum_experiment_trades.csv
are left untouched). The ONLY thing that differs from the baseline run is
one field on MomentumConfig: trail_atr=0.0 instead of 1.5. Everything else
-- momentum calculation, relative-strength ranking, technical confirmation,
market regime filter, position sizing, slippage/fees, the simulator, and
the chronological 70/30 train/valid split -- is identical to the baseline,
because this script imports and calls the exact same
core.strategies.momentum_strategy.run_momentum_experiment() the baseline
uses. trail_atr=0.0 requires no code change anywhere: core/sim/engine.py's
trailing-stop logic is already conditioned on `trail_k > 0`, so setting it
to 0 already fully disables trailing (see
core/sim/engine.py:151/199 and the new regression test in
core/strategies/test_momentum_strategy.py::test_trail_atr_zero_lets_winner_reach_full_target).

Run:  python core/research_hypothesis1_no_trailing_stop.py
Output:
    - printed report per lookback + a direct baseline comparison table
    - core/momentum_hypothesis1_no_trailing_stop_results.csv
    - core/momentum_hypothesis1_no_trailing_stop_trades.csv
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

CACHE_DIR = os.path.join(CORE, ".momentum_cache")  # SAME cache the baseline run populated -- no new network calls
BASELINE_RESULTS_PATH = os.path.join(CORE, "momentum_experiment_results.csv")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--years", type=float, default=4.0, help="Must match the baseline run for a fair comparison")
    p.add_argument("--top-n", type=int, default=5, help="Must match the baseline run for a fair comparison")
    p.add_argument("--symbols", type=str, default=None, help="Comma-separated override of the UNIVERSE (debugging only)")
    return p.parse_args()


def build_cfg(label: str, args) -> MomentumConfig:
    """Every field except trail_atr is left at MomentumConfig's default --
    i.e. identical to the baseline experiment's config (see
    core/research_momentum_experiment.py::build_cfg with no CLI flags
    passed). trail_atr=0.0 is the single, pre-registered change this
    experiment tests."""
    return MomentumConfig(lookback_label=label, top_n=args.top_n, trail_atr=0.0)


def _load_baseline() -> pd.DataFrame | None:
    if not os.path.exists(BASELINE_RESULTS_PATH):
        print(f"\n(No baseline file at {BASELINE_RESULTS_PATH} -- run core/research_momentum_experiment.py "
              f"first for a side-by-side comparison. Continuing without it.)")
        return None
    return pd.read_csv(BASELINE_RESULTS_PATH)


def main():
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else UNIVERSE

    print("Hypothesis Experiment 1: remove the trailing stop (trail_atr: 1.5 -> 0.0), "
          "everything else identical to the baseline default config.")
    print(f"Universe: {len(symbols)} symbols, years={args.years}, top_n={args.top_n}")

    baseline_df = _load_baseline()

    summary_rows = []
    all_trades = []
    selection_diffs = []

    for label in LOOKBACK_BARS:
        cfg = build_cfg(label, args)
        print(f"\n{'=' * 70}\nMomentum lookback {label} ({cfg.lookback_bars} bars) -- trail_atr={cfg.trail_atr}\n{'=' * 70}")
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

            # NOTE (found by running this, not assumed in advance): trail_atr
            # does NOT change candidate-signal generation (generate_symbol_
            # detections has no dependency on it) -- but it DOES change how
            # long a position stays open, and core.sim.engine.simulate()
            # allows only max_open_positions=1 per symbol. A trailing stop
            # that exits quickly frees the slot for the NEXT candidate signal
            # sooner; removing it lets winners (and losers riding the wide
            # fixed stop) run much longer, so a later signal for the same
            # symbol can arrive while the slot is still occupied and never
            # become a trade at all. This check reports how much that
            # (indirect, execution-level) effect actually changed the traded
            # set -- it is NOT expected to be empty/identical, and finding
            # out it isn't is itself a real result (see MOMENTUM_STRATEGY.md
            # section 13).
            baseline_trades_path = os.path.join(CORE, "momentum_experiment_trades.csv")
            if os.path.exists(baseline_trades_path):
                base_t = pd.read_csv(baseline_trades_path)
                base_t = base_t[base_t["lookback"] == label]
                base_keys = set(zip(base_t["symbol"], base_t["entry_bar"]))
                hyp_keys = set(zip(trades_all["symbol"], trades_all["entry_bar"]))
                selection_diffs.append((label, base_keys == hyp_keys, len(base_keys), len(hyp_keys)))

        row = {"lookback": label, "lookback_bars": cfg.lookback_bars,
               "n_symbols": report["n_symbols"], "n_symbols_with_trades": report["n_symbols_with_trades"]}
        row.update({f"overall_{k}": v for k, v in overall.items()})
        row.update({f"train_{k}": v for k, v in train.items()})
        row.update({f"valid_{k}": v for k, v in valid.items()})
        summary_rows.append(row)

    out_df = pd.DataFrame(summary_rows)
    out_path = os.path.join(CORE, "momentum_hypothesis1_no_trailing_stop_results.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote summary: {out_path}")

    if all_trades:
        trades_df = pd.concat(all_trades, ignore_index=True)
        trades_path = os.path.join(CORE, "momentum_hypothesis1_no_trailing_stop_trades.csv")
        trades_df.to_csv(trades_path, index=False)
        print(f"Wrote trade log: {trades_path}")

    if selection_diffs:
        print("\n" + "=" * 70)
        print("TRADED-SET CHECK: candidate-signal generation is unaffected by trail_atr, but the")
        print("REALIZED trade set can still differ via the single-open-position-per-symbol constraint")
        print("(a slot occupied longer by a non-trailing position blocks later candidate signals).")
        print("=" * 70)
        for label, identical, n_base, n_hyp in selection_diffs:
            print(f"  {label}: baseline had {n_base} trades, hypothesis-1 has {n_hyp} trades, "
                  f"same (symbol, entry_bar) set = {identical}")

    print("\n" + "=" * 70)
    print("BASELINE vs HYPOTHESIS 1 (no trailing stop) -- side by side, no winner is being")
    print("declared; this is one pre-registered test, not a parameter search.")
    print("=" * 70)
    cols = ["lookback", "overall_trades", "overall_win_rate", "overall_profit_factor",
            "overall_expectancy", "overall_total_pnl",
            "train_profit_factor", "valid_profit_factor",
            "train_win_rate", "valid_win_rate"]
    present = [c for c in cols if c in out_df.columns]
    if baseline_df is not None:
        merged = out_df[present].merge(
            baseline_df[[c for c in cols if c in baseline_df.columns]],
            on="lookback", suffixes=("_hyp1", "_baseline"),
        )
        print(merged.to_string(index=False))
    else:
        print(out_df[present].to_string(index=False))


if __name__ == "__main__":
    main()
