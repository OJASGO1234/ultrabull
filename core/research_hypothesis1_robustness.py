# core/research_hypothesis1_robustness.py
"""
Hypothesis-1 robustness check (core/MOMENTUM_STRATEGY.md section 13's own "Next
validation step"): retest the SAME FROZEN configurations -- the original
baseline (trail_atr=1.5) and Hypothesis 1 (trail_atr=0.0) -- on a longer
history and a substantially larger universe, to see whether Hypothesis 1's
improvement survives a bigger sample. This is a robustness/generalization
check, NOT a parameter search: no field of either config is changed from
what core/research_momentum_experiment.py and
core/research_hypothesis1_no_trailing_stop.py already run with no CLI
flags. Both frozen configs are reproduced by IMPORTING and CALLING those
two scripts' own `build_cfg()` functions (not retyped/duplicated), so
there is no risk of this script silently drifting from what "baseline" and
"Hypothesis 1" actually mean.

UNIVERSE (reused, not newly curated): the union of the two NSE symbol lists
that ALREADY exist in this repository --
  - core/event_study.py::UNIVERSE (37 symbols; hand-picked for the
    pattern-strategy event study)
  - core/market_context.py::SYMBOL_SECTOR (108 symbols; hand-picked for
    sector-relative-strength mapping)
Neither list was built with any reference to the momentum strategy's
backtest performance -- one predates this feature entirely, the other
exists for a different, unrelated purpose (sector regime lookups) -- so
their union (111 symbols, printed in the run log) is not a
performance-based cherry-pick with respect to momentum results. This is
also the largest universe-like structure already in the repo (see
core/MOMENTUM_STRATEGY.md's Explore-agent survey referenced in the
project's session history); no new list was invented.

HISTORY: requests years=20 (ROBUSTNESS_YEARS below) per symbol from
yfinance, which simply returns however much history actually exists from
each symbol's listing date forward (core/stock_explorer.py::load_data sets
`start = end - 365*years` and yfinance clips to what's available -- it does
not error on a request that predates listing). Actual per-symbol bar counts
and date ranges achieved are reported in the run log, not assumed to all be
20 years.

METHODOLOGY (unchanged from both frozen runs): core/strategies/
momentum_strategy.py's cross-sectional pipeline (momentum -> relative
strength -> rank -> select -> technical/regime confirm -> Signal ->
core.sim.engine.simulate()), the same chronological 70/30 per-symbol
positional train/valid split, the same cost/slippage/position-sizing
defaults (core.strategies.momentum_strategy.default_sim_cfg).

KEY IMPLEMENTATION NOTE: generate_symbol_detections() does not read
trail_atr/sl_atr/tp_atr at all -- only the exit/risk step
(compile_momentum_signals -> core.sim.engine.simulate()) does. So for a
given lookback, the CANDIDATE detections are identical between the baseline
and Hypothesis-1 configs; this script computes them ONCE per (symbol,
lookback) and backtests both configs against that same candidate set,
which makes the "did trail_atr change which signals got taken" comparison
exact rather than approximate, and roughly halves the compute cost.

Run:  python core/research_hypothesis1_robustness.py
Output:
    - printed report
    - core/momentum_robustness_results.csv       (per lookback x config summary)
    - core/momentum_robustness_trades.csv        (every trade, both configs, all lookbacks)
    - core/momentum_robustness_signal_accounting.csv (candidate vs executed counts, per symbol x lookback x config)
Neither the original baseline files (core/momentum_experiment_*.csv) nor
Hypothesis 1's files (core/momentum_hypothesis1_no_trailing_stop_*.csv) are
read for writing, or overwritten -- these are new, separate outputs.
"""
from __future__ import annotations
import os
import sys
import types

CORE = os.path.dirname(os.path.abspath(__file__))
if CORE not in sys.path:
    sys.path.insert(0, CORE)

import numpy as np
import pandas as pd

from event_study import UNIVERSE
from market_context import SYMBOL_SECTOR, build_market_context
from momentum import LOOKBACK_BARS
from strategies.momentum_strategy import (
    MomentumConfig,
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

# Reuse the two FROZEN experiment scripts' own build_cfg() functions --
# these are not reimplemented here, so this script cannot silently drift
# from what "baseline" / "Hypothesis 1" mean elsewhere in the repo.
import research_momentum_experiment as baseline_runner
import research_hypothesis1_no_trailing_stop as hyp1_runner

CACHE_DIR = os.path.join(CORE, ".momentum_cache")  # same cache dir; cache keys already include `years`, so no collision
TRAIN_FRAC = 0.7
ROBUSTNESS_YEARS = 20.0  # "as far back as available" -- see module docstring; actual coverage reported per symbol

ROBUSTNESS_UNIVERSE = sorted(set(UNIVERSE) | set(SYMBOL_SECTOR.keys()))

# Frozen config args -- these namespaces reproduce EXACTLY what running
# `python core/research_momentum_experiment.py` and
# `python core/research_hypothesis1_no_trailing_stop.py` with NO CLI flags
# does (i.e. every default in each script's own argparse definition).
_BASELINE_ARGS = types.SimpleNamespace(
    no_relative=False, selection="top_n", top_n=5, top_percentile=0.1,
    no_technical=False, no_market_regime=False, use_adx=False, adx_min=20.0,
    rebalance_bars=21,
)
_HYP1_ARGS = types.SimpleNamespace(top_n=5)


def build_shared_detections(universe: list[str], years: float, cache_dir: str):
    """Per lookback: load data once, build the momentum/rank/selection
    panels ONCE (identical for both configs -- see module docstring), and
    generate each symbol's candidate detections ONCE. Returns
    {label: {"frames": {...}, "dets": {symbol: DataFrame}}}."""
    out = {}
    bench = load_benchmark_close(years, cache_dir=cache_dir)
    for label, bars in LOOKBACK_BARS.items():
        print(f"  loading/preparing {label} ({bars} bars) ...")
        cfg_shared = MomentumConfig(lookback_label=label)  # only the selection-layer fields matter here
        frames = build_universe_frames(universe, years, cache_dir=cache_dir)
        abs_df, rel_df = compute_momentum_panels(frames, bench, cfg_shared.lookback_bars)
        eligible = eligibility_mask(abs_df, rel_df, cfg_shared)
        rank_df = cross_sectional_rank(rel_df, eligible)
        selected = select_candidates(rank_df, eligible, cfg_shared)

        dets_by_symbol = {}
        for sym, (df_ohlcv, ind_df) in frames.items():
            if sym not in selected.columns:
                continue
            try:
                ctx = build_market_context(sym, df_ohlcv["Close"], years=years, cache_dir=cache_dir)
            except Exception:
                ctx = None
            dets = generate_symbol_detections(
                sym, df_ohlcv, ind_df,
                selected_col=selected[sym], abs_mom_col=abs_df[sym], rel_mom_col=rel_df[sym],
                rank_col=rank_df[sym], ctx=ctx, cfg=cfg_shared,
            )
            dets_by_symbol[sym] = dets
        out[label] = {"frames": frames, "dets": dets_by_symbol}
    return out


def run_frozen_config(label: str, shared: dict, cfg: MomentumConfig, config_name: str) -> dict:
    """Backtest ONE frozen config against the SHARED candidate detections
    for this lookback. Returns trades_all (with period/symbol/lookback/
    config columns), per-symbol metrics, and signal-accounting rows."""
    frames = shared["frames"]
    dets_by_symbol = shared["dets"]

    all_trades = []
    per_symbol_metrics = {}
    accounting_rows = []

    for sym, dets in dets_by_symbol.items():
        df_ohlcv, ind_df = frames[sym]
        n_candidates = len(dets)

        result = run_symbol_backtest(sym, df_ohlcv, ind_df, dets, cfg)
        trades = result["trades"]
        n_trades = 0 if trades is None else len(trades)
        per_symbol_metrics[sym] = result["metrics"]

        # How many candidates could never have filled regardless of
        # position-blocking, because their (delayed) entry bar falls past
        # the end of the available series -- a real, boring edge effect,
        # not the phenomenon under study, and worth separating out.
        n_end_of_series = int((dets["bar_idx"] + 1 >= len(df_ohlcv)).sum()) if n_candidates else 0
        n_unexplained_gap = max(0, n_candidates - n_trades - n_end_of_series)
        accounting_rows.append({
            "config": config_name, "lookback": label, "symbol": sym,
            "n_candidates": n_candidates, "n_trades": n_trades,
            "n_end_of_series": n_end_of_series,
            "n_blocked_or_other": n_unexplained_gap,
        })

        if n_trades:
            trades = trades.copy()
            trades["symbol"] = sym
            trades["lookback"] = label
            trades["config"] = config_name
            split_bar = int(len(df_ohlcv) * TRAIN_FRAC)
            trades["period"] = np.where(trades["entry_bar"] < split_bar, "train", "valid")
            all_trades.append(trades)

    trades_all = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    return {"trades_all": trades_all, "per_symbol_metrics": per_symbol_metrics, "accounting": accounting_rows}


def _exit_class(row, trail_on: bool) -> str:
    if row["reason"] in ("TP", "TP_GAP"):
        return "TP"
    if row["reason"] in ("SL", "SL_GAP"):
        if trail_on and row["pnl"] > 0:
            return "TRAIL_PROFIT"
        return "HARD_SL"
    return row["reason"]


def summarize_config(trades_all: pd.DataFrame, trail_on: bool) -> dict:
    if trades_all is None or trades_all.empty:
        return {"trades": 0}
    t = trades_all.copy()
    t["hold"] = t["exit_bar"] - t["entry_bar"]
    t["exit_class"] = t.apply(lambda r: _exit_class(r, trail_on), axis=1)
    tp_rate = (t["exit_class"] == "TP").mean()
    wins = t["pnl"] > 0
    avg_win = t.loc[wins, "pnl"].mean() if wins.any() else 0.0
    avg_loss = t.loc[~wins, "pnl"].mean() if (~wins).any() else 0.0
    realized_rr = (avg_win / abs(avg_loss)) if avg_loss else float("nan")
    base = summarize_trades_only(t)
    base.update({
        "tp_hit_rate": float(tp_rate),
        "realized_rr": float(realized_rr),
        "mean_hold_bars": float(t["hold"].mean()),
        "median_hold_bars": float(t["hold"].median()),
        "max_hold_bars": float(t["hold"].max()),
    })
    return base


def main():
    print(f"Universe: {len(ROBUSTNESS_UNIVERSE)} symbols "
          f"(event_study.UNIVERSE + market_context.SYMBOL_SECTOR keys, union)")
    print(f"Requested history: {ROBUSTNESS_YEARS} years (actual per-symbol coverage may be less; see below)")
    print("Frozen configs under test: A) baseline (trail_atr=1.5)  B) Hypothesis 1 (trail_atr=0.0)")
    print("No parameter is being searched; both configs are reproduced via the existing scripts' own build_cfg().")

    print("\nBuilding shared candidate detections (data load + ranking, once per lookback, "
          "shared by both configs) ...")
    shared_by_lookback = build_shared_detections(ROBUSTNESS_UNIVERSE, ROBUSTNESS_YEARS, CACHE_DIR)

    # Report actual data coverage achieved, since 20y was REQUESTED, not guaranteed.
    print("\nActual per-symbol history coverage (first lookback's frame set is representative -- "
          "same raw OHLCV load is shared across lookbacks):")
    any_label = next(iter(shared_by_lookback))
    frames0 = shared_by_lookback[any_label]["frames"]
    bars_per_symbol = {s: len(df) for s, (df, _i) in frames0.items()}
    years_per_symbol = {s: n / 252.0 for s, n in bars_per_symbol.items()}
    print(f"  symbols loaded: {len(frames0)} / {len(ROBUSTNESS_UNIVERSE)}")
    print(f"  bars per symbol: min={min(bars_per_symbol.values())}, "
          f"median={int(np.median(list(bars_per_symbol.values())))}, max={max(bars_per_symbol.values())}")
    print(f"  approx years per symbol: min={min(years_per_symbol.values()):.1f}, "
          f"median={np.median(list(years_per_symbol.values())):.1f}, max={max(years_per_symbol.values()):.1f}")
    shortest = sorted(years_per_symbol.items(), key=lambda kv: kv[1])[:5]
    print(f"  shortest-history symbols: {shortest}")

    all_trades_everything = []
    all_accounting = []
    summary_rows = []

    for label in LOOKBACK_BARS:
        shared = shared_by_lookback[label]
        for config_name, args_ns, runner in [
            ("baseline", _BASELINE_ARGS, baseline_runner),
            ("hypothesis1", _HYP1_ARGS, hyp1_runner),
        ]:
            cfg = runner.build_cfg(label, args_ns)
            print(f"\n{'='*70}\n{label} / {config_name} (trail_atr={cfg.trail_atr}, "
                  f"sl_atr={cfg.sl_atr}, tp_atr={cfg.tp_atr})\n{'='*70}")
            res = run_frozen_config(label, shared, cfg, config_name)
            trades_all = res["trades_all"]
            all_accounting.extend(res["accounting"])
            if len(trades_all):
                all_trades_everything.append(trades_all)

            trail_on = cfg.trail_atr > 0
            overall = summarize_config(trades_all, trail_on)
            train = summarize_config(trades_all[trades_all["period"] == "train"] if len(trades_all) else trades_all, trail_on)
            valid = summarize_config(trades_all[trades_all["period"] == "valid"] if len(trades_all) else trades_all, trail_on)

            dds = [m.get("max_drawdown") for m in res["per_symbol_metrics"].values() if m.get("trades", 0) > 0]
            dds = [d for d in dds if d is not None and np.isfinite(d)]
            mean_dd = float(np.mean(dds)) if dds else float("nan")
            worst_dd = float(np.min(dds)) if dds else float("nan")

            n_cand = sum(r["n_candidates"] for r in res["accounting"])
            n_tr = sum(r["n_trades"] for r in res["accounting"])
            n_eos = sum(r["n_end_of_series"] for r in res["accounting"])
            n_blocked = sum(r["n_blocked_or_other"] for r in res["accounting"])
            blocked_rate = n_blocked / n_cand if n_cand else float("nan")

            print(f"OVERALL: {overall}")
            print(f"TRAIN:   {train}")
            print(f"VALID:   {valid}")
            print(f"Per-symbol equity max_drawdown: mean={mean_dd:.3%} worst={worst_dd:.3%} "
                  f"(n={len(dds)} symbols with >=1 trade; NOT a unified portfolio curve -- "
                  f"core.sim.engine.simulate() is single-symbol)")
            print(f"Signal accounting: {n_cand} candidate signals -> {n_tr} trades executed "
                  f"({n_eos} unreachable at series end, {n_blocked} blocked-or-other "
                  f"[{blocked_rate:.1%} of candidates])")

            row = {"lookback": label, "config": config_name, "trail_atr": cfg.trail_atr,
                   "n_candidates": n_cand, "n_trades": n_tr, "n_end_of_series": n_eos,
                   "n_blocked_or_other": n_blocked, "blocked_rate": blocked_rate,
                   "mean_max_drawdown": mean_dd, "worst_max_drawdown": worst_dd}
            row.update({f"overall_{k}": v for k, v in overall.items()})
            row.update({f"train_{k}": v for k, v in train.items()})
            row.update({f"valid_{k}": v for k, v in valid.items()})
            summary_rows.append(row)

    out_df = pd.DataFrame(summary_rows)
    out_path = os.path.join(CORE, "momentum_robustness_results.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote summary: {out_path}")

    trades_df = pd.concat(all_trades_everything, ignore_index=True) if all_trades_everything else pd.DataFrame()
    if len(trades_df):
        trades_path = os.path.join(CORE, "momentum_robustness_trades.csv")
        trades_df.to_csv(trades_path, index=False)
        print(f"Wrote trade log: {trades_path}")

    acc_df = pd.DataFrame(all_accounting)
    acc_path = os.path.join(CORE, "momentum_robustness_signal_accounting.csv")
    acc_df.to_csv(acc_path, index=False)
    print(f"Wrote signal accounting: {acc_path}")

    print("\n" + "=" * 78)
    print("BASELINE vs HYPOTHESIS 1, SAME (larger) universe/history -- pooled across all 4 lookbacks")
    print("=" * 78)
    for config_name in ("baseline", "hypothesis1"):
        # Pool trades across lookbacks for a single pooled PF/win-rate, same
        # convention as core/MOMENTUM_STRATEGY.md's pooled tables.
        pooled = trades_df[trades_df["config"] == config_name] if len(trades_df) else pd.DataFrame()
        for period in ("train", "valid"):
            p = pooled[pooled["period"] == period] if len(pooled) else pooled
            s = summarize_trades_only(p) if len(p) else {"trades": 0}
            print(f"{config_name:12s} {period:5s}: trades={s.get('trades')} "
                  f"win_rate={s.get('win_rate')} profit_factor={s.get('profit_factor')} "
                  f"total_pnl={s.get('total_pnl')}")

    cols = ["lookback", "config", "trail_atr", "overall_trades", "overall_win_rate",
            "overall_profit_factor", "train_profit_factor", "valid_profit_factor",
            "overall_tp_hit_rate", "overall_realized_rr", "overall_median_hold_bars",
            "blocked_rate"]
    print(out_df[[c for c in cols if c in out_df.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
