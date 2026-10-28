# core/research_universe_size_sensitivity.py
"""
core/MOMENTUM_STRATEGY.md section 16: Universe Size Sensitivity + Proper
Walk-Forward.

RESEARCH QUESTION
------------------
Section 15 isolated universe breadth as a single variable (37 vs 111
symbols, same window, same frozen baseline config) and found a real but
modest, lookback-dependent improvement (pooled valid PF 0.690 -> 0.922,
still <1.0). That experiment only compared the two ENDPOINTS (37 and 111)
on a SINGLE positional 70/30 split. This experiment asks two further
questions section 15 could not answer:

  1. Is the effect of universe breadth monotonic/gradual, or a step
     function -- i.e. does it matter whether the universe is 37, 55, 75,
     95 or 111 symbols, not just "narrow" vs "wide"?
  2. Does whatever pattern is found hold up across MULTIPLE, genuinely
     independent, non-overlapping calendar validation windows (a proper
     walk-forward), rather than resting on the single split used in every
     prior section -- and does it survive the exact methodological error
     section 15 itself found and corrected (filtering trades by calendar
     date against a split whose boundary was computed differently)?

HYPOTHESIS (from section 15's own "next step" and this document's research
trajectory -- not asserted true in advance): widening the universe reduces
trade concentration and may improve validation profit factor, but the
effect will not be uniform across lookbacks (section 15 found 9M/12M
improved while 3M/6M did not), and will not by itself be large enough to
call any lookback/universe-size combination "validated as profitable".

WHAT THIS EXPERIMENT DOES NOT DO
---------------------------------
- Does not modify core/momentum.py or core/strategies/momentum_strategy.py.
  Every calculation (momentum, relative strength, cross-sectional rank,
  technical confirmation, market regime, Signal compilation,
  core.sim.engine.simulate()) is the SAME, unmodified pipeline every prior
  section used.
- Does not tune sl_atr/tp_atr/trail_atr/top_n/rebalance_bars/technical
  confirmation/market regime/ADX/RSI or any other strategy parameter. The
  ONLY variable is universe size (and, independently, lookback -- run
  separately, never combined, per instruction).
- Does not select a "best" universe size or lookback, and does not use
  validation results to choose or adjust the fold boundaries, universe
  sizes, or config -- every one of those was fixed BEFORE any backtest in
  this script was run.

FROZEN CONFIG
-------------
Uses the Hypothesis-1 config (trail_atr=0.0 instead of the original
baseline's 1.5) -- per instruction, "because that is the current research
branch" (section 13). Reproduced by importing and calling
core/research_hypothesis1_no_trailing_stop.py's own build_cfg() (not
retyped), with core/research_hypothesis1_robustness.py's own frozen
_HYP1_ARGS (also imported, not retyped) -- so this script cannot silently
drift from what "Hypothesis 1" means elsewhere in this repo. Every other
field (sl_atr=2.0, tp_atr=4.0, top_n=5, use_relative_momentum=True,
use_technical_confirmation=True, use_market_regime=True, rebalance_bars=21,
use_adx=False) is that config's own default, unchanged.

UNIVERSE CONSTRUCTION (deterministic, nested: 37 subset of 55 subset of 75
subset of 95 subset of 111)
------------------------------------------------------------------------
Base: core/event_study.py::UNIVERSE (37 symbols), alphabetically sorted.
Extra: the 74 symbols already used as the "wide" universe in sections
14/15 (core/research_hypothesis1_robustness.py::ROBUSTNESS_UNIVERSE --
event_study.UNIVERSE union market_context.SYMBOL_SECTOR keys) MINUS the 37
base symbols, alphabetically sorted. This is REUSED, not a newly curated
list, and (like ROBUSTNESS_UNIVERSE) was not built with any reference to
momentum-strategy backtest performance.
Each requested size N in {37, 55, 75, 95, 111} is built as:
    base_37_sorted + extra_sorted[: N - 37]
Alphabetical order of the 74-symbol "extra" list is the only ordering
choice made, and it is fixed (Python's stable sort on ticker strings) --
never re-sampled or re-ordered between runs, so universe_55/75/95 are exact
prefixes of universe_111 and supersets of universe_37 by construction.
Exact deltas: 55-37=18, 75-55=20, 95-75=20, 111-95=16 symbols added at each
step (111-37=74 total "extra" symbols do not divide evenly into 4 equal
steps of the requested {55,75,95} midpoints; the sizes are honored exactly
and the step sizes are reported, not smoothed).

WALK-FORWARD METHODOLOGY (the section-15 fix, generalized)
-------------------------------------------------------------
Section 15 found that "filter trades by calendar date out of a run whose
train/valid boundary was actually set POSITIONALLY" silently pulled trades
that were inside that run's own TRAIN allocation into what was reported as
"validation". This script never does a positional split at all. Instead:

  1. Four validation folds are fixed in advance, as plain calendar-year
     windows, before any backtest in this script is run:
         Fold 1: valid = [2022-01-01, 2023-01-01)
         Fold 2: valid = [2023-01-01, 2024-01-01)
         Fold 3: valid = [2024-01-01, 2025-01-01)
         Fold 4: valid = [2025-01-01, 2026-01-01)
     These four validation windows are mutually exclusive and collectively
     exhaustive over 2022-01-01..2026-01-01 -- no trade's entry date can
     fall in more than one fold's validation window.
  2. For fold k, "train" = every trade with entry_date < fold k's
     valid_start (i.e. an EXPANDING window: fold 2's train also contains
     fold 1's validation-period trades, fold 3's contains folds 1 and 2's,
     etc.). This is standard, deliberate expanding-window walk-forward,
     not an error -- only the VALIDATION windows are required to be
     non-overlapping; nothing requires the train windows across folds to
     be disjoint from each other.
  3. Momentum, relative strength, cross-sectional rank, technical
     confirmation and the market-regime filter are all computed ONCE, the
     same unmodified, purely-causal way core/momentum.py and
     core/strategies/momentum_strategy.py already guarantee (see those
     modules' own "LOOK-AHEAD DISCIPLINE" sections): the value at bar t
     never depends on any bar after t, and cross-sectional rank at date t
     depends only on date t's own row. Because of this, there is nothing
     to "refit" per fold -- splitting the resulting trade log by calendar
     date is a strictly honest post-hoc REPORTING partition, not a second
     pipeline run with truncated inputs. This is the same principle
     core/strategies/momentum_strategy.py::run_momentum_experiment's
     positional 70/30 split already relied on; this script only changes
     HOW the partition boundary is chosen (fixed calendar dates, decided
     before running, instead of a positional cut).
  4. Trades whose entry_date falls on/after 2026-01-01 (the tail of the
     available cached history, beyond fold 4) are excluded from all
     fold-based reporting -- reported separately as an explicit count, not
     silently dropped or silently included in fold 4. This keeps every
     fold a full, equal-length calendar year.
  5. "Pooled validation" = the union of all four folds' (disjoint)
     validation trades, reported once per (universe_size, lookback).

DATA / CACHING
----------------
Reuses core/.momentum_cache (the SAME cache_dir every other momentum
research script uses) at years=20.0 -- the SAME request size section 14/15
already used (core.research_hypothesis1_robustness.ROBUSTNESS_YEARS,
imported, not retyped), which is already fully cached on disk for every
symbol in the 111-symbol universe that loaded successfully in that section
(108/111; the same 3 symbols documented in section 14 as clean 404s --
GANESHCO.NS, LTIM.NS, TATAMOTORS.NS -- are expected to fail again here, and
are skipped, not fabricated, by the existing build_universe_frames()). No
new data source or cache convention is introduced.

Run:  python core/research_universe_size_sensitivity.py
Output:
    - printed report
    - core/universe_size_sensitivity_results.csv   (per universe_size x lookback x fold x scope)
    - core/universe_size_sensitivity_trades.csv    (every individual trade, all sizes/lookbacks)
    - core/universe_size_sensitivity_concentration.csv (per universe_size x lookback)
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

CORE = os.path.dirname(os.path.abspath(__file__))
if CORE not in sys.path:
    sys.path.insert(0, CORE)

import numpy as np
import pandas as pd

from event_study import UNIVERSE
from market_context import build_market_context
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

# Reused, not retyped -- see module docstring.
import research_hypothesis1_no_trailing_stop as hyp1_runner
import research_hypothesis1_robustness as rb
from research_hypothesis1_robustness import ROBUSTNESS_UNIVERSE, ROBUSTNESS_YEARS, _HYP1_ARGS

CACHE_DIR = os.path.join(CORE, ".momentum_cache")
YEARS = ROBUSTNESS_YEARS  # 20.0 -- reuses the already-fully-cached request size

UNIVERSE_TARGET_SIZES = (37, 55, 75, 95, 111)

# ---------------------------------------------------------------------------
# Pre-registered, fixed BEFORE any backtest in this script is run.
# ---------------------------------------------------------------------------
FOLDS: List[dict] = [
    {"fold": 1, "valid_start": pd.Timestamp("2022-01-01"), "valid_end": pd.Timestamp("2023-01-01")},
    {"fold": 2, "valid_start": pd.Timestamp("2023-01-01"), "valid_end": pd.Timestamp("2024-01-01")},
    {"fold": 3, "valid_start": pd.Timestamp("2024-01-01"), "valid_end": pd.Timestamp("2025-01-01")},
    {"fold": 4, "valid_start": pd.Timestamp("2025-01-01"), "valid_end": pd.Timestamp("2026-01-01")},
]


def build_cfg(label: str) -> MomentumConfig:
    """The FROZEN Hypothesis-1 config, reproduced by calling that script's
    own build_cfg() -- see module docstring."""
    return hyp1_runner.build_cfg(label, _HYP1_ARGS)


# ---------------------------------------------------------------------------
# Universe construction -- pure, deterministic, no I/O.
# ---------------------------------------------------------------------------
def build_nested_universes(
    base: List[str], extra_pool: List[str], sizes: Tuple[int, ...]
) -> Dict[int, List[str]]:
    """base (kept whole, sorted) + a fixed-order prefix of extra_pool (sorted,
    with any base members removed) for each requested size. Guarantees
    universe[size_a] is a subset of universe[size_b] whenever size_a <=
    size_b, because every larger universe's list is a strict prefix
    extension of every smaller one's."""
    base_sorted = sorted(set(base))
    extra_sorted = sorted(set(extra_pool) - set(base_sorted))
    out: Dict[int, List[str]] = {}
    for size in sizes:
        n_extra = size - len(base_sorted)
        if n_extra < 0:
            raise ValueError(f"Requested size {size} is smaller than the base universe ({len(base_sorted)})")
        if n_extra > len(extra_sorted):
            raise ValueError(
                f"Requested size {size} needs {n_extra} extra symbols but only {len(extra_sorted)} are available"
            )
        out[size] = base_sorted + extra_sorted[:n_extra]
    return out


UNIVERSE_SIZES = build_nested_universes(UNIVERSE, ROBUSTNESS_UNIVERSE, UNIVERSE_TARGET_SIZES)


# ---------------------------------------------------------------------------
# Walk-forward fold logic -- pure, no I/O.
# ---------------------------------------------------------------------------
def fold_for_date(d) -> Optional[int]:
    """Which fold's (mutually-exclusive) validation window `d` falls in, or
    None if `d` is before the first fold's validation start or on/after the
    last fold's validation end."""
    ts = pd.Timestamp(d)
    for f in FOLDS:
        if f["valid_start"] <= ts < f["valid_end"]:
            return f["fold"]
    return None


def train_trades_for_fold(trades: pd.DataFrame, fold: dict) -> pd.DataFrame:
    """Every trade strictly before this fold's validation start -- an
    EXPANDING window across folds (see module docstring). `trades` must
    have an `entry_date` column."""
    if trades is None or trades.empty:
        return trades
    return trades[trades["entry_date"] < fold["valid_start"]]


def valid_trades_for_fold(trades: pd.DataFrame, fold: dict) -> pd.DataFrame:
    if trades is None or trades.empty:
        return trades
    mask = (trades["entry_date"] >= fold["valid_start"]) & (trades["entry_date"] < fold["valid_end"])
    return trades[mask]


def pooled_valid_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Union of all four folds' validation trades -- safe to concatenate
    directly (not re-filter+concat) because the fold windows are disjoint
    by construction."""
    if trades is None or trades.empty:
        return trades
    return trades[trades["valid_fold"].notna()]


# ---------------------------------------------------------------------------
# Concentration -- pure, given a date x symbol boolean `selected` mask and
# the matching `eligible` mask.
# ---------------------------------------------------------------------------
def measure_concentration(selected: pd.DataFrame, eligible: pd.DataFrame) -> dict:
    if selected is None or selected.empty:
        return {
            "total_selection_slots": 0, "top5_share": float("nan"),
            "top5_symbols_json": "{}", "n_distinct_selected_symbols": 0,
            "avg_eligible_universe_size": float(eligible.sum(axis=1).mean()) if len(eligible) else float("nan"),
        }
    slot_counts = selected.sum(axis=0)
    total_slots = int(slot_counts.sum())
    # Only symbols actually selected at least once -- padding with
    # never-selected symbols (head(5) would otherwise include them whenever
    # fewer than 5 distinct symbols were ever selected) would misreport
    # "top 5" as including symbols with zero selection slots.
    nonzero = slot_counts[slot_counts > 0]
    top5 = nonzero.sort_values(ascending=False).head(5)
    top5_share = float(top5.sum() / total_slots) if total_slots else float("nan")
    return {
        "total_selection_slots": total_slots,
        "top5_share": top5_share,
        "top5_symbols_json": json.dumps({k: int(v) for k, v in top5.to_dict().items()}),
        "n_distinct_selected_symbols": int((slot_counts > 0).sum()),
        "avg_eligible_universe_size": float(eligible.sum(axis=1).mean()) if len(eligible) else float("nan"),
    }


# ---------------------------------------------------------------------------
# One (universe_size, lookback) run: rank/select restricted to that
# universe's symbols, detect, backtest every symbol, tag trades with dates
# and fold membership.
# ---------------------------------------------------------------------------
def run_universe_lookback(
    label: str,
    size: int,
    symbols: List[str],
    frames: Dict[str, tuple],
    ctx_by_symbol: Dict[str, Optional[pd.DataFrame]],
    abs_df_full: pd.DataFrame,
    rel_df_full: pd.DataFrame,
    eligible_full: pd.DataFrame,
    cfg: MomentumConfig,
) -> dict:
    cols = [s for s in symbols if s in frames]
    abs_sub, rel_sub, eligible_sub = abs_df_full[cols], rel_df_full[cols], eligible_full[cols]
    rank_basis = rel_sub if cfg.use_relative_momentum else abs_sub
    rank_df = cross_sectional_rank(rank_basis, eligible_sub)
    selected = select_candidates(rank_df, eligible_sub, cfg)

    all_trades = []
    accounting_rows = []
    for sym in cols:
        df_ohlcv, ind_df = frames[sym]
        dets = generate_symbol_detections(
            sym, df_ohlcv, ind_df,
            selected_col=selected[sym], abs_mom_col=abs_sub[sym], rel_mom_col=rel_sub[sym],
            rank_col=rank_df[sym], ctx=ctx_by_symbol.get(sym), cfg=cfg,
        )
        n_candidates = len(dets)
        result = run_symbol_backtest(sym, df_ohlcv, ind_df, dets, cfg)
        trades = result["trades"]
        n_trades = 0 if trades is None else len(trades)
        n_eos = int((dets["bar_idx"] + 1 >= len(df_ohlcv)).sum()) if n_candidates else 0
        accounting_rows.append({
            "symbol": sym, "n_candidates": n_candidates, "n_trades": n_trades,
            "n_end_of_series": n_eos, "n_blocked_or_other": max(0, n_candidates - n_trades - n_eos),
        })
        if n_trades:
            trades = trades.copy()
            trades["symbol"] = sym
            trades["entry_date"] = df_ohlcv.index[trades["entry_bar"].to_numpy()]
            trades["exit_date"] = df_ohlcv.index[trades["exit_bar"].to_numpy()]
            all_trades.append(trades)

    trades_all = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    if len(trades_all):
        trades_all["valid_fold"] = trades_all["entry_date"].apply(fold_for_date)

    concentration = measure_concentration(selected, eligible_sub)
    concentration.update({
        "n_symbols_requested": len(symbols), "n_symbols_loaded": len(cols),
        "n_candidates_total": sum(r["n_candidates"] for r in accounting_rows),
        "n_trades_total": sum(r["n_trades"] for r in accounting_rows),
        "n_end_of_series_total": sum(r["n_end_of_series"] for r in accounting_rows),
        "n_blocked_or_other_total": sum(r["n_blocked_or_other"] for r in accounting_rows),
    })
    n_cand = concentration["n_candidates_total"]
    concentration["blocked_rate"] = (concentration["n_blocked_or_other_total"] / n_cand) if n_cand else float("nan")

    return {"trades": trades_all, "concentration": concentration, "accounting": accounting_rows}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main():
    print("Universe Size Sensitivity + Proper Walk-Forward (core/MOMENTUM_STRATEGY.md section 16)")
    print(f"Frozen config: Hypothesis 1 (trail_atr=0.0), reproduced via research_hypothesis1_no_trailing_stop.build_cfg()")
    print(f"Universe sizes (target -> actual list length): "
          + ", ".join(f"{s}->{len(UNIVERSE_SIZES[s])}" for s in UNIVERSE_TARGET_SIZES))
    print(f"Lookbacks: {list(LOOKBACK_BARS)} (run independently, never combined)")
    print(f"Data: years={YEARS}, cache_dir={CACHE_DIR}")
    print("Walk-forward folds (fixed BEFORE any backtest in this script ran):")
    for f in FOLDS:
        print(f"  Fold {f['fold']}: valid=[{f['valid_start'].date()}, {f['valid_end'].date()}) "
              f"train=[<all available history>, {f['valid_start'].date()})")

    print("\nLoading the full 111-symbol universe's data ONCE (shared across all universe sizes and lookbacks) ...")
    frames = build_universe_frames(UNIVERSE_SIZES[111], YEARS, cache_dir=CACHE_DIR)
    bench = load_benchmark_close(YEARS, cache_dir=CACHE_DIR)
    print(f"  loaded {len(frames)} / {len(UNIVERSE_SIZES[111])} symbols "
          f"({sorted(set(UNIVERSE_SIZES[111]) - set(frames))} failed to load)")

    print("Building each symbol's market-regime context ONCE (shared across lookbacks/universe sizes) ...")
    ctx_by_symbol = {}
    for sym, (df_ohlcv, _ind) in frames.items():
        try:
            ctx_by_symbol[sym] = build_market_context(sym, df_ohlcv["Close"], years=YEARS, cache_dir=CACHE_DIR)
        except Exception:
            ctx_by_symbol[sym] = None

    results_rows = []
    concentration_rows = []
    all_trades = []

    for label in LOOKBACK_BARS:
        cfg = build_cfg(label)
        print(f"\n{'=' * 78}\nLookback {label} ({cfg.lookback_bars} bars) -- computing momentum panels over "
              f"the full loaded frame set ...\n{'=' * 78}")
        abs_df_full, rel_df_full = compute_momentum_panels(frames, bench, cfg.lookback_bars)
        eligible_full = eligibility_mask(abs_df_full, rel_df_full, cfg)

        for size in UNIVERSE_TARGET_SIZES:
            symbols = UNIVERSE_SIZES[size]
            print(f"\n--- {label} / universe_size={size} (requested {len(symbols)} symbols) ---")
            out = run_universe_lookback(
                label, size, symbols, frames, ctx_by_symbol,
                abs_df_full, rel_df_full, eligible_full, cfg,
            )
            trades = out["trades"]
            if len(trades):
                t = trades.copy()
                t["universe_size"] = size
                t["lookback"] = label
                all_trades.append(t)

            conc = out["concentration"]
            conc_row = {"lookback": label, "universe_size": size}
            conc_row.update(conc)
            concentration_rows.append(conc_row)
            print(f"  loaded {conc['n_symbols_loaded']}/{conc['n_symbols_requested']} symbols | "
                  f"avg eligible pool={conc['avg_eligible_universe_size']:.1f} | "
                  f"top5 share of selection slots={conc['top5_share']:.1%} | "
                  f"distinct selected symbols={conc['n_distinct_selected_symbols']} | "
                  f"candidates={conc['n_candidates_total']} trades={conc['n_trades_total']} "
                  f"blocked_rate={conc['blocked_rate']:.1%}")

            base_row = {"lookback": label, "universe_size": size, "n_symbols_loaded": conc["n_symbols_loaded"]}
            for f in FOLDS:
                train_t = train_trades_for_fold(trades, f)
                valid_t = valid_trades_for_fold(trades, f)
                train_m = rb.summarize_config(train_t, trail_on=False)
                valid_m = rb.summarize_config(valid_t, trail_on=False)
                r_row = dict(base_row)
                r_row.update({
                    "fold": f["fold"], "fold_valid_start": f["valid_start"].date().isoformat(),
                    "fold_valid_end": f["valid_end"].date().isoformat(), "scope": "train",
                })
                r_row.update(train_m)
                results_rows.append(r_row)

                v_row = dict(base_row)
                v_row.update({
                    "fold": f["fold"], "fold_valid_start": f["valid_start"].date().isoformat(),
                    "fold_valid_end": f["valid_end"].date().isoformat(), "scope": "valid",
                })
                v_row.update(valid_m)
                results_rows.append(v_row)
                print(f"    Fold {f['fold']} [{f['valid_start'].date()}, {f['valid_end'].date()}): "
                      f"train_trades={train_m.get('trades')} train_pf={train_m.get('profit_factor')} | "
                      f"valid_trades={valid_m.get('trades')} valid_pf={valid_m.get('profit_factor')}")

            pooled = pooled_valid_trades(trades)
            pooled_m = rb.summarize_config(pooled, trail_on=False)
            p_row = dict(base_row)
            p_row.update({
                "fold": "POOLED", "fold_valid_start": FOLDS[0]["valid_start"].date().isoformat(),
                "fold_valid_end": FOLDS[-1]["valid_end"].date().isoformat(), "scope": "valid",
            })
            p_row.update(pooled_m)
            results_rows.append(p_row)

            # Trades before fold 1's valid_start are "pre-period" (they still
            # count as TRAIN for every fold, per the expanding-window design)
            # -- only trades on/after fold 4's valid_end are excluded from
            # all fold-based reporting (see module docstring, point 4).
            n_excluded = int((trades["entry_date"] >= FOLDS[-1]["valid_end"]).sum()) if len(trades) else 0
            print(f"    Pooled VALID (all 4 folds): trades={pooled_m.get('trades')} pf={pooled_m.get('profit_factor')} "
                  f"| trades excluded (entry >= {FOLDS[-1]['valid_end'].date()}, outside all folds): {n_excluded}")

    results_df = pd.DataFrame(results_rows)
    results_path = os.path.join(CORE, "universe_size_sensitivity_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nWrote results: {results_path} ({len(results_df)} rows)")

    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    trades_path = os.path.join(CORE, "universe_size_sensitivity_trades.csv")
    trades_df.to_csv(trades_path, index=False)
    print(f"Wrote trade log: {trades_path} ({len(trades_df)} rows)")

    concentration_df = pd.DataFrame(concentration_rows)
    concentration_path = os.path.join(CORE, "universe_size_sensitivity_concentration.csv")
    concentration_df.to_csv(concentration_path, index=False)
    print(f"Wrote concentration report: {concentration_path} ({len(concentration_df)} rows)")

    print("\n" + "=" * 78)
    print("POOLED VALIDATION PROFIT FACTOR, by lookback x universe_size (evidence only -- no")
    print("configuration is being selected or ranked as 'best').")
    print("=" * 78)
    pooled_view = results_df[(results_df["fold"] == "POOLED")]
    cols = ["lookback", "universe_size", "trades", "win_rate", "profit_factor", "expectancy", "total_pnl"]
    present = [c for c in cols if c in pooled_view.columns]
    print(pooled_view[present].to_string(index=False))


if __name__ == "__main__":
    main()
