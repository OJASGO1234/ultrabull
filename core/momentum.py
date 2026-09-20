# core/momentum.py
"""
Absolute and relative price momentum over multi-month lookback windows.

This is the shared math layer for the daily momentum / relative-strength
strategy (see core/strategies/momentum_strategy.py for the cross-sectional
research pipeline built on top of it, and api/main.py for the per-symbol API
surface). It intentionally does not duplicate any indicator that already
exists elsewhere in the codebase (MA20/MA50/ADX/ATR/RSI all still come from
core/vbt_to_pd_ind.py); this module only adds the multi-month momentum and
relative-strength calculations that nothing else in the repo provides.

Formula (Jegadeesh-Titman style, the standard used in the 3/6/9/12-month
Indian relative-strength research this strategy is based on):

    momentum[t]          = close[t] / close[t - lookback_bars] - 1
    relative_momentum[t] = momentum[t] - benchmark_momentum[t]

Lookbacks are expressed in TRADING DAYS (bars), matching how the rest of the
pipeline (patterns, gates, the simulator) addresses time -- by positional bar
index, not calendar days. TRADING_DAYS_PER_MONTH=21 is the conventional
approximation used to turn "3/6/9/12 months" into a bar count.

LOOK-AHEAD DISCIPLINE
----------------------
Every value at position t is derived strictly from close[t] and
close[t - lookback_bars] (pandas .shift(), which can only look backward).
Rows before `lookback_bars` history exists are NaN, never a fabricated
signal. relative_momentum aligns the benchmark to the stock's own index with
a forward-fill that only ever carries the last KNOWN (i.e. past) benchmark
value forward across a missing session -- it never reaches into the future.
See core/test_momentum.py for a regression test that mutates only the tail
of a series and asserts momentum at an earlier date is unchanged.
"""
from __future__ import annotations
from typing import Dict, Optional

import numpy as np
import pandas as pd

TRADING_DAYS_PER_MONTH = 21

# The four lookback periods the mentor asked to compare. Not a recommendation
# of which one "works" -- see core/research_momentum_experiment.py, which
# backtests all four independently and reports them side by side.
LOOKBACK_MONTHS: Dict[str, int] = {"3M": 3, "6M": 6, "9M": 9, "12M": 12}

LOOKBACK_BARS: Dict[str, int] = {
    label: months * TRADING_DAYS_PER_MONTH for label, months in LOOKBACK_MONTHS.items()
}  # {"3M": 63, "6M": 126, "9M": 189, "12M": 252}


def absolute_momentum(close: pd.Series, lookback_bars: int) -> pd.Series:
    """Own-history return over `lookback_bars` trading days.

    Returns NaN wherever fewer than `lookback_bars` prior observations exist
    (via `.shift`), or where the base price is zero/undefined -- an
    insufficient-history bar is never silently treated as zero momentum.
    """
    if lookback_bars <= 0:
        raise ValueError("lookback_bars must be a positive integer")
    base = close.shift(lookback_bars)
    mom = close / base - 1.0
    return mom.replace([np.inf, -np.inf], np.nan)


def relative_momentum(
    stock_close: pd.Series,
    benchmark_close: pd.Series,
    lookback_bars: int,
) -> pd.Series:
    """Excess momentum vs. a benchmark (default usage: NIFTY, ^NSEI) --
    relative_momentum[t] = stock_momentum[t] - benchmark_momentum[t].

    This is intentionally a SEPARATE calculation from market_context.py's
    rs_20/rs_60 (20/60-bar excess return vs. the mapped SECTOR index, used by
    the existing breakout RS gate). Those windows are too short for 3-12
    month relative-strength research and the benchmark there is
    sector-first, index-fallback; this function is benchmark-explicit and
    horizon-explicit so the two are never confused.
    """
    bench_aligned = benchmark_close.reindex(stock_close.index).ffill()
    stock_mom = absolute_momentum(stock_close, lookback_bars)
    bench_mom = absolute_momentum(bench_aligned, lookback_bars)
    return stock_mom - bench_mom


def compute_multi_period_momentum(
    stock_close: pd.Series,
    benchmark_close: Optional[pd.Series] = None,
    lookbacks: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """Absolute (+ relative, if a benchmark is given) momentum for every
    lookback in `lookbacks` (default: the 3M/6M/9M/12M set above), kept as
    separate columns -- momentum_3M, momentum_6M, ..., relative_momentum_3M,
    ... -- rather than collapsed into a single score, so the periods stay
    independently comparable."""
    lookbacks = lookbacks or LOOKBACK_BARS
    out = pd.DataFrame(index=stock_close.index)
    for label, bars in lookbacks.items():
        out[f"momentum_{label}"] = absolute_momentum(stock_close, bars)
        if benchmark_close is not None:
            out[f"relative_momentum_{label}"] = relative_momentum(stock_close, benchmark_close, bars)
    return out


def technical_confirmation(
    close: pd.Series,
    ind_df: pd.DataFrame,
    *,
    require_close_above_ma20: bool = True,
    require_close_above_ma50: bool = True,
    require_ma20_above_ma50: bool = True,
    adx_min: float = 0.0,
) -> pd.DataFrame:
    """Row-wise technical-confirmation flags built ONLY from indicators that
    already exist in ind_df (MA20/MA50/ADX, from
    core/vbt_to_pd_ind.py::build_ind_df_from_vbt) -- no indicator is
    recomputed here. Each condition is independently switchable and every
    threshold (adx_min) is a parameter, not a baked-in assumption; a
    disabled/zeroed condition is treated as always-true so it drops out of
    the AND without special-casing callers.

    Returns a DataFrame indexed like ind_df with boolean columns
    close_above_ma20, close_above_ma50, ma20_above_ma50, adx_ok, and their
    AND as `confirmed`.
    """
    idx = ind_df.index
    out = pd.DataFrame(index=idx)
    out["close_above_ma20"] = (close > ind_df["MA20"]) if require_close_above_ma20 else True
    out["close_above_ma50"] = (close > ind_df["MA50"]) if require_close_above_ma50 else True
    out["ma20_above_ma50"] = (ind_df["MA20"] > ind_df["MA50"]) if require_ma20_above_ma50 else True
    out["adx_ok"] = (ind_df["ADX"] >= adx_min) if adx_min > 0 else True
    out["confirmed"] = (
        out["close_above_ma20"].astype(bool)
        & out["close_above_ma50"].astype(bool)
        & out["ma20_above_ma50"].astype(bool)
        & out["adx_ok"].astype(bool)
    )
    return out
