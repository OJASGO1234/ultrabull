# core/test_momentum.py
"""
Tests for core/momentum.py: absolute/relative momentum math and technical
confirmation. Pure synthetic data, no network calls.

Run with pytest: pytest core/test_momentum.py
Run standalone:  python core/test_momentum.py   (falls back to manually
calling every test_* function if pytest isn't installed, same convention as
core/sim/test_engine_regressions.py).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from momentum import (
    LOOKBACK_BARS,
    absolute_momentum,
    relative_momentum,
    compute_multi_period_momentum,
    technical_confirmation,
)


def _dates(n):
    return pd.date_range("2020-01-01", periods=n, freq="B")


# ---------------------------------------------------------------------------
# 1. Momentum calculation correctness
# ---------------------------------------------------------------------------
def test_absolute_momentum_basic():
    n = 100
    close = pd.Series(np.linspace(100.0, 200.0, n), index=_dates(n))  # steady climb
    mom = absolute_momentum(close, lookback_bars=20)
    # close[t] / close[t-20] - 1, checked directly at one interior point
    t = 50
    expected = close.iloc[t] / close.iloc[t - 20] - 1.0
    assert abs(mom.iloc[t] - expected) < 1e-12


def test_absolute_momentum_doubling():
    n = 30
    close = pd.Series([100.0] * 10 + [200.0] * 20, index=_dates(n))
    mom = absolute_momentum(close, lookback_bars=10)
    # at bar 20, close=200, close[10]=200 (still in the flat 200 stretch) -> mom=0
    assert abs(mom.iloc[20] - 0.0) < 1e-9
    # at bar 10 (first bar of the jump), close=200 vs close[0]=100 -> mom=1.0
    assert abs(mom.iloc[10] - 1.0) < 1e-9


def test_absolute_momentum_rejects_nonpositive_lookback():
    close = pd.Series(np.arange(10.0), index=_dates(10))
    try:
        absolute_momentum(close, lookback_bars=0)
        raised = False
    except ValueError:
        raised = True
    assert raised


# ---------------------------------------------------------------------------
# 2. Insufficient-history handling
# ---------------------------------------------------------------------------
def test_absolute_momentum_insufficient_history_is_nan():
    n = 15
    close = pd.Series(np.linspace(100, 110, n), index=_dates(n))
    mom = absolute_momentum(close, lookback_bars=20)  # more history required than exists
    assert mom.isna().all()


def test_absolute_momentum_nan_only_before_lookback():
    n = 50
    close = pd.Series(np.linspace(100, 150, n), index=_dates(n))
    lookback = 21
    mom = absolute_momentum(close, lookback_bars=lookback)
    assert mom.iloc[:lookback].isna().all()
    assert mom.iloc[lookback:].notna().all()


# ---------------------------------------------------------------------------
# 3. Relative momentum
# ---------------------------------------------------------------------------
def test_relative_momentum_matches_difference():
    n = 80
    idx = _dates(n)
    stock = pd.Series(np.linspace(100, 180, n), index=idx)
    bench = pd.Series(np.linspace(100, 130, n), index=idx)
    lookback = 30
    rel = relative_momentum(stock, bench, lookback)
    expected = absolute_momentum(stock, lookback) - absolute_momentum(bench, lookback)
    pd.testing.assert_series_equal(rel, expected, check_names=False)


def test_relative_momentum_positive_when_stock_outperforms():
    n = 60
    idx = _dates(n)
    stock = pd.Series(np.linspace(100, 200, n), index=idx)   # +100%
    bench = pd.Series(np.linspace(100, 110, n), index=idx)   # +10%
    rel = relative_momentum(stock, bench, lookback_bars=40)
    assert rel.iloc[-1] > 0.3  # clearly outperforming


def test_relative_momentum_no_lookahead():
    """Mutating the tail of either series must not change relative momentum
    at an earlier date -- the defining no-lookahead property."""
    n = 100
    idx = _dates(n)
    rng = np.random.default_rng(42)
    stock = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=idx)
    bench = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.008, n)), index=idx)
    lookback = 20
    t = 60

    rel_before = relative_momentum(stock, bench, lookback).iloc[t]

    stock2 = stock.copy()
    bench2 = bench.copy()
    stock2.iloc[t + 1:] = stock2.iloc[t + 1:] * 5.0   # blow up the future
    bench2.iloc[t + 1:] = bench2.iloc[t + 1:] * 0.1

    rel_after = relative_momentum(stock2, bench2, lookback).iloc[t]
    assert abs(rel_before - rel_after) < 1e-9


# ---------------------------------------------------------------------------
# 4. Multi-period momentum keeps periods separate
# ---------------------------------------------------------------------------
def test_multi_period_momentum_columns_independent():
    n = 300
    idx = _dates(n)
    rng = np.random.default_rng(7)
    stock = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=idx)
    bench = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.008, n)), index=idx)
    out = compute_multi_period_momentum(stock, bench)
    for label in LOOKBACK_BARS:
        assert f"momentum_{label}" in out.columns
        assert f"relative_momentum_{label}" in out.columns
    # different lookbacks should generally give different values at the same date
    row = out.iloc[-1]
    assert row["momentum_3M"] != row["momentum_12M"]


# ---------------------------------------------------------------------------
# 5. Missing-data handling
# ---------------------------------------------------------------------------
def test_absolute_momentum_handles_nan_gap():
    n = 60
    close = pd.Series(np.linspace(100, 160, n), index=_dates(n))
    close.iloc[30] = np.nan  # a single missing bar (e.g. a data glitch)
    mom = absolute_momentum(close, lookback_bars=20)
    # momentum computed off the NaN base/close should itself be NaN, not a
    # fabricated number, at the bars that touch it
    assert pd.isna(mom.iloc[30])
    assert pd.isna(mom.iloc[50])  # close[50] is fine but base close[30] is NaN


# ---------------------------------------------------------------------------
# 6. Technical confirmation
# ---------------------------------------------------------------------------
def test_technical_confirmation_all_pass():
    idx = _dates(5)
    ind_df = pd.DataFrame({
        "MA20": [10, 10, 10, 10, 10],
        "MA50": [9, 9, 9, 9, 9],
        "ADX": [25, 25, 25, 25, 25],
    }, index=idx)
    close = pd.Series([11, 11, 11, 11, 11], index=idx)
    out = technical_confirmation(close, ind_df, adx_min=20.0)
    assert out["confirmed"].all()


def test_technical_confirmation_fails_when_close_below_ma20():
    idx = _dates(3)
    ind_df = pd.DataFrame({"MA20": [10, 10, 10], "MA50": [9, 9, 9], "ADX": [25, 25, 25]}, index=idx)
    close = pd.Series([5, 5, 5], index=idx)  # below MA20
    out = technical_confirmation(close, ind_df)
    assert not out["confirmed"].any()


def test_technical_confirmation_disabled_conditions_pass_through():
    idx = _dates(3)
    ind_df = pd.DataFrame({"MA20": [10, 10, 10], "MA50": [9, 9, 9], "ADX": [5, 5, 5]}, index=idx)
    close = pd.Series([5, 5, 5], index=idx)  # would fail close>MA20 if enabled
    out = technical_confirmation(
        close, ind_df,
        require_close_above_ma20=False,
        require_close_above_ma50=False,
        require_ma20_above_ma50=True,
        adx_min=0.0,  # disabled -> always true regardless of low ADX
    )
    assert out["confirmed"].all()


if __name__ == "__main__":
    import sys
    try:
        import pytest
        raise SystemExit(pytest.main([__file__, "-v"]))
    except ImportError:
        g = dict(globals())
        tests = [(k, v) for k, v in g.items() if k.startswith("test_") and callable(v)]
        failed = 0
        for name, fn in tests:
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                failed += 1
                print(f"FAIL {name}: {e}")
        print(f"\n{len(tests) - failed}/{len(tests)} passed")
        sys.exit(1 if failed else 0)
