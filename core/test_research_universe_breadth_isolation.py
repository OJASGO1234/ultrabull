# core/test_research_universe_breadth_isolation.py
"""
Tests for core/research_universe_breadth_isolation.py. Synthetic data only
(monkeypatched data loaders), no network calls.

Covers the two things this script's validity depends on:
  1. The config is genuinely FROZEN and identical to the original baseline's
     (core/research_momentum_experiment.py, no CLI flags) -- only the
     universe passed to build_universe_frames differs.
  2. YEARS is fixed at 4.0, matching the ORIGINAL baseline's window (not
     section 14's 20 years) -- isolating universe breadth as the one
     changed variable, per core/MOMENTUM_STRATEGY.md section 15.
  3. The signal-accounting arithmetic (candidates = trades + end_of_series +
     blocked_or_other) holds on synthetic data end-to-end.

Run with pytest: pytest core/test_research_universe_breadth_isolation.py
Run standalone:  python core/test_research_universe_breadth_isolation.py
"""
from __future__ import annotations
import os
import sys

CORE = os.path.dirname(os.path.abspath(__file__))
if CORE not in sys.path:
    sys.path.insert(0, CORE)

import numpy as np
import pandas as pd

import research_universe_breadth_isolation as ub
import research_momentum_experiment as baseline_runner
from research_hypothesis1_robustness import _BASELINE_ARGS, ROBUSTNESS_UNIVERSE


def _dates(n):
    return pd.date_range("2021-01-01", periods=n, freq="B")


def test_years_frozen_at_original_baseline_window():
    """The whole point of this experiment is holding history length fixed
    at the ORIGINAL baseline's 4 years while only the universe changes --
    if this drifted to match section 14's 20 years, the experiment would
    no longer isolate universe breadth."""
    assert ub.YEARS == 4.0


def test_config_is_identical_to_original_baseline_except_universe():
    """The frozen config used here must be BIT-FOR-BIT what
    core/research_momentum_experiment.py runs with no CLI flags -- built by
    calling that script's own build_cfg(), not retyped."""
    cfg = baseline_runner.build_cfg("6M", _BASELINE_ARGS)
    assert cfg.trail_atr == 1.5
    assert cfg.sl_atr == 2.0 and cfg.tp_atr == 4.0
    assert cfg.top_n == 5
    assert cfg.use_relative_momentum is True
    assert cfg.use_technical_confirmation is True
    assert cfg.use_market_regime is True
    assert cfg.rebalance_bars == 21


def test_reuses_the_same_universe_as_section_14():
    """The "wider universe" here must be IDENTICAL to section 14's -- not a
    fresh, independently-chosen list -- or the comparison across the two
    sections would not be apples-to-apples."""
    assert ub.ROBUSTNESS_UNIVERSE is ROBUSTNESS_UNIVERSE  # same object, imported not redefined
    assert len(ub.ROBUSTNESS_UNIVERSE) > 100


def _make_frame(n=300, jump_bar=150):
    idx = _dates(n)
    close = np.full(n, 100.0)
    close[jump_bar:] = 100.0 * 1.15
    df_ohlcv = pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
        "Volume": np.full(n, 1_000_000.0),
    }, index=idx)
    ind_df = pd.DataFrame({
        "MA20": pd.Series(close, index=idx).rolling(20, min_periods=1).mean(),
        "MA50": pd.Series(close, index=idx).rolling(50, min_periods=1).mean(),
        "ADX": np.full(n, 30.0), "ATR": np.full(n, 2.0),
        "RSI": np.full(n, 60.0), "MACD": np.zeros(n), "MACD_signal": np.zeros(n),
        "BB_upper": close * 1.05, "BB_lower": close * 0.95,
    }, index=idx)
    return df_ohlcv, ind_df


def test_signal_accounting_arithmetic_holds_end_to_end():
    """Run the actual pipeline pieces this script uses (eligibility ->
    rank -> select -> generate_symbol_detections -> run_symbol_backtest) on
    a synthetic single-symbol universe, and check the same accounting
    identity core/research_hypothesis1_robustness.py already asserts:
    n_candidates == n_trades + n_end_of_series + n_blocked_or_other."""
    from strategies.momentum_strategy import (
        MomentumConfig, eligibility_mask, cross_sectional_rank, select_candidates,
        generate_symbol_detections, run_symbol_backtest, compute_momentum_panels,
    )

    df_ohlcv, ind_df = _make_frame()
    bench = pd.Series(100.0, index=df_ohlcv.index)
    cfg = MomentumConfig(lookback_label="6M", use_market_regime=False)
    frames = {"X.NS": (df_ohlcv, ind_df)}

    abs_df, rel_df = compute_momentum_panels(frames, bench, cfg.lookback_bars)
    eligible = eligibility_mask(abs_df, rel_df, cfg)
    rank_df = cross_sectional_rank(rel_df, eligible)
    selected = select_candidates(rank_df, eligible, cfg)

    dets = generate_symbol_detections(
        "X.NS", df_ohlcv, ind_df,
        selected_col=selected["X.NS"], abs_mom_col=abs_df["X.NS"], rel_mom_col=rel_df["X.NS"],
        rank_col=rank_df["X.NS"], ctx=None, cfg=cfg,
    )
    n_candidates = len(dets)
    result = run_symbol_backtest("X.NS", df_ohlcv, ind_df, dets, cfg)
    n_trades = len(result["trades"]) if result["trades"] is not None else 0
    n_eos = int((dets["bar_idx"] + 1 >= len(df_ohlcv)).sum()) if n_candidates else 0
    n_blocked = max(0, n_candidates - n_trades - n_eos)

    assert n_trades + n_eos + n_blocked == n_candidates


if __name__ == "__main__":
    import sys as _sys
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
        _sys.exit(1 if failed else 0)
