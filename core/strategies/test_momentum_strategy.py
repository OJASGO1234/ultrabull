# core/strategies/test_momentum_strategy.py
"""
Tests for core/strategies/momentum_strategy.py: cross-sectional ranking,
selection, signal generation and integration with the EXISTING simulator
(core.sim.engine.simulate). All synthetic data -- no network calls, so these
run offline and deterministically.

Run with pytest: pytest core/strategies/test_momentum_strategy.py
Run standalone:  python core/strategies/test_momentum_strategy.py
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd

# core/strategies/ -> core/ on sys.path, matching api/main.py's convention so
# this file's `from stock_explorer import ...`-style absolute imports (made
# inside strategies/momentum_strategy.py) resolve whether this test is run
# via pytest, `python core/strategies/test_momentum_strategy.py`, or from
# inside core/.
_CORE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)

import strategies.momentum_strategy as ms
from strategies.momentum_strategy import (
    MomentumConfig,
    eligibility_mask,
    cross_sectional_rank,
    select_candidates,
    generate_symbol_detections,
    compile_momentum_signals,
    run_symbol_backtest,
    default_sim_cfg,
    screen_universe_now,
)


def _dates(n):
    return pd.date_range("2021-01-01", periods=n, freq="B")


# ---------------------------------------------------------------------------
# Cross-sectional ranking
# ---------------------------------------------------------------------------
def test_cross_sectional_rank_orders_by_momentum():
    idx = _dates(3)
    mom = pd.DataFrame({
        "A": [0.10, 0.20, 0.05],
        "B": [0.30, 0.10, 0.40],
        "C": [0.05, 0.05, np.nan],
    }, index=idx)
    eligible = pd.DataFrame(True, index=idx, columns=mom.columns)
    eligible.loc[idx[2], "C"] = False
    ranks = cross_sectional_rank(mom, eligible)
    # date 0: B(0.30) > A(0.10) > C(0.05) -> ranks 1,2,3
    assert ranks.loc[idx[0], "B"] == 1
    assert ranks.loc[idx[0], "A"] == 2
    assert ranks.loc[idx[0], "C"] == 3
    # date 2: C is ineligible -> NaN rank, not competing with A/B
    assert pd.isna(ranks.loc[idx[2], "C"])
    assert ranks.loc[idx[2], "B"] == 1  # B(0.40) outranks A(0.05)
    assert ranks.loc[idx[2], "A"] == 2


def test_cross_sectional_rank_strict_ordering():
    idx = _dates(1)
    mom = pd.DataFrame({"A": [0.05], "B": [0.40], "C": [0.10]}, index=idx)
    eligible = pd.DataFrame(True, index=idx, columns=mom.columns)
    ranks = cross_sectional_rank(mom, eligible)
    assert ranks.loc[idx[0], "B"] == 1
    assert ranks.loc[idx[0], "C"] == 2
    assert ranks.loc[idx[0], "A"] == 3


def test_cross_sectional_rank_no_lookahead():
    """Rank at date t must depend only on date t's cross-section -- mutating
    a LATER date's momentum values must not change an earlier date's rank."""
    idx = _dates(5)
    mom = pd.DataFrame({
        "A": [0.10, 0.20, 0.05, 0.01, 0.01],
        "B": [0.30, 0.10, 0.40, 0.01, 0.01],
    }, index=idx)
    eligible = pd.DataFrame(True, index=idx, columns=mom.columns)
    ranks_before = cross_sectional_rank(mom, eligible)

    mom2 = mom.copy()
    mom2.iloc[3:, :] = mom2.iloc[3:, :] * 1000.0  # blow up the future rows

    ranks_after = cross_sectional_rank(mom2, eligible)
    pd.testing.assert_series_equal(ranks_before.iloc[0], ranks_after.iloc[0], check_names=False)
    pd.testing.assert_series_equal(ranks_before.iloc[1], ranks_after.iloc[1], check_names=False)
    pd.testing.assert_series_equal(ranks_before.iloc[2], ranks_after.iloc[2], check_names=False)


# ---------------------------------------------------------------------------
# Eligibility / thresholds (configuration changes)
# ---------------------------------------------------------------------------
def test_eligibility_mask_absolute_threshold():
    idx = _dates(2)
    abs_df = pd.DataFrame({"A": [0.02, 0.10], "B": [-0.01, 0.30]}, index=idx)
    rel_df = pd.DataFrame({"A": [0.02, 0.10], "B": [-0.01, 0.30]}, index=idx)
    cfg = MomentumConfig(use_absolute_momentum=True, min_absolute_momentum=0.0, use_relative_momentum=False)
    mask = eligibility_mask(abs_df, rel_df, cfg)
    assert mask.loc[idx[0], "A"]        # 0.02 >= 0.0
    assert not mask.loc[idx[0], "B"]    # -0.01 < 0.0


def test_eligibility_mask_excludes_missing_history():
    idx = _dates(2)
    abs_df = pd.DataFrame({"A": [np.nan, 0.10]}, index=idx)
    rel_df = pd.DataFrame({"A": [np.nan, 0.10]}, index=idx)
    cfg = MomentumConfig()
    mask = eligibility_mask(abs_df, rel_df, cfg)
    assert not mask.loc[idx[0], "A"]  # NaN momentum -> never eligible
    assert mask.loc[idx[1], "A"]


# ---------------------------------------------------------------------------
# Selection (top_n vs top_percentile) -- configuration changes
# ---------------------------------------------------------------------------
def test_select_top_n():
    idx = _dates(1)
    ranks = pd.DataFrame({"A": [1], "B": [2], "C": [3], "D": [4]}, index=idx)
    eligible = pd.DataFrame(True, index=idx, columns=ranks.columns)
    cfg = MomentumConfig(selection_method="top_n", top_n=2)
    sel = select_candidates(ranks, eligible, cfg)
    assert sel.loc[idx[0], "A"] and sel.loc[idx[0], "B"]
    assert not sel.loc[idx[0], "C"] and not sel.loc[idx[0], "D"]


def test_select_top_percentile():
    idx = _dates(1)
    ranks = pd.DataFrame({c: [i + 1] for i, c in enumerate("ABCDEFGHIJ")}, index=idx)  # ranks 1..10
    eligible = pd.DataFrame(True, index=idx, columns=ranks.columns)
    cfg = MomentumConfig(selection_method="top_percentile", top_percentile=0.2)  # top 20% of 10 = 2
    sel = select_candidates(ranks, eligible, cfg)
    assert sel.loc[idx[0]].sum() == 2
    assert sel.loc[idx[0], "A"] and sel.loc[idx[0], "B"]


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------
def _make_symbol_data(n=120, uptrend_start=60):
    idx = _dates(n)
    close = np.full(n, 100.0)
    close[uptrend_start:] = 100.0 * (1.02 ** np.arange(n - uptrend_start))  # sharp uptrend
    df_ohlcv = pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
        "Volume": np.full(n, 1_000_000.0),
    }, index=idx)
    ma20 = pd.Series(close, index=idx).rolling(20, min_periods=1).mean()
    ma50 = pd.Series(close, index=idx).rolling(50, min_periods=1).mean()
    ind_df = pd.DataFrame({
        "MA20": ma20, "MA50": ma50,
        "ADX": np.full(n, 30.0),
        "ATR": np.full(n, 2.0),
        "RSI": np.full(n, 60.0),
        "MACD": np.zeros(n), "MACD_signal": np.zeros(n),
        "BB_upper": close * 1.05, "BB_lower": close * 0.95,
    }, index=idx)
    return df_ohlcv, ind_df


def test_generate_symbol_detections_fires_once_per_rebalance():
    df_ohlcv, ind_df = _make_symbol_data(n=150, uptrend_start=60)
    idx = df_ohlcv.index
    # "selected" true for a long stretch (as if this symbol sat in the top N
    # for months) -- detections should still only fire sparsely.
    selected = pd.Series(False, index=idx)
    selected.iloc[65:140] = True
    abs_mom = pd.Series(0.5, index=idx)
    rel_mom = pd.Series(0.3, index=idx)
    rank = pd.Series(1.0, index=idx)
    cfg = MomentumConfig(rebalance_bars=21, use_market_regime=False, use_technical_confirmation=True)

    dets = generate_symbol_detections("TEST.NS", df_ohlcv, ind_df, selected, abs_mom, rel_mom, rank, ctx=None, cfg=cfg)
    assert len(dets) >= 1
    # consecutive fires must be spaced by at least rebalance_bars
    bars = dets["bar_idx"].to_numpy()
    assert np.all(np.diff(bars) >= cfg.rebalance_bars)
    # first fire should be right where `selected` first turns True (bar 65),
    # since technical confirmation is already satisfied by the uptrend there
    assert bars[0] >= 65


def test_generate_symbol_detections_missing_data_handling():
    """A symbol whose momentum/rank columns don't fully cover df_ohlcv's date
    range (as happens after an outer-join across an uneven universe) must not
    crash -- missing entries are reindexed to False/NaN and excluded."""
    df_ohlcv, ind_df = _make_symbol_data(n=80, uptrend_start=40)
    idx = df_ohlcv.index
    # selected series covering only PART of the date range
    partial_idx = idx[10:50]
    selected = pd.Series(True, index=partial_idx)
    abs_mom = pd.Series(0.2, index=partial_idx)
    rel_mom = pd.Series(0.1, index=partial_idx)
    rank = pd.Series(1.0, index=partial_idx)
    cfg = MomentumConfig(use_market_regime=False, rebalance_bars=5)

    dets = generate_symbol_detections("TEST.NS", df_ohlcv, ind_df, selected, abs_mom, rel_mom, rank, ctx=None, cfg=cfg)
    # must not raise, and must not fire outside the covered range
    if len(dets):
        assert dets["bar_idx"].min() >= 10
        assert dets["bar_idx"].max() < 50


def test_compile_momentum_signals_shape():
    dets = pd.DataFrame([
        {"bar_idx": 10, "date": pd.Timestamp("2021-01-01"), "symbol": "X", "pattern": "Momentum_6M",
         "lookback": "6M", "absolute_momentum": 0.3, "relative_momentum": 0.2, "rank": 1.0,
         "technical_confirmed": True, "market_regime_ok": True},
    ])
    cfg = MomentumConfig(lookback_label="6M", sl_atr=2.0, tp_atr=4.0, trail_atr=1.5)
    signals = compile_momentum_signals(dets, cfg)
    assert len(signals) == 1
    row = signals.iloc[0]
    assert row["bar_idx"] == 10
    assert row["side"] == "long"
    assert row["sl_atr"] == 2.0 and row["tp_atr"] == 4.0 and row["trail_atr"] == 1.5
    assert row["tag"] == "Momentum_6M"
    assert row["enter_at"] == "open" and row["delay_bars"] == 1
    assert row["meta"]["relative_momentum"] == 0.2


def test_compile_momentum_signals_empty_detections():
    cfg = MomentumConfig()
    signals = compile_momentum_signals(pd.DataFrame(), cfg)
    assert len(signals) == 0
    assert list(signals.columns) == ["bar_idx", "side", "sl_atr", "tp_atr", "trail_atr", "tag",
                                      "size_cash", "enter_at", "delay_bars", "priority", "meta"]


# ---------------------------------------------------------------------------
# Integration with the EXISTING simulator (core.sim.engine.simulate)
# ---------------------------------------------------------------------------
def test_integration_with_existing_simulator_produces_trade():
    df_ohlcv, ind_df = _make_symbol_data(n=150, uptrend_start=60)
    dets = pd.DataFrame([{
        "bar_idx": 60, "date": df_ohlcv.index[60], "symbol": "TEST.NS", "pattern": "Momentum_6M",
        "lookback": "6M", "absolute_momentum": 0.5, "relative_momentum": 0.3, "rank": 1.0,
        "technical_confirmed": True, "market_regime_ok": True,
    }])
    cfg = MomentumConfig(lookback_label="6M", sl_atr=2.0, tp_atr=4.0, trail_atr=0.0, max_hold_bars=0)
    # Override the default warmup cutoff (which is sized for a REAL 6M
    # lookback, ~131 bars) since this test fires a signal at bar 60 purely
    # to exercise signal-compile -> simulate() plumbing, not warmup logic.
    sim_cfg = default_sim_cfg(cfg)
    sim_cfg["filters"]["warmup_cutoff_bars"] = 0
    result = run_symbol_backtest("TEST.NS", df_ohlcv, ind_df, dets, cfg, sim_cfg=sim_cfg)

    trades = result["trades"]
    assert len(trades) >= 1
    t0 = trades.iloc[0]
    # signal fired at bar_idx=60, delay_bars=1 -> entry_bar must be 61, never
    # 60 or earlier (this IS core.sim.engine.simulate's own no-lookahead
    # entry timing, exercised here through the momentum strategy's signals)
    assert t0["entry_bar"] == 61
    assert t0["side"] == "long"
    assert t0["tag"] == "Momentum_6M"
    # metrics were computed via the shared summarize_trades_only/_equity_metrics helpers
    assert result["metrics"]["trades"] == len(trades)
    assert "max_drawdown" in result["metrics"]


def test_config_change_technical_confirmation_reduces_signals():
    """Toggling use_technical_confirmation off must not DECREASE the
    candidate set relative to it being on (on is a strict AND filter) --
    demonstrates the confirmation flag is actually load-bearing."""
    df_ohlcv, ind_df = _make_symbol_data(n=150, uptrend_start=60)
    idx = df_ohlcv.index
    # force technical confirmation to fail everywhere by pushing MA20 above close
    ind_df = ind_df.copy()
    ind_df["MA20"] = df_ohlcv["Close"] * 1.5  # close is always "below" MA20

    selected = pd.Series(False, index=idx)
    selected.iloc[65:140] = True
    abs_mom = pd.Series(0.5, index=idx)
    rel_mom = pd.Series(0.3, index=idx)
    rank = pd.Series(1.0, index=idx)

    cfg_on = MomentumConfig(use_technical_confirmation=True, use_market_regime=False, rebalance_bars=21)
    cfg_off = MomentumConfig(use_technical_confirmation=False, use_market_regime=False, rebalance_bars=21)

    dets_on = generate_symbol_detections("T", df_ohlcv, ind_df, selected, abs_mom, rel_mom, rank, None, cfg_on)
    dets_off = generate_symbol_detections("T", df_ohlcv, ind_df, selected, abs_mom, rel_mom, rank, None, cfg_off)

    assert len(dets_on) == 0        # confirmation always fails -> no signals
    assert len(dets_off) > 0        # confirmation disabled -> signals fire


def test_screen_universe_now_respects_technical_confirmation_flag():
    """Regression test for a real bug found while building the paper-trading
    bridge: screen_universe_now originally ignored cfg.use_technical_
    confirmation and always required trend confirmation, unlike
    generate_symbol_detections (the backtest path), which correctly bypasses
    it when the flag is off. A flat-price series never satisfies close>MA20
    (they're equal), so it's a clean case to assert the flag's effect on."""
    df_ohlcv, ind_df = _make_symbol_data(n=150, uptrend_start=200)  # slice past the end -> stays flat throughout
    idx = df_ohlcv.index

    def fake_frames(symbols, years, cache_dir=None):
        return {"X.NS": (df_ohlcv, ind_df)}

    def fake_bench(years, cache_dir=None):
        return pd.Series(100.0, index=idx)  # flat benchmark, matching the flat stock

    orig_frames, orig_bench = ms.build_universe_frames, ms.load_benchmark_close
    ms.build_universe_frames = fake_frames
    ms.load_benchmark_close = fake_bench
    try:
        cfg_on = MomentumConfig(lookback_label="3M", top_n=5, use_technical_confirmation=True, use_market_regime=False)
        cfg_off = MomentumConfig(lookback_label="3M", top_n=5, use_technical_confirmation=False, use_market_regime=False)
        screen_on = screen_universe_now(["X.NS"], 1.0, cfg_on)
        screen_off = screen_universe_now(["X.NS"], 1.0, cfg_off)
    finally:
        ms.build_universe_frames = orig_frames
        ms.load_benchmark_close = orig_bench

    assert len(screen_on["rows"]) == 1 and len(screen_off["rows"]) == 1
    row_on, row_off = screen_on["rows"][0], screen_off["rows"][0]
    assert row_on["technical_confirmed"] is False
    assert row_on["signal"] is None           # withheld: fails the (enabled) trend filter
    assert row_off["technical_confirmed"] is True    # bypassed when the flag is off
    assert row_off["signal"] == "long"


def test_trail_atr_zero_lets_winner_reach_full_target():
    """Regression test for Hypothesis Experiment 1 (core/MOMENTUM_STRATEGY.md
    §12.9/§13): setting trail_atr=0 must fully disable the trailing stop
    (core/sim/engine.py only activates it when `trail_k > 0`), so a trade
    that would have been cut short by a 1.5-ATR trailing stop instead rides
    to the full 4-ATR target on the SAME price path.

    Hand-built price path (entry_bar=11, ATR held constant at 2.0, sl_atr=2
    -> sl=96, tp_atr=4 -> tp=108, trail_atr=1.5 -> trail_dist=3):
      bars 12-14 climb to a close of 106, arming the trailing stop at 103
      (106 - 1.5*2) by bar 15; bar 15 pulls back to low=100, which is BELOW
      the trailing level (103) but well above the fixed stop (96). With
      trailing on, this exits at 103 -- profitable, but far short of the
      108 target. With trail_atr=0, the fixed stop (96) is never touched,
      price continues climbing, and bar 18's high (110) clears the 108
      target for a clean TP exit.
    """
    idx = _dates(40)
    o = np.full(40, 100.0); h = np.full(40, 100.0); l = np.full(40, 100.0); c = np.full(40, 100.0)
    # bar: open, high, low, close
    path = {
        11: (100, 101.0, 99.5, 101.0),
        12: (101, 103.5, 100.5, 103.0),
        13: (103, 105.5, 102.5, 105.0),
        14: (105, 106.5, 104.5, 106.0),
        15: (104, 105.0, 100.0, 101.0),   # pullback: breaches the trail (103) but not the fixed stop (96)
        16: (101, 104.5, 100.5, 104.0),
        17: (104, 107.5, 103.5, 107.0),
        18: (107, 110.0, 106.5, 109.0),   # high clears the 108 target
    }
    for b, (bo, bh, bl, bc) in path.items():
        o[b], h[b], l[b], c[b] = bo, bh, bl, bc
    for b in range(19, 40):
        o[b] = h[b] = l[b] = c[b] = 109.0

    df_ohlcv = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": np.full(40, 1_000_000.0)}, index=idx)
    ind_df = pd.DataFrame({
        "MA20": np.full(40, 90.0), "MA50": np.full(40, 80.0),  # irrelevant here (feeding simulate() directly)
        "ADX": np.full(40, 30.0), "ATR": np.full(40, 2.0),
        "RSI": np.full(40, 60.0), "MACD": np.zeros(40), "MACD_signal": np.zeros(40),
        "BB_upper": c * 1.05, "BB_lower": c * 0.95,
    }, index=idx)

    dets = pd.DataFrame([{
        "bar_idx": 10, "date": idx[10], "symbol": "T.NS", "pattern": "Momentum_6M", "lookback": "6M",
        "absolute_momentum": 0.2, "relative_momentum": 0.1, "rank": 1.0,
        "technical_confirmed": True, "market_regime_ok": True,
    }])

    def _clean_sim_cfg(cfg):
        sc = default_sim_cfg(cfg)
        sc["filters"]["warmup_cutoff_bars"] = 0
        sc["costs"] = {"per_share": 0.0, "per_trade": 0.0, "bps_fee": 0.0, "slippage_bps": 0.0}
        return sc

    cfg_trail = MomentumConfig(lookback_label="6M", sl_atr=2.0, tp_atr=4.0, trail_atr=1.5)
    result_trail = run_symbol_backtest("T.NS", df_ohlcv, ind_df, dets, cfg_trail, sim_cfg=_clean_sim_cfg(cfg_trail))
    trades_trail = result_trail["trades"]
    assert len(trades_trail) == 1
    t0 = trades_trail.iloc[0]
    assert t0["reason"] == "SL"                 # the trailing stop, mislabeled "SL" by the engine
    assert abs(t0["exit_price"] - 103.0) < 1e-9
    assert t0["exit_bar"] == 15
    assert t0["pnl"] > 0                         # still profitable -- this is the "TRAIL_PROFIT" class from §12.1

    cfg_no_trail = MomentumConfig(lookback_label="6M", sl_atr=2.0, tp_atr=4.0, trail_atr=0.0)
    result_no_trail = run_symbol_backtest("T.NS", df_ohlcv, ind_df, dets, cfg_no_trail, sim_cfg=_clean_sim_cfg(cfg_no_trail))
    trades_no_trail = result_no_trail["trades"]
    assert len(trades_no_trail) == 1
    t1 = trades_no_trail.iloc[0]
    assert t1["reason"] == "TP"
    assert abs(t1["exit_price"] - 108.0) < 1e-9
    assert t1["exit_bar"] == 18

    # The whole point of the hypothesis: removing the trail should realize
    # MORE of the designed reward on an identical price path, not less.
    assert t1["pnl"] > t0["pnl"]


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
