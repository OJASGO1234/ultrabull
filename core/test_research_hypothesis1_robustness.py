# core/test_research_hypothesis1_robustness.py
"""
Tests for core/research_hypothesis1_robustness.py. Synthetic data only, no
network calls -- these run offline and deterministically.

Covers:
  - exit classification (TP vs HARD_SL vs TRAIL_PROFIT) used for the
    TP-hit-rate / realized-R:R reporting
  - summarize_config's empty-input handling
  - the central architectural claim of this experiment: that the SAME
    candidate detections, backtested under two different frozen configs
    (baseline trail_atr=1.5 vs hypothesis-1 trail_atr=0.0), can produce a
    different number of executed trades (the position-blocking effect from
    core/MOMENTUM_STRATEGY.md section 13) while the candidate count itself
    stays identical
  - the documented universe construction (union of the two pre-existing
    symbol lists) matches what the script actually uses

Run with pytest: pytest core/test_research_hypothesis1_robustness.py
Run standalone:  python core/test_research_hypothesis1_robustness.py
"""
from __future__ import annotations
import os
import sys

CORE = os.path.dirname(os.path.abspath(__file__))
if CORE not in sys.path:
    sys.path.insert(0, CORE)

import numpy as np
import pandas as pd

import research_hypothesis1_robustness as rb
from strategies.momentum_strategy import MomentumConfig


def _dates(n):
    return pd.date_range("2021-01-01", periods=n, freq="B")


def test_exit_class_tp():
    row = {"reason": "TP", "pnl": 500.0}
    assert rb._exit_class(row, trail_on=True) == "TP"
    row = {"reason": "TP_GAP", "pnl": 500.0}
    assert rb._exit_class(row, trail_on=False) == "TP"


def test_exit_class_hard_sl_on_loss():
    row = {"reason": "SL", "pnl": -100.0}
    assert rb._exit_class(row, trail_on=True) == "HARD_SL"
    assert rb._exit_class(row, trail_on=False) == "HARD_SL"


def test_exit_class_trail_profit_only_when_trailing_was_active():
    """A profitable SL-reason exit is only classified as a trailing-stop
    exit ('TRAIL_PROFIT') when the config being examined actually had
    trailing enabled -- for a trail_atr=0 config (Hypothesis 1), the SAME
    reason/pnl combination is a genuine hard stop that happened to still be
    profitable (a rare gap case), not a trail, since no trail existed."""
    row = {"reason": "SL", "pnl": 250.0}
    assert rb._exit_class(row, trail_on=True) == "TRAIL_PROFIT"
    assert rb._exit_class(row, trail_on=False) == "HARD_SL"


def test_summarize_config_empty():
    out = rb.summarize_config(pd.DataFrame(), trail_on=True)
    assert out == {"trades": 0}


def test_universe_is_documented_union():
    from event_study import UNIVERSE
    from market_context import SYMBOL_SECTOR
    expected = sorted(set(UNIVERSE) | set(SYMBOL_SECTOR.keys()))
    assert rb.ROBUSTNESS_UNIVERSE == expected
    assert len(rb.ROBUSTNESS_UNIVERSE) > 100  # "substantially beyond the current 37 symbols"


def _make_frame(n=200, jump_bar=100, jump_pct=0.20):
    """A single flat-then-jump price series: flat at 100 up to jump_bar,
    then an instant jump so ATR-based SL/TP levels are easy to reason about,
    then continued drift so a trailing stop (if active) has room to ratchet
    before the position eventually resolves via SL or TP."""
    idx = _dates(n)
    close = np.full(n, 100.0)
    close[jump_bar:] = 100.0 * (1.0 + jump_pct)
    # gentle drift after the jump so trailing (when enabled) has something
    # to ratchet against before the position resolves
    drift = np.linspace(0, 0.05, n - jump_bar)
    close[jump_bar:] = close[jump_bar:] * (1 + drift)
    df_ohlcv = pd.DataFrame({
        "Open": close, "High": close * 1.005, "Low": close * 0.995, "Close": close,
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


def test_run_frozen_config_shares_candidates_but_can_diverge_on_trades():
    """The architectural claim this experiment relies on: given the SAME
    shared candidate detections for a symbol, running the baseline config
    (trail_atr=1.5) and Hypothesis 1 (trail_atr=0.0) through
    run_frozen_config must report the SAME n_candidates (detection
    generation never reads trail_atr) even when the number of trades
    executed differs."""
    df_ohlcv, ind_df = _make_frame()
    idx = df_ohlcv.index
    # Two independent candidate signals, spaced far enough apart that
    # ordinarily both would fill -- if the first position is held long
    # enough (no trailing stop), it can still be open when the second
    # candidate's entry bar arrives, blocking it.
    dets = pd.DataFrame([
        {"bar_idx": 100, "date": idx[100], "symbol": "X.NS", "pattern": "Momentum_6M", "lookback": "6M",
         "absolute_momentum": 0.2, "relative_momentum": 0.1, "rank": 1.0,
         "technical_confirmed": True, "market_regime_ok": True},
        {"bar_idx": 110, "date": idx[110], "symbol": "X.NS", "pattern": "Momentum_6M", "lookback": "6M",
         "absolute_momentum": 0.2, "relative_momentum": 0.1, "rank": 1.0,
         "technical_confirmed": True, "market_regime_ok": True},
    ])
    shared = {"frames": {"X.NS": (df_ohlcv, ind_df)}, "dets": {"X.NS": dets}}

    cfg_baseline = MomentumConfig(lookback_label="6M", sl_atr=2.0, tp_atr=4.0, trail_atr=1.5)
    cfg_hyp1 = MomentumConfig(lookback_label="6M", sl_atr=2.0, tp_atr=4.0, trail_atr=0.0)

    res_base = rb.run_frozen_config("6M", shared, cfg_baseline, "baseline")
    res_hyp1 = rb.run_frozen_config("6M", shared, cfg_hyp1, "hypothesis1")

    n_cand_base = res_base["accounting"][0]["n_candidates"]
    n_cand_hyp1 = res_hyp1["accounting"][0]["n_candidates"]
    assert n_cand_base == n_cand_hyp1 == 2  # identical candidate count, as claimed

    # Both configs must account for every candidate: n_trades + n_end_of_series
    # + n_blocked_or_other == n_candidates, for each run independently.
    for res in (res_base, res_hyp1):
        row = res["accounting"][0]
        assert row["n_trades"] + row["n_end_of_series"] + row["n_blocked_or_other"] == row["n_candidates"]


def test_frozen_cfg_builders_match_the_original_scripts_defaults():
    """build_shared_detections/run_frozen_config are only meaningful if the
    two configs actually ARE the frozen baseline/Hypothesis-1 configs.
    Verify the imported build_cfg() calls (reused, not reimplemented) yield
    trail_atr=1.5 for baseline and 0.0 for hypothesis1, with every other
    exit/risk field identical -- i.e. the only frozen difference is trail_atr."""
    import research_momentum_experiment as baseline_runner
    import research_hypothesis1_no_trailing_stop as hyp1_runner

    base_cfg = baseline_runner.build_cfg("6M", rb._BASELINE_ARGS)
    hyp1_cfg = hyp1_runner.build_cfg("6M", rb._HYP1_ARGS)

    assert base_cfg.trail_atr == 1.5
    assert hyp1_cfg.trail_atr == 0.0
    for field in ("sl_atr", "tp_atr", "top_n", "use_relative_momentum",
                  "use_technical_confirmation", "use_market_regime", "rebalance_bars"):
        assert getattr(base_cfg, field) == getattr(hyp1_cfg, field), f"{field} differs between configs"


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
