# core/test_research_universe_size_sensitivity.py
"""
Tests for core/research_universe_size_sensitivity.py (core/MOMENTUM_STRATEGY.md
section 16). Synthetic data only, no network calls -- these run offline and
deterministically.

Covers:
  - deterministic, nested universe construction (37 subset of 55 subset of
    75 subset of 95 subset of 111), both the pure helper and the real
    module-level UNIVERSE_SIZES built from event_study.UNIVERSE /
    research_hypothesis1_robustness.ROBUSTNESS_UNIVERSE
  - the walk-forward fold boundaries are chronological and mutually
    non-overlapping
  - train/validation partitioning: train is strictly before a fold's
    validation start, validation trades fall only inside that fold's
    window, and the two sets never share a trade (no lookahead, no overlap)
  - pooled validation is the exact disjoint union of the four folds'
    validation trades, and its aggregated metrics equal a manual sum
    (metric aggregation, not a fake constant check)
  - concentration measurement (top-5 share, distinct symbols, average
    eligible pool) on a hand-built selection mask with a known answer
  - the central mechanism this experiment exists to test: restricting
    run_universe_lookback to a SUBSET of symbols changes which symbols get
    selected (built from real absolute-momentum math on synthetic
    geometric-drift price series with a known, distinct ranking)
  - graceful handling of insufficient history (no candidates, no crash)
  - deterministic repeated execution of the full run_universe_lookback path

Run with pytest: pytest core/test_research_universe_size_sensitivity.py
Run standalone:  python core/test_research_universe_size_sensitivity.py
"""
from __future__ import annotations
import json
import os
import sys

CORE = os.path.dirname(os.path.abspath(__file__))
if CORE not in sys.path:
    sys.path.insert(0, CORE)

import numpy as np
import pandas as pd

import research_universe_size_sensitivity as rus
import research_hypothesis1_robustness as rb
from strategies.momentum_strategy import MomentumConfig, compute_momentum_panels, eligibility_mask


# ---------------------------------------------------------------------------
# Universe construction
# ---------------------------------------------------------------------------
def test_build_nested_universes_pure_and_deterministic():
    base = ["b", "a", "c"]
    extra_pool = ["z", "y", "x", "w", "c"]  # "c" duplicates a base symbol -- must be excluded from "extra"
    sizes = (3, 4, 5)
    out1 = rus.build_nested_universes(base, extra_pool, sizes)
    out2 = rus.build_nested_universes(base, extra_pool, sizes)
    assert out1 == out2, "same inputs must produce the exact same output on repeated calls"

    assert out1[3] == ["a", "b", "c"]
    assert out1[4] == ["a", "b", "c", "w"]
    assert out1[5] == ["a", "b", "c", "w", "x"]
    assert len(out1[3]) == 3 and len(out1[4]) == 4 and len(out1[5]) == 5
    assert set(out1[3]).issubset(set(out1[4]))
    assert set(out1[4]).issubset(set(out1[5]))


def test_build_nested_universes_rejects_size_smaller_than_base():
    raised = False
    try:
        rus.build_nested_universes(["a", "b", "c"], ["d", "e"], (2,))
    except ValueError:
        raised = True
    assert raised


def test_build_nested_universes_rejects_size_beyond_available_pool():
    raised = False
    try:
        rus.build_nested_universes(["a", "b", "c"], ["d"], (10,))
    except ValueError:
        raised = True
    assert raised


def test_real_universe_sizes_are_nested_and_correct_lengths():
    sizes = rus.UNIVERSE_SIZES
    assert set(sizes.keys()) == set(rus.UNIVERSE_TARGET_SIZES)
    for target in rus.UNIVERSE_TARGET_SIZES:
        assert len(sizes[target]) == target, f"universe size {target} has {len(sizes[target])} symbols"
    for a, b in zip(rus.UNIVERSE_TARGET_SIZES, rus.UNIVERSE_TARGET_SIZES[1:]):
        assert set(sizes[a]).issubset(set(sizes[b])), f"universe {a} must be a subset of universe {b}"


def test_universe_37_matches_event_study_universe():
    from event_study import UNIVERSE
    assert set(rus.UNIVERSE_SIZES[37]) == set(UNIVERSE)


def test_universe_111_matches_robustness_universe_as_a_set():
    from research_hypothesis1_robustness import ROBUSTNESS_UNIVERSE
    assert set(rus.UNIVERSE_SIZES[111]) == set(ROBUSTNESS_UNIVERSE)
    assert len(rus.UNIVERSE_SIZES[111]) == len(ROBUSTNESS_UNIVERSE)


# ---------------------------------------------------------------------------
# Walk-forward folds
# ---------------------------------------------------------------------------
def test_folds_are_chronological_and_non_overlapping():
    for f in rus.FOLDS:
        assert f["valid_start"] < f["valid_end"]
    for a, b in zip(rus.FOLDS, rus.FOLDS[1:]):
        assert a["valid_end"] <= b["valid_start"], "consecutive folds' validation windows must not overlap"


def test_fold_for_date_boundaries():
    assert rus.fold_for_date("2021-12-31") is None
    assert rus.fold_for_date("2022-01-01") == 1
    assert rus.fold_for_date("2022-12-31") == 1
    assert rus.fold_for_date("2023-01-01") == 2
    assert rus.fold_for_date("2025-06-15") == 4
    assert rus.fold_for_date("2025-12-31") == 4
    assert rus.fold_for_date("2026-01-01") is None  # on/after the last fold's validation end


def test_train_valid_partition_no_overlap_and_chronological():
    """A trade dated inside fold 2's validation window must appear in
    fold 2's VALID bucket and never in fold 2's TRAIN bucket (no
    lookahead); a trade strictly before fold 2's validation window must
    appear in TRAIN and never in VALID (no data loss); train is always
    chronologically before validation."""
    trades = pd.DataFrame({"entry_date": pd.to_datetime(
        ["2020-01-01", "2022-06-15", "2023-06-15", "2025-12-31", "2026-06-01"]
    )})
    fold2 = rus.FOLDS[1]  # valid = [2023-01-01, 2024-01-01)

    train_t = rus.train_trades_for_fold(trades, fold2)
    valid_t = rus.valid_trades_for_fold(trades, fold2)

    assert len(train_t) > 0 and len(valid_t) > 0
    assert train_t["entry_date"].max() < fold2["valid_start"]
    assert valid_t["entry_date"].min() >= fold2["valid_start"]
    assert valid_t["entry_date"].max() < fold2["valid_end"]
    assert set(train_t.index).isdisjoint(set(valid_t.index))

    assert (train_t["entry_date"] == pd.Timestamp("2023-06-15")).sum() == 0
    assert (valid_t["entry_date"] == pd.Timestamp("2023-06-15")).sum() == 1
    # the 2020 trade is far before fold 2's window -> TRAIN only
    assert (train_t["entry_date"] == pd.Timestamp("2020-01-01")).sum() == 1
    assert (valid_t["entry_date"] == pd.Timestamp("2020-01-01")).sum() == 0


def test_pooled_valid_is_disjoint_union_of_fold_validations():
    dates = pd.to_datetime(["2021-01-01", "2022-05-01", "2023-05-01", "2024-05-01", "2025-05-01", "2026-05-01"])
    trades = pd.DataFrame({"entry_date": dates})
    trades["valid_fold"] = trades["entry_date"].apply(rus.fold_for_date)

    pooled = rus.pooled_valid_trades(trades)
    assert len(pooled) == 4  # only the 2022/2023/2024/2025 dates fall inside a fold window
    assert set(pooled["valid_fold"]) == {1, 2, 3, 4}

    manual = pd.concat([rus.valid_trades_for_fold(trades, f) for f in rus.FOLDS])
    assert sorted(pooled["entry_date"].tolist()) == sorted(manual["entry_date"].tolist())


def test_pooled_metrics_equal_manual_sum_across_folds():
    """Metric aggregation: pooling four disjoint folds' validation trades
    and summarizing them must equal the sum/count of the individual trades
    that actually fall inside a fold window -- not a re-typed constant."""
    dates = pd.to_datetime(["2022-03-01", "2023-03-01", "2024-03-01", "2025-03-01", "2020-01-01", "2026-06-01"])
    pnls = [100.0, -50.0, 200.0, -25.0, 999.0, 888.0]  # the last two are OUTSIDE every fold
    trades = pd.DataFrame({
        "entry_date": dates, "entry_bar": [10] * 6, "exit_bar": [15] * 6,
        "pnl": pnls, "reason": ["TP", "SL", "TP", "SL", "TP", "TP"],
    })
    trades["valid_fold"] = trades["entry_date"].apply(rus.fold_for_date)
    pooled = rus.pooled_valid_trades(trades)
    assert len(pooled) == 4
    expected_pnl = 100.0 - 50.0 + 200.0 - 25.0
    assert abs(pooled["pnl"].sum() - expected_pnl) < 1e-9

    m = rb.summarize_config(pooled, trail_on=False)
    assert m["trades"] == 4
    assert abs(m["total_pnl"] - expected_pnl) < 1e-9
    assert abs(m["win_rate"] - 0.5) < 1e-9  # 2 of 4 pooled trades are wins


# ---------------------------------------------------------------------------
# Concentration
# ---------------------------------------------------------------------------
def test_measure_concentration_basic_counts():
    idx = pd.date_range("2021-01-01", periods=4, freq="B")
    selected = pd.DataFrame({
        "A": [True, True, True, False],
        "B": [False, True, False, False],
        "C": [False, False, False, True],
    }, index=idx)
    eligible = pd.DataFrame(True, index=idx, columns=["A", "B", "C"])
    out = rus.measure_concentration(selected, eligible)
    assert out["total_selection_slots"] == 5  # 3 + 1 + 1
    assert out["n_distinct_selected_symbols"] == 3
    assert abs(out["top5_share"] - 1.0) < 1e-9  # only 3 symbols exist, top-5 captures all of them
    assert out["avg_eligible_universe_size"] == 3.0
    top5 = json.loads(out["top5_symbols_json"])
    assert top5 == {"A": 3, "B": 1, "C": 1}


def test_measure_concentration_empty_selected_is_graceful():
    idx = pd.date_range("2021-01-01", periods=3, freq="B")
    eligible = pd.DataFrame({"A": [True, True, True], "B": [True, False, True]}, index=idx)
    out = rus.measure_concentration(pd.DataFrame(), eligible)
    assert out["total_selection_slots"] == 0
    assert np.isnan(out["top5_share"])
    assert abs(out["avg_eligible_universe_size"] - (2 + 1 + 2) / 3) < 1e-9


# ---------------------------------------------------------------------------
# run_universe_lookback: the mechanism under test (does universe size
# actually change which symbols get selected?), plus graceful edge cases
# and determinism -- built on real momentum math, synthetic price data.
# ---------------------------------------------------------------------------
def _drift_frame(n: int, daily_return: float, start="2015-01-01"):
    """Constant-daily-return synthetic price series: absolute_momentum over
    ANY lookback is then exactly (1+daily_return)**lookback_bars - 1 for
    every bar once enough history exists -- a stable, known ranking, not a
    one-off jump that decays."""
    idx = pd.bdate_range(start, periods=n)
    t = np.arange(n)
    close = 100.0 * (1.0 + daily_return) ** t
    df_ohlcv = pd.DataFrame({
        "Open": close, "High": close * 1.001, "Low": close * 0.999, "Close": close,
        "Volume": np.full(n, 1_000_000.0),
    }, index=idx)
    ind_df = pd.DataFrame({
        "MA20": close, "MA50": close, "ADX": np.full(n, 30.0), "ATR": np.full(n, 2.0),
        "RSI": np.full(n, 60.0), "MACD": np.zeros(n), "MACD_signal": np.zeros(n),
        "BB_upper": close * 1.05, "BB_lower": close * 0.95,
    }, index=idx)
    return df_ohlcv, ind_df


def _six_symbol_frames(n=200):
    # Strictly decreasing daily-return ranking: A > B > C > D > E > F, always.
    rates = {"A.NS": 0.0080, "B.NS": 0.0060, "C.NS": 0.0040, "D.NS": 0.0020, "E.NS": 0.0010, "F.NS": 0.0005}
    frames = {sym: _drift_frame(n, r) for sym, r in rates.items()}
    bench = pd.Series(100.0, index=next(iter(frames.values()))[0].index)  # flat benchmark, 0% momentum
    return frames, bench


def test_universe_subset_changes_which_symbols_are_selected():
    """The core claim this experiment exists to test: the SAME underlying
    price data and the SAME config, restricted to a smaller universe,
    selects a DIFFERENT top-N than the full universe does. C.NS ranks 3rd
    of 6 (never selected with top_n=2) but 1st of the 4-symbol subset
    {C,D,E,F} (always selected)."""
    frames, bench = _six_symbol_frames()
    cfg = MomentumConfig(lookback_label="3M", trail_atr=0.0, top_n=2,
                          use_technical_confirmation=False, use_market_regime=False)
    ctx_by_symbol = {s: None for s in frames}
    abs_full, rel_full = compute_momentum_panels(frames, bench, cfg.lookback_bars)
    eligible_full = eligibility_mask(abs_full, rel_full, cfg)

    out_full = rus.run_universe_lookback(
        "3M", 6, list(frames.keys()), frames, ctx_by_symbol, abs_full, rel_full, eligible_full, cfg,
    )
    selected_full = set(json.loads(out_full["concentration"]["top5_symbols_json"]).keys())
    assert selected_full == {"A.NS", "B.NS"}

    subset_symbols = ["C.NS", "D.NS", "E.NS", "F.NS"]
    out_subset = rus.run_universe_lookback(
        "3M", 4, subset_symbols, frames, ctx_by_symbol, abs_full, rel_full, eligible_full, cfg,
    )
    selected_subset = set(json.loads(out_subset["concentration"]["top5_symbols_json"]).keys())
    assert selected_subset == {"C.NS", "D.NS"}, (
        "C.NS must become selected once the stronger A.NS/B.NS are removed from the universe -- "
        "this is the universe-breadth effect under study"
    )


def test_run_universe_lookback_is_deterministic_on_repeated_calls():
    frames, bench = _six_symbol_frames(n=150)
    cfg = MomentumConfig(lookback_label="6M", trail_atr=0.0, top_n=2,
                          use_technical_confirmation=False, use_market_regime=False)
    ctx_by_symbol = {s: None for s in frames}
    abs_full, rel_full = compute_momentum_panels(frames, bench, cfg.lookback_bars)
    eligible_full = eligibility_mask(abs_full, rel_full, cfg)
    symbols = list(frames.keys())

    out1 = rus.run_universe_lookback("6M", 6, symbols, frames, ctx_by_symbol, abs_full, rel_full, eligible_full, cfg)
    out2 = rus.run_universe_lookback("6M", 6, symbols, frames, ctx_by_symbol, abs_full, rel_full, eligible_full, cfg)

    pd.testing.assert_frame_equal(out1["trades"], out2["trades"])
    assert out1["concentration"] == out2["concentration"]


def test_run_universe_lookback_insufficient_history_is_graceful():
    """A series far shorter than the lookback's bar requirement must
    produce zero candidates and zero trades -- never a crash, never a
    fabricated signal."""
    n = 30  # well under 3M's 63-bar lookback
    idx = pd.bdate_range("2021-01-01", periods=n)
    close = np.linspace(100, 110, n)
    df_ohlcv = pd.DataFrame({
        "Open": close, "High": close * 1.001, "Low": close * 0.999, "Close": close,
        "Volume": np.full(n, 1_000_000.0),
    }, index=idx)
    ind_df = pd.DataFrame({
        "MA20": close, "MA50": close, "ADX": np.full(n, 30.0), "ATR": np.full(n, 2.0),
        "RSI": np.full(n, 60.0), "MACD": np.zeros(n), "MACD_signal": np.zeros(n),
        "BB_upper": close * 1.05, "BB_lower": close * 0.95,
    }, index=idx)
    frames = {"X.NS": (df_ohlcv, ind_df)}
    bench = pd.Series(100.0, index=idx)
    cfg = MomentumConfig(lookback_label="3M", trail_atr=0.0, use_market_regime=False)

    abs_full, rel_full = compute_momentum_panels(frames, bench, cfg.lookback_bars)
    eligible_full = eligibility_mask(abs_full, rel_full, cfg)
    out = rus.run_universe_lookback("3M", 1, ["X.NS"], frames, {"X.NS": None}, abs_full, rel_full, eligible_full, cfg)

    assert len(out["trades"]) == 0
    assert out["concentration"]["n_candidates_total"] == 0
    assert out["concentration"]["n_trades_total"] == 0
    assert np.isnan(out["concentration"]["blocked_rate"])
    assert out["concentration"]["total_selection_slots"] == 0


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
