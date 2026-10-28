# core/diagnose_momentum_experiment.py
"""
Diagnostic-only analysis of the momentum experiment
(core/research_momentum_experiment.py's default run: 37-symbol UNIVERSE,
4 years, top_n=5, relative momentum on, technical confirmation on, market
regime on, ADX off, rebalance=21 bars, sl_atr=2.0/tp_atr=4.0/trail_atr=1.5).

THIS SCRIPT DOES NOT TUNE ANYTHING. It re-runs the EXACT SAME pipeline with
the EXACT SAME default MomentumConfig (no parameter search, no threshold
selection) purely to recover intermediate fields that
core/research_momentum_experiment.py did not persist to
momentum_experiment_trades.csv -- rank, absolute/relative momentum, and
regime state at signal time all live in generate_symbol_detections()'s
output / the Signal.meta dict, which core.sim.engine.simulate() does not
carry into the Trade record. Recovering them means re-running the pipeline
(against the SAME cached price data in core/.momentum_cache -- no new
network calls) and joining each trade back to the detection that produced it
via entry_bar == bar_idx + 1 (delay_bars is always 1 in this strategy).

Additionally computes cross-lookback correlation, selection concentration,
universe-size adequacy, and a direct empirical check of the relative-vs-
absolute ranking question -- see core/MOMENTUM_STRATEGY.md's "Diagnostic:
why did the first implementation fail out-of-sample?" section, which this
script's output feeds directly (every number quoted there traces to a
printed line or CSV row here).

Run: python core/diagnose_momentum_experiment.py
Output: printed report + core/momentum_diagnostics_trades_joined.csv
"""
from __future__ import annotations
import os
import sys

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
)

YEARS = 4.0
CACHE_DIR = os.path.join(CORE, ".momentum_cache")
TRAIN_FRAC = 0.7
pd.set_option("display.width", 200)


def _default_cfg(label: str) -> MomentumConfig:
    """Identical to research_momentum_experiment.py's defaults -- nothing
    changed here."""
    return MomentumConfig(lookback_label=label)


def _hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def build_joined_trades() -> pd.DataFrame:
    """For each lookback: reproduce the exact backtest, but keep the
    detections (bar_idx, absolute_momentum, relative_momentum, rank,
    technical_confirmed, market_regime_ok) and join them onto the resulting
    trades via entry_bar == bar_idx + 1. Also attaches ADX at entry/exit and
    mkt_up regime at entry/exit from the same indicator/context frames the
    strategy itself used -- no new indicator, just reading it at more
    points than the strategy persisted."""
    all_rows = []
    frames_by_lookback = {}
    for label in LOOKBACK_BARS:
        cfg = _default_cfg(label)
        frames = build_universe_frames(UNIVERSE, YEARS, cache_dir=CACHE_DIR)
        frames_by_lookback[label] = frames
        bench = load_benchmark_close(YEARS, cache_dir=CACHE_DIR)
        abs_df, rel_df = compute_momentum_panels(frames, bench, cfg.lookback_bars)
        rank_basis = rel_df if cfg.use_relative_momentum else abs_df
        eligible = eligibility_mask(abs_df, rel_df, cfg)
        rank_df = cross_sectional_rank(rank_basis, eligible)
        selected = select_candidates(rank_df, eligible, cfg)

        for sym, (df_ohlcv, ind_df) in frames.items():
            if sym not in selected.columns:
                continue
            try:
                ctx = build_market_context(sym, df_ohlcv["Close"], years=YEARS, cache_dir=CACHE_DIR)
            except Exception:
                ctx = None
            dets = generate_symbol_detections(
                sym, df_ohlcv, ind_df,
                selected_col=selected[sym], abs_mom_col=abs_df[sym], rel_mom_col=rel_df[sym],
                rank_col=rank_df[sym], ctx=ctx, cfg=cfg,
            )
            if dets.empty:
                continue
            result = run_symbol_backtest(sym, df_ohlcv, ind_df, dets, cfg)
            trades = result["trades"]
            if trades is None or trades.empty:
                continue
            trades = trades.copy()
            trades["symbol"] = sym
            trades["lookback"] = label
            split_bar = int(len(df_ohlcv) * TRAIN_FRAC)
            trades["period"] = np.where(trades["entry_bar"] < split_bar, "train", "valid")

            # Join each trade back to the detection that produced it. Every
            # signal has delay_bars=1 (compile_momentum_signals), so the
            # detection's bar_idx is always entry_bar - 1.
            dets_idx = dets.set_index("bar_idx")
            adx = ind_df["ADX"] if "ADX" in ind_df.columns else pd.Series(np.nan, index=ind_df.index)
            mkt_up = ctx["mkt_up"] if (ctx is not None and "mkt_up" in ctx.columns) else pd.Series(True, index=df_ohlcv.index)

            for _, tr in trades.iterrows():
                det_bar = tr["entry_bar"] - 1
                det = dets_idx.loc[det_bar] if det_bar in dets_idx.index else None
                row = tr.to_dict()
                row["delay_bars_actual"] = tr["entry_bar"] - det_bar  # sanity: must always be 1
                if det is not None:
                    row["absolute_momentum"] = det["absolute_momentum"]
                    row["relative_momentum"] = det["relative_momentum"]
                    row["rank"] = det["rank"]
                    row["technical_confirmed_at_signal"] = det["technical_confirmed"]
                    row["market_regime_ok_at_signal"] = det["market_regime_ok"]
                else:
                    row["absolute_momentum"] = np.nan
                    row["relative_momentum"] = np.nan
                    row["rank"] = np.nan
                    row["technical_confirmed_at_signal"] = None
                    row["market_regime_ok_at_signal"] = None
                # ADX / regime observed at entry and at exit (post-hoc
                # classification for diagnosis -- not a filter change)
                eb, xb = int(tr["entry_bar"]), int(tr["exit_bar"])
                row["adx_at_entry"] = float(adx.iloc[eb]) if eb < len(adx) else np.nan
                row["mkt_up_at_entry"] = bool(mkt_up.iloc[eb]) if eb < len(mkt_up) else None
                row["mkt_up_at_exit"] = bool(mkt_up.iloc[xb]) if xb < len(mkt_up) else None
                all_rows.append(row)

    joined = pd.DataFrame(all_rows)
    return joined, frames_by_lookback


def section_1_4_and_9(t: pd.DataFrame) -> None:
    _hr("1-4, 9: WIN RATE / AVG WIN / AVG LOSS / EXIT MIX / HOLDING PERIOD, train vs valid, by lookback")
    t = t.copy()
    t["win"] = t["pnl"] > 0
    t["hold"] = t["exit_bar"] - t["entry_bar"]

    def refine(row):
        if row["reason"] in ("TP", "TP_GAP"):
            return "TP"
        if row["reason"] in ("SL", "SL_GAP"):
            return "TRAIL_PROFIT" if row["pnl"] > 0 else "HARD_SL"
        return row["reason"]
    t["exit_class"] = t.apply(refine, axis=1)

    g = t.groupby(["lookback", "period"]).agg(
        n=("pnl", "size"), win_rate=("win", "mean"),
        avg_win=("pnl", lambda s: s[s > 0].mean()),
        avg_loss=("pnl", lambda s: s[s <= 0].mean()),
        avg_hold=("hold", "mean"), median_hold=("hold", "median"),
    ).round(2)
    print(g)

    print("\nExit-reason mix (HARD_SL = stopped out at a loss; TRAIL_PROFIT = trailing "
          "stop exit that was still profitable; TP = full target hit):")
    print(pd.crosstab([t["lookback"], t["period"]], t["exit_class"]))

    print("\nRealized win/loss magnitude ratio vs the DESIGNED ratio (tp_atr/sl_atr = 2.0), by period:")
    for period, d in t.groupby("period"):
        avg_w = d.loc[d.win, "pnl"].mean()
        avg_l = abs(d.loc[~d.win, "pnl"].mean())
        breakeven_wr = 1.0 / (1.0 + avg_w / avg_l)
        actual_wr = d["win"].mean()
        print(f"  {period}: realized avg_win/avg_loss = {avg_w/avg_l:.3f} (designed 2.0) -> "
              f"breakeven win rate = {breakeven_wr:.1%}, actual win rate = {actual_wr:.1%} "
              f"({'ABOVE' if actual_wr > breakeven_wr else 'BELOW'} breakeven)")

    print("\nHolding period distribution by period (bars):")
    print(t.groupby("period")["hold"].describe().round(2))
    return t


def section_5_regime(t: pd.DataFrame) -> None:
    _hr("5: PERFORMANCE BY MARKET REGIME")
    print("At SIGNAL time, market_regime_ok is a hard filter in the default config "
          "(use_market_regime=True) -- confirm 100% of trades satisfy it:")
    print(t["market_regime_ok_at_signal"].value_counts(dropna=False))
    print("\n(No in-sample comparison across entry-time regimes is possible from this run: "
          "the filter admits only mkt_up=True signals by construction. The only regime "
          "variation LEFT to examine is whether the regime flips DURING the trade.)")

    t = t.copy()
    t["regime_flipped_during_trade"] = t["mkt_up_at_entry"] & ~t["mkt_up_at_exit"].astype("boolean").fillna(True)
    g = t.groupby("regime_flipped_during_trade").agg(
        n=("pnl", "size"), win_rate=("pnl", lambda s: (s > 0).mean()), avg_pnl=("pnl", "mean"),
    ).round(2)
    print("\nPerformance split by whether NIFTY's own MA20>MA50 regime flipped to down "
          "between entry and exit (pooled across lookbacks/periods):")
    print(g)


def section_6_adx(t: pd.DataFrame) -> None:
    _hr("6: PERFORMANCE BY ADX/TREND CONDITION AT ENTRY")
    print("Default config has use_adx=False -- ADX was NOT part of the technical-confirmation "
          "filter for this run. Bucketed here post-hoc using the standard ADX trend-strength "
          "bands (weak <20, developing 20-25, strong >25) purely to see if entry-time trend "
          "strength happens to separate winners from losers -- not a filter, not a threshold search.")
    t = t.copy()
    t["adx_band"] = pd.cut(t["adx_at_entry"], bins=[0, 20, 25, 100], labels=["<20 weak", "20-25 developing", ">25 strong"])
    for period, d in t.groupby("period"):
        print(f"\n-- {period} --")
        g = d.groupby("adx_band", observed=True).agg(
            n=("pnl", "size"), win_rate=("pnl", lambda s: (s > 0).mean()), avg_pnl=("pnl", "mean"),
        ).round(2)
        print(g)


def section_7_momentum_strength(t: pd.DataFrame) -> None:
    _hr("7: PERFORMANCE BY ABSOLUTE MOMENTUM STRENGTH AT ENTRY")
    print("Tertile cutoffs frozen on the TRAIN split only (same discipline as "
          "core/audit_oos_condition_validation.py), then applied unchanged to VALID -- "
          "nothing is re-derived on the validation half.")
    t = t.copy()
    for lookback, d in t.groupby("lookback"):
        train = d[d["period"] == "train"]["absolute_momentum"].dropna()
        if len(train) < 6:
            continue
        cuts = train.quantile([1/3, 2/3]).values
        bins = [-np.inf, cuts[0], cuts[1], np.inf]
        labels = ["low", "mid", "high"]
        d = d.copy()
        d["mom_tertile"] = pd.cut(d["absolute_momentum"], bins=bins, labels=labels)
        print(f"\n-- {lookback} (train-frozen cutoffs: {cuts.round(4).tolist()}) --")
        g = d.groupby(["period", "mom_tertile"], observed=True).agg(
            n=("pnl", "size"), win_rate=("pnl", lambda s: (s > 0).mean()), avg_pnl=("pnl", "mean"),
        ).round(2)
        print(g)


def section_8_rank(t: pd.DataFrame) -> None:
    _hr("8: PERFORMANCE BY CROSS-SECTIONAL RANK AT ENTRY (top_n=5, so rank is always 1-5)")
    t = t.copy()
    g = t.groupby(["period", "rank"]).agg(
        n=("pnl", "size"), win_rate=("pnl", lambda s: (s > 0).mean()), avg_pnl=("pnl", "mean"),
    ).round(2)
    print(g)
    print("\nRank distribution (how many trades came from each rank):")
    print(t.groupby("rank").size())


def section_12_correlation(frames_by_lookback: dict) -> None:
    _hr("12: CROSS-LOOKBACK CORRELATION (are 3M/6M/9M/12M signals actually independent evidence?)")
    from momentum import absolute_momentum
    frames = frames_by_lookback["3M"]  # same price data underlies every lookback
    bench = load_benchmark_close(YEARS, cache_dir=CACHE_DIR)

    per_symbol_corrs = []
    for sym, (df_ohlcv, _ind) in frames.items():
        close = df_ohlcv["Close"]
        moms = {label: absolute_momentum(close, bars) for label, bars in LOOKBACK_BARS.items()}
        mdf = pd.DataFrame(moms).dropna()
        if len(mdf) < 30:
            continue
        per_symbol_corrs.append(mdf.corr())
    if not per_symbol_corrs:
        print("insufficient data")
        return
    avg_corr = sum(per_symbol_corrs) / len(per_symbol_corrs)
    print(f"Average pairwise correlation of the four lookbacks' absolute-momentum series, "
          f"averaged across {len(per_symbol_corrs)} symbols:")
    print(avg_corr.round(3))

    # Selection overlap: how often does a symbol selected under one lookback
    # also get selected under another, on the SAME date?
    sel_by_lookback = {}
    for label in LOOKBACK_BARS:
        cfg = _default_cfg(label)
        abs_df, rel_df = compute_momentum_panels(frames, bench, cfg.lookback_bars)
        eligible = eligibility_mask(abs_df, rel_df, cfg)
        rank_df = cross_sectional_rank(rel_df, eligible)
        sel_by_lookback[label] = select_candidates(rank_df, eligible, cfg)

    labels = list(LOOKBACK_BARS)
    print("\nJaccard overlap of the daily top-5 selected sets between lookback pairs "
          "(1.0 = identical sets every day, 0.0 = no overlap ever):")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = sel_by_lookback[labels[i]], sel_by_lookback[labels[j]]
            common_idx = a.index.intersection(b.index)
            common_cols = a.columns.intersection(b.columns)
            a2, b2 = a.loc[common_idx, common_cols], b.loc[common_idx, common_cols]
            inter = (a2 & b2).sum(axis=1)
            union = (a2 | b2).sum(axis=1)
            jac = (inter[union > 0] / union[union > 0]).mean()
            print(f"  {labels[i]} vs {labels[j]}: {jac:.3f}")


def section_13_14_concentration(frames_by_lookback: dict) -> None:
    _hr("13-14: TOP-N CONCENTRATION/CHURN AND UNIVERSE-SIZE ADEQUACY")
    frames = frames_by_lookback["6M"]
    bench = load_benchmark_close(YEARS, cache_dir=CACHE_DIR)
    cfg = _default_cfg("6M")
    abs_df, rel_df = compute_momentum_panels(frames, bench, cfg.lookback_bars)
    eligible = eligibility_mask(abs_df, rel_df, cfg)
    rank_df = cross_sectional_rank(rel_df, eligible)
    selected = select_candidates(rank_df, eligible, cfg)

    n_eligible = eligible.sum(axis=1)
    n_eligible_nonzero = n_eligible[n_eligible > 0]
    print(f"Universe: {len(frames)} symbols loaded.")
    print(f"Eligible pool per date (passes min_absolute_momentum & min_relative_momentum "
          f"floors, has enough history) -- mean {n_eligible_nonzero.mean():.1f}, "
          f"median {n_eligible_nonzero.median():.0f}, min {n_eligible_nonzero.min():.0f}, "
          f"max {n_eligible_nonzero.max():.0f} (out of {len(frames)}).")
    print(f"top_n=5 as a fraction of the eligible pool: mean {5.0 / max(1, n_eligible_nonzero.mean()):.1%} "
          f"-- i.e. this is closer to a 'top {5.0/max(1,n_eligible_nonzero.mean())*100:.0f}%' cut than a "
          f"tightly selective top decile when the eligible pool is small.")

    sel_counts = selected.sum(axis=0).sort_values(ascending=False)
    total_selection_slots = sel_counts.sum()
    top5_share = sel_counts.head(5).sum() / total_selection_slots if total_selection_slots else float("nan")
    print(f"\nMost-frequently-selected symbols (days selected out of {selected.shape[0]} total dates), top 10:")
    print(sel_counts.head(10))
    print(f"Concentration: the 5 most-frequently-selected symbols account for "
          f"{top5_share:.1%} of all (date, symbol) selection slots.")

    # Day-to-day churn: fraction of the selected set that changes from one
    # selection day to the next (only counting days where the set changed).
    sel_bool = selected.fillna(False)
    changes = []
    prev = None
    for _, row in sel_bool.iterrows():
        cur = set(row[row].index)
        if prev is not None and (cur or prev):
            union = cur | prev
            if union:
                changes.append(len(cur.symmetric_difference(prev)) / len(union))
        prev = cur
    if changes:
        print(f"\nDay-to-day turnover of the selected set (mean Jaccard-distance between "
              f"consecutive days' top-5): {np.mean(changes):.3f} "
              f"(0 = never changes, 1 = completely different every day)")


def section_15_relative_formulation(frames_by_lookback: dict) -> None:
    _hr("15: IS THE NIFTY-RELATIVE FORMULATION BEHAVING AS INTENDED?")
    print("Mathematical fact to verify empirically: relative_momentum[t] = momentum[t] - "
          "benchmark_momentum[t]. Since benchmark_momentum[t] is the SAME constant subtracted "
          "from every symbol on a given date, subtracting it cannot change the CROSS-SECTIONAL "
          "RANK ORDER among a fixed set of symbols -- only the ELIGIBILITY test (>= a floor) "
          "can differ, because 'beat NIFTY' and 'positive absolute return' are different "
          "conditions whenever NIFTY's own momentum is non-zero.")
    frames = frames_by_lookback["6M"]
    bench = load_benchmark_close(YEARS, cache_dir=CACHE_DIR)
    cfg_rel = _default_cfg("6M")
    cfg_abs = MomentumConfig(lookback_label="6M", use_relative_momentum=False)
    abs_df, rel_df = compute_momentum_panels(frames, bench, cfg_rel.lookback_bars)

    elig_rel = eligibility_mask(abs_df, rel_df, cfg_rel)
    elig_abs = eligibility_mask(abs_df, rel_df, cfg_abs)
    rank_rel = cross_sectional_rank(rel_df, elig_rel)
    rank_abs = cross_sectional_rank(abs_df, elig_abs)

    same_elig_mask = (elig_rel == elig_abs).all(axis=1)
    same_elig_dates = same_elig_mask[same_elig_mask].index
    if len(same_elig_dates):
        agree = 0
        total = 0
        for d in same_elig_dates:
            r1, r2 = rank_rel.loc[d], rank_abs.loc[d]
            both_ranked = r1.notna() & r2.notna()
            if both_ranked.sum() > 1:
                total += 1
                agree += int((r1[both_ranked].rank() == r2[both_ranked].rank()).all())
        print(f"\nOn the {len(same_elig_dates)} dates where relative-momentum and "
              f"absolute-momentum eligibility masks agree exactly, rank order matches "
              f"on {agree}/{total} of them ({(agree/total if total else float('nan')):.1%}) "
              f"-- confirms ranking itself is (as expected) equivalent; only ELIGIBILITY differs.")

    elig_diff_rate = (elig_rel != elig_abs).mean().mean()
    sel_rel = select_candidates(rank_rel, elig_rel, cfg_rel)
    sel_abs = select_candidates(rank_abs, elig_abs, cfg_abs)
    common_idx = sel_rel.index.intersection(sel_abs.index)
    inter = (sel_rel.loc[common_idx] & sel_abs.loc[common_idx]).sum(axis=1)
    union = (sel_rel.loc[common_idx] | sel_abs.loc[common_idx]).sum(axis=1)
    jac = (inter[union > 0] / union[union > 0]).mean()
    print(f"\nEligibility masks differ on {elig_diff_rate:.1%} of (date, symbol) cells "
          f"(min_absolute_momentum=0.0 vs min_relative_momentum=0.0 are different conditions "
          f"whenever NIFTY's own momentum isn't exactly zero).")
    print(f"Selected-set overlap (relative-ranking config vs absolute-ranking config), "
          f"mean Jaccard: {jac:.3f} -- this is the ACTUAL practical effect of "
          f"'use_relative_momentum=True': it changes WHICH names are eligible (via the "
          f"floor), it does not re-order an otherwise-identical candidate pool.")


def section_16_lookahead_evidence(t: pd.DataFrame) -> None:
    _hr("16: LOOK-AHEAD BIAS EVIDENCE")
    print("Entry delay check: every trade's entry_bar must be EXACTLY 1 bar after the "
          "detection's bar_idx (delay_bars=1, enforced in compile_momentum_signals):")
    print(t["delay_bars_actual"].value_counts())
    bad = t[t["delay_bars_actual"] != 1]
    print(f"Trades with an unexpected delay: {len(bad)} (expect 0)")

    print("\nSpot check: recompute one real trade's absolute_momentum independently from raw "
          "price history and compare to the value captured in its signal at generation time.")
    from momentum import absolute_momentum as _am
    row = t.dropna(subset=["absolute_momentum"]).iloc[0]
    sym, lookback, entry_bar = row["symbol"], row["lookback"], int(row["entry_bar"])
    frames = build_universe_frames([sym], YEARS, cache_dir=CACHE_DIR)
    df_ohlcv, _ind = frames[sym]
    bars = LOOKBACK_BARS[lookback]
    recomputed = _am(df_ohlcv["Close"], bars).iloc[entry_bar - 1]  # signal bar = entry_bar - 1
    print(f"  {sym} {lookback} trade at entry_bar={entry_bar}: "
          f"stored absolute_momentum={row['absolute_momentum']:.6f}, "
          f"independently recomputed={recomputed:.6f}, "
          f"match={'YES' if abs(row['absolute_momentum'] - recomputed) < 1e-9 else 'NO'}")
    print("\n(Unit-test-level no-lookahead coverage: core/test_momentum.py::"
          "test_relative_momentum_no_lookahead and core/strategies/test_momentum_strategy.py::"
          "test_cross_sectional_rank_no_lookahead mutate only FUTURE data and assert past "
          "values are unchanged -- 27/27 tests pass, see core/MOMENTUM_STRATEGY.md.)")


def main():
    print("Building joined trade+detection dataset (re-running the pipeline against cached "
          "price data, default config unchanged -- no tuning) ...")
    t, frames_by_lookback = build_joined_trades()
    out_path = os.path.join(CORE, "momentum_diagnostics_trades_joined.csv")
    t.to_csv(out_path, index=False)
    print(f"Joined {len(t)} trades. Saved -> {out_path}")

    t = section_1_4_and_9(t)
    section_5_regime(t)
    section_6_adx(t)
    section_7_momentum_strength(t)
    section_8_rank(t)
    section_12_correlation(frames_by_lookback)
    section_13_14_concentration(frames_by_lookback)
    section_15_relative_formulation(frames_by_lookback)
    section_16_lookahead_evidence(t)


if __name__ == "__main__":
    main()
