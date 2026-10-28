# core/audit_trade_distribution_diagnosis.py
"""
Trade-return distribution diagnosis: HOW did the edge disappear (lower win
rate? smaller winners? larger losers? different exit mix? clustering in
time?), not a search for a new rule. Training vs validation, plus
finer-grained chronological blocks across the full 4-year period.

Uses the exact baseline config (build_production_cfg() / run_symbol() from
audit_baseline_backtest.py, imported not retyped) and the same 70/30
chronological split used by every prior walk-forward script in this
series. MAE/MFE are NOT columns simulate() produces -- they are computed
HERE, independently, from each trade's own entry_bar/exit_bar and the
symbol's own High/Low arrays (both already returned by run_symbol()'s
df_ohlcv) -- this is a read-only post-hoc computation on simulate()'s own
output, not a change to the engine.

Does NOT modify any production file. Does NOT optimize, tune, or propose
any new rule/threshold/gate. Diagnosis only.

Run:  python audit_trade_distribution_diagnosis.py <cache_dir>
Output: core/audit_trade_distribution_diagnosis_results.csv (+ printed report)
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

from event_study import UNIVERSE
from audit_baseline_backtest import build_production_cfg, run_symbol, YEARS

SPLIT = 0.7
OUT_ROWS = []


def log(section, rows):
    for r in rows:
        d = dict(r); d["section"] = section
        OUT_ROWS.append(d)


def compute_mae_mfe(t, high, low, n):
    eb = int(np.clip(t["entry_bar"], 0, n - 1))
    xb = int(np.clip(t["exit_bar"], 0, n - 1))
    lo, hi = min(eb, xb), max(eb, xb)
    window_high = high[lo:hi + 1]
    window_low = low[lo:hi + 1]
    entry = float(t["entry_price"])
    if t["side"] == "long":
        mfe_pct = (float(np.max(window_high)) - entry) / entry * 100.0
        mae_pct = (float(np.min(window_low)) - entry) / entry * 100.0  # negative = adverse
    else:
        mfe_pct = (entry - float(np.min(window_low))) / entry * 100.0
        mae_pct = (entry - float(np.max(window_high))) / entry * 100.0  # negative = adverse
    return mae_pct, mfe_pct


def collect(cache):
    cfg = build_production_cfg()
    rows = []
    bad = []
    for sym in UNIVERSE:
        r = run_symbol(sym, cfg, cache)
        if r["error"]:
            bad.append(r)
            continue
        trades = r["trades"]
        if len(trades) == 0:
            continue
        df_ohlcv = r["df_ohlcv"]
        n = len(df_ohlcv)
        split_bar = int(n * SPLIT)
        high = df_ohlcv["High"].to_numpy()
        low = df_ohlcv["Low"].to_numpy()
        dates = df_ohlcv.index

        for _, t in trades.iterrows():
            eb = int(np.clip(t["entry_bar"], 0, n - 1))
            xb = int(np.clip(t["exit_bar"], 0, n - 1))
            mae_pct, mfe_pct = compute_mae_mfe(t, high, low, n)
            rows.append({
                "symbol": sym, "tag": t["tag"], "side": t["side"],
                "entry_bar": eb, "exit_bar": xb, "hold_bars": xb - eb,
                "entry_date": dates[eb], "exit_date": dates[xb],
                "period": "train" if eb < split_bar else "valid",
                "pnl": float(t["pnl"]), "reason": t["reason"],
                "entry_price": float(t["entry_price"]), "exit_price": float(t["exit_price"]),
                "trail_used": pd.notna(t.get("trail_price_exit", np.nan)),
                "mae_pct": mae_pct, "mfe_pct": mfe_pct,
                "win": bool(t["pnl"] > 0),
            })
    return pd.DataFrame(rows), bad


def summary_stats(g: pd.DataFrame) -> dict:
    if len(g) == 0:
        return {"n": 0}
    wins, losses = g[g.win], g[~g.win]
    gp, gl = float(wins["pnl"].sum()), -float(losses["pnl"].sum())
    return {
        "n": len(g),
        "win_rate": float(g["win"].mean()),
        "profit_factor": (gp / gl) if gl > 0 else np.inf,
        "expectancy": float(g["pnl"].mean()),
        "total_pnl": float(g["pnl"].sum()),
        "avg_win": float(wins["pnl"].mean()) if len(wins) else 0.0,
        "avg_loss": float(losses["pnl"].mean()) if len(losses) else 0.0,
        "median_win": float(wins["pnl"].median()) if len(wins) else 0.0,
        "median_loss": float(losses["pnl"].median()) if len(losses) else 0.0,
        "win_loss_ratio": (abs(float(wins["pnl"].mean()) / float(losses["pnl"].mean()))
                            if len(losses) and len(wins) and losses["pnl"].mean() != 0 else np.nan),
        "avg_hold_bars": float(g["hold_bars"].mean()),
        "median_hold_bars": float(g["hold_bars"].median()),
        "avg_mae_pct": float(g["mae_pct"].mean()),
        "avg_mfe_pct": float(g["mfe_pct"].mean()),
        "median_mae_pct": float(g["mae_pct"].median()),
        "median_mfe_pct": float(g["mfe_pct"].median()),
    }


def max_consecutive(seq: list) -> dict:
    """seq: list of bool (win=True). Returns max win streak, max loss streak,
    avg win streak, avg loss streak."""
    if not seq:
        return {"max_win_streak": 0, "max_loss_streak": 0, "avg_win_streak": 0.0, "avg_loss_streak": 0.0}
    win_streaks, loss_streaks = [], []
    cur, cur_val = 0, None
    for v in seq:
        if v == cur_val:
            cur += 1
        else:
            if cur_val is True:
                win_streaks.append(cur)
            elif cur_val is False:
                loss_streaks.append(cur)
            cur_val, cur = v, 1
    if cur_val is True:
        win_streaks.append(cur)
    elif cur_val is False:
        loss_streaks.append(cur)
    return {
        "max_win_streak": max(win_streaks) if win_streaks else 0,
        "max_loss_streak": max(loss_streaks) if loss_streaks else 0,
        "avg_win_streak": float(np.mean(win_streaks)) if win_streaks else 0.0,
        "avg_loss_streak": float(np.mean(loss_streaks)) if loss_streaks else 0.0,
    }


def consecutive_analysis(trades: pd.DataFrame, period: str) -> dict:
    """Streaks computed PER SYMBOL (the only real chronological sequence that
    exists, since each symbol is an independent backtest) then aggregated."""
    g = trades[trades.period == period]
    per_symbol = []
    for sym, sg in g.sort_values("entry_bar").groupby("symbol"):
        per_symbol.append(max_consecutive(list(sg["win"])))
    if not per_symbol:
        return {"max_win_streak": np.nan, "max_loss_streak": np.nan,
                "avg_win_streak": np.nan, "avg_loss_streak": np.nan}
    df = pd.DataFrame(per_symbol)
    return {
        "max_win_streak": int(df["max_win_streak"].max()),
        "max_loss_streak": int(df["max_loss_streak"].max()),
        "avg_win_streak": float(df["avg_win_streak"].mean()),
        "avg_loss_streak": float(df["avg_loss_streak"].mean()),
    }


def decompose_expectancy(train_stats: dict, valid_stats: dict) -> dict:
    """Exact, additive decomposition of the change in expectancy into a
    win-rate effect, an avg-win effect, and an avg-loss effect, via sequential
    counterfactual substitution (order: win_rate -> avg_win -> avg_loss).
    This ordering is one valid choice among several (the decomposition is not
    unique when all three terms move at once) but is exact and additive."""
    p_t, w_t, l_t = train_stats["win_rate"], train_stats["avg_win"], train_stats["avg_loss"]
    p_v, w_v, l_v = valid_stats["win_rate"], valid_stats["avg_win"], valid_stats["avg_loss"]

    e_train = p_t * w_t + (1 - p_t) * l_t
    e_cf1 = p_v * w_t + (1 - p_v) * l_t       # only win_rate changed
    e_cf2 = p_v * w_v + (1 - p_v) * l_t       # win_rate + avg_win changed
    e_valid = p_v * w_v + (1 - p_v) * l_v     # all three changed (actual)

    return {
        "expectancy_train": e_train, "expectancy_valid": e_valid,
        "total_change": e_valid - e_train,
        "effect_win_rate": e_cf1 - e_train,
        "effect_avg_win": e_cf2 - e_cf1,
        "effect_avg_loss": e_valid - e_cf2,
    }


def main():
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    print("Collecting trades + independently computed MAE/MFE (from entry/exit bars and "
          "each symbol's own High/Low -- not a simulate() output, computed here read-only) "
          "for the full baseline strategy, same config/split as every prior script.\n")

    trades, bad = collect(cache)
    print(f"Total trades: {len(trades)} ({len(bad)} data issues)")
    train, valid = trades[trades.period == "train"], trades[trades.period == "valid"]
    print(f"Training: {len(train)}, Validation: {len(valid)}")

    pd.set_option("display.width", 240)

    # ---- 1. overall train vs valid summary ----
    s_train, s_valid = summary_stats(train), summary_stats(valid)
    print("\n" + "=" * 100)
    print("1. OVERALL TRAIN vs VALID SUMMARY (trade-pool, not portfolio)")
    print("=" * 100)
    cmp_df = pd.DataFrame({"train": s_train, "valid": s_valid})
    cmp_df["delta"] = cmp_df["valid"] - cmp_df["train"]
    print(cmp_df.round(4).to_string())
    log("OVERALL_SUMMARY", [{"period": "train", **s_train}, {"period": "valid", **s_valid}])

    # ---- 2. exit reason distribution, with trail usage noted ----
    print("\n" + "=" * 100)
    print("2. EXIT REASON DISTRIBUTION (SL/TP/TIME, with gap variants and trail usage)")
    print("=" * 100)
    for period, g in (("train", train), ("valid", valid)):
        dist = g["reason"].value_counts()
        dist_pct = (dist / len(g) * 100).round(1)
        trail_among_sl = g[g["reason"].isin(["SL", "SL_GAP"]) & g["trail_used"]]
        print(f"\n[{period}] n={len(g)}")
        print(pd.DataFrame({"count": dist, "pct": dist_pct}).to_string())
        print(f"  of which SL exits that used a trailing stop (trail_price_exit set): "
              f"{len(trail_among_sl)} / {len(g[g['reason'].isin(['SL','SL_GAP'])])}")
        for reason, cnt in dist.items():
            log("EXIT_REASON_DIST", [{"period": period, "reason": reason, "count": int(cnt), "pct": float(dist_pct[reason])}])

    # ---- 3. winner size vs loser size ----
    print("\n" + "=" * 100)
    print("3. WINNER SIZE vs LOSER SIZE")
    print("=" * 100)
    size_df = pd.DataFrame({
        "train": {k: s_train[k] for k in ["avg_win", "avg_loss", "median_win", "median_loss", "win_loss_ratio"]},
        "valid": {k: s_valid[k] for k in ["avg_win", "avg_loss", "median_win", "median_loss", "win_loss_ratio"]},
    })
    size_df["delta"] = size_df["valid"] - size_df["train"]
    print(size_df.round(2).to_string())
    log("WIN_LOSS_SIZE", [{"period": "train", **{k: s_train[k] for k in ["avg_win","avg_loss","median_win","median_loss","win_loss_ratio"]}},
                            {"period": "valid", **{k: s_valid[k] for k in ["avg_win","avg_loss","median_win","median_loss","win_loss_ratio"]}}])

    # ---- 4. MAE / MFE ----
    print("\n" + "=" * 100)
    print("4. MAE (max adverse excursion %) / MFE (max favorable excursion %), winners vs losers")
    print("=" * 100)
    for period, g in (("train", train), ("valid", valid)):
        w, l = g[g.win], g[~g.win]
        rows = [
            {"period": period, "group": "winners", "n": len(w),
             "avg_mae_pct": w["mae_pct"].mean(), "avg_mfe_pct": w["mfe_pct"].mean(),
             "median_mae_pct": w["mae_pct"].median(), "median_mfe_pct": w["mfe_pct"].median()},
            {"period": period, "group": "losers", "n": len(l),
             "avg_mae_pct": l["mae_pct"].mean(), "avg_mfe_pct": l["mfe_pct"].mean(),
             "median_mae_pct": l["mae_pct"].median(), "median_mfe_pct": l["mfe_pct"].median()},
        ]
        print(pd.DataFrame(rows).round(3).to_string(index=False))
        log("MAE_MFE", rows)

    # ---- 5. holding period ----
    print("\n" + "=" * 100)
    print("5. HOLDING PERIOD (bars)")
    print("=" * 100)
    hold_df = pd.DataFrame({
        "train": train["hold_bars"].describe(),
        "valid": valid["hold_bars"].describe(),
    })
    print(hold_df.round(2).to_string())
    log("HOLDING_PERIOD", [{"period": "train", **train["hold_bars"].describe().to_dict()},
                             {"period": "valid", **valid["hold_bars"].describe().to_dict()}])

    # ---- 6. consecutive win/loss streaks (per symbol, aggregated) ----
    print("\n" + "=" * 100)
    print("6. CONSECUTIVE WINS/LOSSES (computed per symbol's own chronological trade sequence)")
    print("=" * 100)
    ca_train, ca_valid = consecutive_analysis(trades, "train"), consecutive_analysis(trades, "valid")
    print(pd.DataFrame({"train": ca_train, "valid": ca_valid}).to_string())
    log("CONSECUTIVE_STREAKS", [{"period": "train", **ca_train}, {"period": "valid", **ca_valid}])

    # ---- 7. monthly / quarterly performance across the full 4-year period ----
    print("\n" + "=" * 100)
    print("7. MONTHLY PERFORMANCE, FULL 4-YEAR PERIOD (pooled across all symbols)")
    print("=" * 100)
    trades["month"] = pd.to_datetime(trades["entry_date"]).dt.to_period("M")
    monthly = trades.groupby("month").apply(lambda g: pd.Series(summary_stats(g)), include_groups=False)
    print(monthly[["n", "win_rate", "profit_factor", "expectancy", "total_pnl"]].round(3).to_string())
    log("MONTHLY_PERFORMANCE", [{"month": str(m), **row} for m, row in monthly.to_dict("index").items()])

    print("\n" + "=" * 100)
    print("QUARTERLY PERFORMANCE, FULL 4-YEAR PERIOD")
    print("=" * 100)
    trades["quarter"] = pd.to_datetime(trades["entry_date"]).dt.to_period("Q")
    quarterly = trades.groupby("quarter").apply(lambda g: pd.Series(summary_stats(g)), include_groups=False)
    print(quarterly[["n", "win_rate", "profit_factor", "expectancy", "total_pnl"]].round(3).to_string())
    log("QUARTERLY_PERFORMANCE", [{"quarter": str(q), **row} for q, row in quarterly.to_dict("index").items()])

    # ---- equal chronological blocks (quartiles by entry_bar, relative per symbol) ----
    print("\n" + "=" * 100)
    print("EQUAL CHRONOLOGICAL BLOCKS (4 quartiles of each symbol's own bar range, pooled)")
    print("=" * 100)
    def block_of(row, n_bars_by_symbol):
        n = n_bars_by_symbol[row["symbol"]]
        frac = row["entry_bar"] / max(1, n)
        return min(3, int(frac * 4))
    n_bars_by_symbol = {}
    for sym in trades["symbol"].unique():
        sub = trades[trades.symbol == sym]
        n_bars_by_symbol[sym] = int(sub["exit_bar"].max()) + 1
    trades["block"] = trades.apply(lambda r: block_of(r, n_bars_by_symbol), axis=1)
    block_stats = trades.groupby("block").apply(lambda g: pd.Series(summary_stats(g)), include_groups=False)
    print(block_stats[["n", "win_rate", "profit_factor", "expectancy", "total_pnl"]].round(3).to_string())
    log("CHRONOLOGICAL_QUARTILES", [{"block": int(b), **row} for b, row in block_stats.to_dict("index").items()])

    # ---- 8. decomposition: where did the edge go? ----
    print("\n" + "=" * 100)
    print("8. DECOMPOSITION: how much of the expectancy decline is win-rate vs avg-win vs avg-loss?")
    print("(sequential counterfactual substitution, order: win_rate -> avg_win -> avg_loss; exact & additive)")
    print("=" * 100)
    decomp = decompose_expectancy(s_train, s_valid)
    print(pd.Series(decomp).round(2).to_string())
    total = decomp["total_change"]
    for k in ["effect_win_rate", "effect_avg_win", "effect_avg_loss"]:
        pct = decomp[k] / total * 100 if total != 0 else np.nan
        print(f"  {k}: {decomp[k]:.2f}  ({pct:.1f}% of total expectancy change)")
    log("EXPECTANCY_DECOMPOSITION", [decomp])

    if bad:
        print(f"\n{len(bad)} symbols had data issues: {[b['symbol'] for b in bad]}")

    out_df = pd.DataFrame(OUT_ROWS)
    out_df.to_csv("audit_trade_distribution_diagnosis_results.csv", index=False)
    print(f"\nSaved: audit_trade_distribution_diagnosis_results.csv ({len(out_df)} rows)")
    print("\nNOTE: diagnosis only. No threshold searched for or tuned, no production file/parameter changed.")


if __name__ == "__main__":
    main()
