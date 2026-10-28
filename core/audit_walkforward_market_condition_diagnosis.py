# core/audit_walkforward_market_condition_diagnosis.py
"""
Follow-up to audit_walkforward_pattern_diagnosis.py: that script found the
out-of-sample decline is concentrated in Hammer, BearishEngulfing, and
PullbackToMA20 (86% of the total P&L decline). This script asks WHY, by
attaching each of those patterns' trades to the market/indicator conditions
that already existed at entry time (no new features invented) and
comparing:
  (a) training-period vs validation-period entry conditions, and
  (b) winning-trade vs losing-trade entry conditions,
for each of the three patterns.

Features used (all already computed by production code, nothing new):
  - market regime (mkt_up/mkt_dn from market_context.build_market_context)
  - ATR, and ATR as a % of entry price (volatility, normalized for cross-
    symbol comparability)
  - ADX (trend strength) from ind_df
  - RSI from ind_df
  - MA20 vs MA50 spread, as (MA20-MA50)/MA50*100 (%) -- the same quantity
    gates.py's trend conditions threshold at zero
  - relative strength (rs_20) and distribution-days/breadth from ctx --
    already used by the market/RS/breadth gates, included since the
    prompt asked for "any other existing features already available"

Reuses build_production_cfg()/run_symbol() from audit_baseline_backtest.py
and market_context.build_market_context() exactly as run_symbol calls it
internally (run_symbol's own return dict doesn't carry ctx, so it is
recomputed here with IDENTICAL arguments -- same function, same cache,
same years -- not a different calculation).

Does NOT modify any production file. Does NOT search for thresholds, tune
anything, or change any parameter. Diagnosis only.

Run:  python audit_walkforward_market_condition_diagnosis.py <cache_dir>
Output: core/audit_walkforward_market_condition_results.csv (+ printed report)
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

from event_study import UNIVERSE
from audit_baseline_backtest import build_production_cfg, run_symbol, YEARS
from market_context import build_market_context

SPLIT = 0.7
FOCUS_PATTERNS = ["Hammer", "BearishEngulfing", "PullbackToMA20"]

FEATURES = ["mkt_regime", "atr_pct", "adx", "rsi", "ma_spread_pct", "rs_20", "dist_days", "breadth"]


def collect_trade_features(cache: str | None) -> tuple[pd.DataFrame, list]:
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
        df_ohlcv, ind_df = r["df_ohlcv"], r["ind_df"]
        n = len(df_ohlcv)
        split_bar = int(n * SPLIT)

        ctx = build_market_context(sym, df_ohlcv["Close"], years=YEARS, cache_dir=cache)

        close = df_ohlcv["Close"].to_numpy()
        atr = ind_df["ATR"].to_numpy() if "ATR" in ind_df.columns else np.full(n, np.nan)
        adx = ind_df["ADX"].to_numpy() if "ADX" in ind_df.columns else np.full(n, np.nan)
        rsi = ind_df["RSI"].to_numpy() if "RSI" in ind_df.columns else np.full(n, np.nan)
        ma20 = ind_df["MA20"].to_numpy() if "MA20" in ind_df.columns else np.full(n, np.nan)
        ma50 = ind_df["MA50"].to_numpy() if "MA50" in ind_df.columns else np.full(n, np.nan)
        mkt_up = ctx["mkt_up"].to_numpy() if "mkt_up" in ctx.columns else np.full(n, np.nan)
        mkt_dn = ctx["mkt_dn"].to_numpy() if "mkt_dn" in ctx.columns else np.full(n, np.nan)
        rs_20 = ctx["rs_20"].to_numpy() if "rs_20" in ctx.columns else np.full(n, np.nan)
        dist_days = ctx["dist_days"].to_numpy() if "dist_days" in ctx.columns else np.full(n, np.nan)
        breadth = ctx["breadth"].to_numpy() if "breadth" in ctx.columns else np.full(n, np.nan)

        for _, t in trades.iterrows():
            eb = int(np.clip(t["entry_bar"], 0, n - 1))
            mkt_regime = "uptrend" if bool(mkt_up[eb]) else ("downtrend" if bool(mkt_dn[eb]) else "neutral")
            c = close[eb]
            ma_spread_pct = ((ma20[eb] - ma50[eb]) / ma50[eb] * 100.0) if ma50[eb] not in (0, np.nan) and np.isfinite(ma50[eb]) and ma50[eb] != 0 else np.nan
            rows.append({
                "symbol": sym, "tag": t["tag"], "side": t["side"], "entry_bar": eb,
                "period": "train" if eb < split_bar else "valid",
                "pnl": t["pnl"], "win": bool(t["pnl"] > 0), "reason": t["reason"],
                "mkt_regime": mkt_regime,
                "atr_pct": (float(atr[eb]) / c * 100.0) if c and np.isfinite(atr[eb]) else np.nan,
                "adx": float(adx[eb]) if np.isfinite(adx[eb]) else np.nan,
                "rsi": float(rsi[eb]) if np.isfinite(rsi[eb]) else np.nan,
                "ma_spread_pct": ma_spread_pct,
                "rs_20": float(rs_20[eb]) if np.isfinite(rs_20[eb]) else np.nan,
                "dist_days": float(dist_days[eb]) if np.isfinite(dist_days[eb]) else np.nan,
                "breadth": float(breadth[eb]) if np.isfinite(breadth[eb]) else np.nan,
            })
    return pd.DataFrame(rows), bad


def numeric_summary(g: pd.DataFrame, cols: list) -> dict:
    out = {}
    for c in cols:
        if c not in g.columns:
            continue
        s = pd.to_numeric(g[c], errors="coerce").dropna()
        out[f"{c}_mean"] = float(s.mean()) if len(s) else np.nan
        out[f"{c}_median"] = float(s.median()) if len(s) else np.nan
    return out


def regime_dist(g: pd.DataFrame) -> dict:
    if len(g) == 0:
        return {}
    return (g["mkt_regime"].value_counts(normalize=True) * 100.0).round(1).to_dict()


def main():
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    print("Collecting per-trade entry-time features (market regime, ATR%, ADX, RSI, "
          "MA20/MA50 spread, RS_20, distribution-days, breadth) for all closed trades, "
          "same config/universe/split as audit_walkforward_pattern_diagnosis.py.\n")

    trades, bad = collect_trade_features(cache)
    print(f"Collected features for {len(trades)} trades across {trades['symbol'].nunique()} symbols "
          f"({len(bad)} data issues)")

    numeric_cols = ["atr_pct", "adx", "rsi", "ma_spread_pct", "rs_20", "dist_days", "breadth"]
    pd.set_option("display.width", 240)

    out_rows = []

    for pat in FOCUS_PATTERNS:
        pt = trades[trades["tag"] == pat]
        print("\n" + "=" * 100)
        print(f"PATTERN: {pat}  (n={len(pt)}: train={len(pt[pt.period=='train'])}, valid={len(pt[pt.period=='valid'])})")
        print("=" * 100)

        # (a) train vs valid entry conditions
        tr, va = pt[pt.period == "train"], pt[pt.period == "valid"]
        print("\n-- (a) Training vs Validation entry conditions --")
        print("Market regime distribution (%):")
        print(f"  train: {regime_dist(tr)}")
        print(f"  valid: {regime_dist(va)}")
        row_a = {"section": "TRAIN_VS_VALID", "tag": pat,
                  "n_train": len(tr), "n_valid": len(va)}
        sum_tr, sum_va = numeric_summary(tr, numeric_cols), numeric_summary(va, numeric_cols)
        tbl_rows = []
        for c in numeric_cols:
            tbl_rows.append({
                "feature": c,
                "train_mean": sum_tr.get(f"{c}_mean"), "valid_mean": sum_va.get(f"{c}_mean"),
                "train_median": sum_tr.get(f"{c}_median"), "valid_median": sum_va.get(f"{c}_median"),
                "delta_mean": (sum_va.get(f"{c}_mean", np.nan) - sum_tr.get(f"{c}_mean", np.nan)),
            })
        tbl = pd.DataFrame(tbl_rows)
        print(tbl.round(3).to_string(index=False))
        for _, r in tbl.iterrows():
            d = r.to_dict(); d["section"] = "TRAIN_VS_VALID"; d["tag"] = pat
            out_rows.append(d)
        for k, v in regime_dist(tr).items():
            out_rows.append({"section": "REGIME_DIST", "tag": pat, "period": "train", "regime": k, "pct": v})
        for k, v in regime_dist(va).items():
            out_rows.append({"section": "REGIME_DIST", "tag": pat, "period": "valid", "regime": k, "pct": v})

        # (b) winners vs losers (pooled across both periods, and separately per period)
        print("\n-- (b) Winning vs Losing trades (pooled across train+valid) --")
        win, lose = pt[pt.win], pt[~pt.win]
        print(f"n_win={len(win)}, n_lose={len(lose)}")
        print("Market regime distribution (%):")
        print(f"  win : {regime_dist(win)}")
        print(f"  lose: {regime_dist(lose)}")
        sum_win, sum_lose = numeric_summary(win, numeric_cols), numeric_summary(lose, numeric_cols)
        tbl2_rows = []
        for c in numeric_cols:
            tbl2_rows.append({
                "feature": c,
                "win_mean": sum_win.get(f"{c}_mean"), "lose_mean": sum_lose.get(f"{c}_mean"),
                "win_median": sum_win.get(f"{c}_median"), "lose_median": sum_lose.get(f"{c}_median"),
                "delta_mean_lose_minus_win": (sum_lose.get(f"{c}_mean", np.nan) - sum_win.get(f"{c}_mean", np.nan)),
            })
        tbl2 = pd.DataFrame(tbl2_rows)
        print(tbl2.round(3).to_string(index=False))
        for _, r in tbl2.iterrows():
            d = r.to_dict(); d["section"] = "WIN_VS_LOSE"; d["tag"] = pat; out_rows.append(d)
        for k, v in regime_dist(win).items():
            out_rows.append({"section": "WIN_LOSE_REGIME_DIST", "tag": pat, "group": "win", "regime": k, "pct": v})
        for k, v in regime_dist(lose).items():
            out_rows.append({"section": "WIN_LOSE_REGIME_DIST", "tag": pat, "group": "lose", "regime": k, "pct": v})

        # (b2) win/lose split WITHIN each period, to see if the win/lose feature
        # gap itself changes between train and valid (not just overall level shift)
        print("\n-- (b2) Win vs Lose, split by period (does the win/lose gap itself change?) --")
        for period, g in (("train", tr), ("valid", va)):
            w, l = g[g.win], g[~g.win]
            s_w, s_l = numeric_summary(w, numeric_cols), numeric_summary(l, numeric_cols)
            print(f"  [{period}] n_win={len(w)}, n_lose={len(l)}")
            for c in numeric_cols:
                wm, lm = s_w.get(f"{c}_mean"), s_l.get(f"{c}_mean")
                print(f"    {c:15s} win_mean={wm!s:>10}  lose_mean={lm!s:>10}")
                out_rows.append({"section": "WIN_VS_LOSE_BY_PERIOD", "tag": pat, "period": period,
                                  "feature": c, "win_mean": wm, "lose_mean": lm})

    # ---- raw per-trade feature table, saved in full for further analysis ----
    trades_out = trades.copy()
    trades_out["section"] = "RAW_TRADE_FEATURES"
    for_csv = pd.concat([pd.DataFrame(out_rows), trades_out], ignore_index=True, sort=False)
    cols = ["section"] + [c for c in for_csv.columns if c != "section"]
    for_csv = for_csv[cols]
    for_csv.to_csv("audit_walkforward_market_condition_results.csv", index=False)

    print("\n" + "=" * 100)
    print("ALL-PATTERN CONTEXT (for comparison): pooled market-regime distribution, all trades")
    print("=" * 100)
    print(f"train: {regime_dist(trades[trades.period=='train'])}")
    print(f"valid: {regime_dist(trades[trades.period=='valid'])}")

    if bad:
        print(f"\n{len(bad)} symbols had data issues:")
        for r in bad:
            print(f"  {r['symbol']}: {r['error']}")

    print(f"\nSaved: audit_walkforward_market_condition_results.csv ({len(for_csv)} rows)")
    print("\nNOTE: diagnosis only. No threshold was searched for, no parameter was changed, "
          "no gate/pattern was modified based on these results.")


if __name__ == "__main__":
    main()
