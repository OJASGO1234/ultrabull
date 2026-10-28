# core/audit_benchmark_control_strategies.py
"""
Benchmark/control-strategy audit: does the severe UltraBull validation
deterioration reflect BROAD market deterioration (simple directional
strategies generally struggled in the same window), or is it UNUSUALLY
SEVERE to UltraBull specifically?

Four simple, FIXED, textbook control strategies, none tuned or searched on
this data (every parameter below is a standard convention: Donchian
20/10 breakout, Wilder RSI 30/70 levels with a 50 midline exit, classic
MA20/MA50 crossover):
  1. Buy-and-hold
  2. MA20/MA50 trend-following (crossover)
  3. Simple breakout (Donchian 20-day entry / 10-day exit)
  4. Simple RSI mean-reversion (RSI<30 entry / RSI>50 exit)

All four reuse the EXACT SAME data as UltraBull's own baseline audit: the
same 37-symbol universe (event_study.UNIVERSE), the same per-symbol
OHLCV/indicator pipeline (df_ohlcv/ind_df come straight from
audit_baseline_backtest.run_symbol()'s own return values -- i.e. the real
production compute_indicators()/build_ind_df_from_vbt() output, not a
separate recomputation), and the same 70/30 chronological per-symbol split
used by every walk-forward script in this series. Only the ENTRY/EXIT
RULES differ -- these are NOT UltraBull's patterns/gates and do not use
ATR-based sizing, SL/TP, gates, or cost/slippage modeling; each control
takes one position at a time, sized at full notional (shares =
capital/entry_price, capital=1,000,000, no leverage), with a fixed 1-bar
execution delay (signal observed at bar t's close, executed at bar t+1's
open) to avoid look-ahead -- kept deliberately simple, since the point is
a ROUGH benchmark, not a competing strategy.

Does NOT modify any UltraBull production file or parameter. Does NOT
search for or tune any control-strategy threshold using validation data --
every threshold (20/10/30/50/20/50) is a pre-existing textbook convention
fixed before looking at any result.

Run:  python audit_benchmark_control_strategies.py <cache_dir>
Output: core/audit_benchmark_control_strategies_results.csv (+ printed report)
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

from event_study import UNIVERSE
from audit_baseline_backtest import build_production_cfg, run_symbol

SPLIT = 0.7
CAPITAL = 1_000_000.0
OUT_ROWS = []


def log(section, control, rows):
    for r in rows:
        d = dict(r); d["section"] = section; d["control"] = control
        OUT_ROWS.append(d)


def summary_stats(g: pd.DataFrame) -> dict:
    if len(g) == 0:
        return {"n": 0, "win_rate": np.nan, "profit_factor": np.nan, "expectancy": np.nan, "total_pnl": 0.0}
    wins, losses = g[g.pnl > 0], g[g.pnl <= 0]
    gp, gl = float(wins["pnl"].sum()), -float(losses["pnl"].sum())
    return {
        "n": len(g), "win_rate": float((g.pnl > 0).mean()),
        "profit_factor": (gp / gl) if gl > 0 else np.inf,
        "expectancy": float(g["pnl"].mean()), "total_pnl": float(g["pnl"].sum()),
    }


def make_trade(symbol, entry_bar, exit_bar, entry_px, exit_px, n, split_bar):
    shares = CAPITAL / entry_px
    pnl = (exit_px - entry_px) * shares
    return {"symbol": symbol, "entry_bar": entry_bar, "exit_bar": exit_bar,
            "pnl": pnl, "period": "train" if entry_bar < split_bar else "valid"}


# ---------------------------------------------------------------------------
# control strategies -- fixed, textbook, non-optimized rules
# ---------------------------------------------------------------------------
def control_buy_and_hold(sym, df_ohlcv, ind_df, n, split_bar, warmup=60):
    close = df_ohlcv["Close"].to_numpy()
    trades = []
    start = min(warmup, n - 2)
    if split_bar > start + 1:
        trades.append(make_trade(sym, start, split_bar - 1, close[start], close[split_bar - 1], n, split_bar))
    if n - 1 > split_bar:
        trades.append(make_trade(sym, split_bar, n - 1, close[split_bar], close[n - 1], n, split_bar))
    return trades


def control_ma_crossover(sym, df_ohlcv, ind_df, n, split_bar, warmup=60):
    close = df_ohlcv["Close"].to_numpy()
    open_ = df_ohlcv["Open"].to_numpy()
    ma20 = ind_df["MA20"].to_numpy()
    ma50 = ind_df["MA50"].to_numpy()
    trades = []
    in_pos = False
    entry_bar = entry_px = None
    for t in range(max(1, warmup), n - 1):
        if not np.isfinite(ma20[t]) or not np.isfinite(ma50[t]) or not np.isfinite(ma20[t - 1]) or not np.isfinite(ma50[t - 1]):
            continue
        cross_up = ma20[t - 1] <= ma50[t - 1] and ma20[t] > ma50[t]
        cross_down = ma20[t - 1] >= ma50[t - 1] and ma20[t] < ma50[t]
        if not in_pos and cross_up:
            entry_bar, entry_px, in_pos = t + 1, float(open_[t + 1]), True
        elif in_pos and cross_down:
            trades.append(make_trade(sym, entry_bar, t + 1, entry_px, float(open_[t + 1]), n, split_bar))
            in_pos = False
    if in_pos:
        trades.append(make_trade(sym, entry_bar, n - 1, entry_px, close[n - 1], n, split_bar))
    return trades


def control_breakout(sym, df_ohlcv, ind_df, n, split_bar, warmup=60, entry_n=20, exit_n=10):
    high = df_ohlcv["High"].to_numpy()
    low = df_ohlcv["Low"].to_numpy()
    close = df_ohlcv["Close"].to_numpy()
    open_ = df_ohlcv["Open"].to_numpy()
    roll_high = pd.Series(high).rolling(entry_n).max().shift(1).to_numpy()
    roll_low = pd.Series(low).rolling(exit_n).min().shift(1).to_numpy()
    trades = []
    in_pos = False
    entry_bar = entry_px = None
    for t in range(max(entry_n, warmup), n - 1):
        if not np.isfinite(roll_high[t]):
            continue
        if not in_pos and close[t] > roll_high[t]:
            entry_bar, entry_px, in_pos = t + 1, float(open_[t + 1]), True
        elif in_pos and np.isfinite(roll_low[t]) and close[t] < roll_low[t]:
            trades.append(make_trade(sym, entry_bar, t + 1, entry_px, float(open_[t + 1]), n, split_bar))
            in_pos = False
    if in_pos:
        trades.append(make_trade(sym, entry_bar, n - 1, entry_px, close[n - 1], n, split_bar))
    return trades


def control_rsi_reversion(sym, df_ohlcv, ind_df, n, split_bar, warmup=60, buy_thr=30.0, sell_thr=50.0):
    close = df_ohlcv["Close"].to_numpy()
    open_ = df_ohlcv["Open"].to_numpy()
    rsi = ind_df["RSI"].to_numpy()
    trades = []
    in_pos = False
    entry_bar = entry_px = None
    for t in range(max(1, warmup), n - 1):
        if not np.isfinite(rsi[t]) or not np.isfinite(rsi[t - 1]):
            continue
        cross_below = rsi[t - 1] >= buy_thr and rsi[t] < buy_thr
        cross_above = rsi[t - 1] <= sell_thr and rsi[t] > sell_thr
        if not in_pos and cross_below:
            entry_bar, entry_px, in_pos = t + 1, float(open_[t + 1]), True
        elif in_pos and cross_above:
            trades.append(make_trade(sym, entry_bar, t + 1, entry_px, float(open_[t + 1]), n, split_bar))
            in_pos = False
    if in_pos:
        trades.append(make_trade(sym, entry_bar, n - 1, entry_px, close[n - 1], n, split_bar))
    return trades


CONTROLS = {
    "buy_and_hold": control_buy_and_hold,
    "ma20_ma50_crossover": control_ma_crossover,
    "breakout_donchian_20_10": control_breakout,
    "rsi_reversion_30_50": control_rsi_reversion,
}


def run_control(control_name, fn, cache):
    cfg = build_production_cfg()
    all_trades = []
    bad = []
    for sym in UNIVERSE:
        r = run_symbol(sym, cfg, cache)  # reuses the exact production OHLCV/indicator pipeline
        if r["error"]:
            bad.append(r)
            continue
        df_ohlcv, ind_df = r["df_ohlcv"], r["ind_df"]
        n = len(df_ohlcv)
        split_bar = int(n * SPLIT)
        trades = fn(sym, df_ohlcv, ind_df, n, split_bar)
        for t in trades:
            t["control"] = control_name
            t["entry_date"] = df_ohlcv.index[min(t["entry_bar"], n - 1)]
        all_trades.extend(trades)
    return pd.DataFrame(all_trades), bad


def main():
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    pd.set_option("display.width", 240)

    # reference figures from the already-completed UltraBull baseline audit (not
    # recomputed here -- pulled from audit_trade_distribution_diagnosis.py's own
    # printed output in this conversation, for direct side-by-side comparison)
    ultrabull_ref = {
        "control": "ultrabull_baseline",
        "n_train": 744, "n_valid": 307,
        "win_rate_train": 0.461, "win_rate_valid": 0.332,
        "profit_factor_train": 1.422, "profit_factor_valid": 0.668,
        "expectancy_train": 871.36, "expectancy_valid": -986.26,
        "total_pnl_train": 648289.75, "total_pnl_valid": -302781.94,
    }

    comparison_rows = [ultrabull_ref]

    for name, fn in CONTROLS.items():
        print("\n" + "=" * 100)
        print(f"CONTROL: {name}")
        print("=" * 100)
        trades, bad = run_control(name, fn, cache)
        if len(trades) == 0:
            print("  no trades generated -- skipping")
            continue
        train, valid = trades[trades.period == "train"], trades[trades.period == "valid"]
        s_train, s_valid = summary_stats(train), summary_stats(valid)
        print(f"Training:   {s_train}")
        print(f"Validation: {s_valid}")
        log("TRAIN_VALID", name, [{"period": "train", **s_train}, {"period": "valid", **s_valid}])

        delta_pf = s_valid["profit_factor"] - s_train["profit_factor"] if np.isfinite(s_valid.get("profit_factor", np.nan)) and np.isfinite(s_train.get("profit_factor", np.nan)) else np.nan
        delta_exp = s_valid["expectancy"] - s_train["expectancy"]
        print(f"Delta: PF {delta_pf}, expectancy {delta_exp:.2f}, total_pnl {s_valid['total_pnl'] - s_train['total_pnl']:.2f}")

        comparison_rows.append({
            "control": name, "n_train": s_train["n"], "n_valid": s_valid["n"],
            "win_rate_train": s_train["win_rate"], "win_rate_valid": s_valid["win_rate"],
            "profit_factor_train": s_train["profit_factor"], "profit_factor_valid": s_valid["profit_factor"],
            "expectancy_train": s_train["expectancy"], "expectancy_valid": s_valid["expectancy"],
            "total_pnl_train": s_train["total_pnl"], "total_pnl_valid": s_valid["total_pnl"],
        })

        # chronological deterioration: quarterly total_pnl
        trades["quarter"] = pd.to_datetime(trades["entry_date"]).dt.to_period("Q")
        q = trades.groupby("quarter").apply(lambda g: pd.Series(summary_stats(g)), include_groups=False)
        print("\nQuarterly (chronological deterioration check):")
        print(q[["n", "win_rate", "profit_factor", "expectancy", "total_pnl"]].round(2).to_string())
        log("QUARTERLY", name, [{"quarter": str(qq), **row} for qq, row in q.to_dict("index").items()])

        # equal chronological quartile blocks
        def block_of(row, n_bars_by_symbol):
            nn = n_bars_by_symbol[row["symbol"]]
            frac = row["entry_bar"] / max(1, nn)
            return min(3, int(frac * 4))
        n_bars_by_symbol = {s: int(trades[trades.symbol == s]["exit_bar"].max()) + 1 for s in trades["symbol"].unique()}
        trades["block"] = trades.apply(lambda r: block_of(r, n_bars_by_symbol), axis=1)
        blocks = trades.groupby("block").apply(lambda g: pd.Series(summary_stats(g)), include_groups=False)
        print("\nEqual chronological quartile blocks:")
        print(blocks[["n", "win_rate", "profit_factor", "expectancy", "total_pnl"]].round(2).to_string())
        log("CHRONOLOGICAL_BLOCKS", name, [{"block": int(b), **row} for b, row in blocks.to_dict("index").items()])

        if bad:
            print(f"\n{len(bad)} symbols had data issues")

    comp_df = pd.DataFrame(comparison_rows)
    comp_df["delta_profit_factor"] = comp_df["profit_factor_valid"] - comp_df["profit_factor_train"]
    comp_df["delta_expectancy"] = comp_df["expectancy_valid"] - comp_df["expectancy_train"]
    comp_df["delta_win_rate"] = comp_df["win_rate_valid"] - comp_df["win_rate_train"]
    comp_df["pf_ratio_valid_over_train"] = comp_df["profit_factor_valid"] / comp_df["profit_factor_train"]

    print("\n" + "=" * 100)
    print("FINAL COMPARISON: UltraBull vs all 4 controls (broad market decline, or UltraBull-specific?)")
    print("=" * 100)
    print(comp_df[["control", "n_train", "n_valid", "win_rate_train", "win_rate_valid",
                    "profit_factor_train", "profit_factor_valid", "delta_profit_factor",
                    "pf_ratio_valid_over_train", "expectancy_train", "expectancy_valid", "delta_expectancy"]]
          .round(4).to_string(index=False))
    log("FINAL_COMPARISON", "ALL", [{**r} for r in comp_df.to_dict("records")])

    out_df = pd.DataFrame(OUT_ROWS)
    out_df.to_csv("audit_benchmark_control_strategies_results.csv", index=False)
    print(f"\nSaved: audit_benchmark_control_strategies_results.csv ({len(out_df)} rows)")
    print("\nNOTE: all control thresholds are fixed textbook conventions (Donchian 20/10, RSI 30/50, "
          "MA20/MA50), none searched or tuned on this data, none using validation data to choose "
          "anything. Diagnosis only -- no UltraBull production file/parameter changed.")


if __name__ == "__main__":
    main()
