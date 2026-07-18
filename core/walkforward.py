# core/walkforward.py
"""
Walk-forward preset validation across the universe.

Each preset is a fixed config (nothing is fitted), so we simulate the full 4y
per symbol and split resulting trades at the 70% date boundary:
  tune  — first 70% of bars: used to RANK presets
  valid — last 30% of bars: out-of-sample check of that ranking

A preset whose edge is real keeps its relative rank out-of-sample; one that
won in-sample by luck collapses. Only trust presets that hold up in `valid`.

Run:  python walkforward.py <cache_dir>   (from the core/ directory)
"""
from __future__ import annotations
import copy
import numpy as np
import pandas as pd

from event_study import UNIVERSE, prepare_symbol
from patterns.postprocess import enrich_detections
from rules.gates import apply_gates
from rules.signals import compile_signals
from sim.engine import simulate
from sim.tune import _preset_overrides, _deep_update
from market_context import build_market_context
from evaluate_gates import GATE_CFG, _load_events_cached

PRESETS = ["baseline", "mr_tighter_tp", "time_stop_10", "min_atr_p20",
           "trend_wider_sl", "cooldown_plus2"]

BASE_CFG = {
    "execution": {"delay_bars": 1, "enter_at": "Open", "max_open_positions": 1,
                  "cooldown_bars": 4, "max_hold_bars": 10},
    "risk": {"capital": 1_000_000.0, "max_risk_per_trade": 0.005,
             "position_sizing": "atr"},
    "filters": {"min_atr": 0.0, "warmup_cutoff_bars": 60},
    "patterns": {
        "Breakout20": {"side": "long",  "sl_atr": 1.5, "tp_atr": 3.0, "trail_atr": 1.0},
        "Breakout55": {"side": "long",  "sl_atr": 2.0, "tp_atr": 4.0, "trail_atr": 1.5},
        "PullbackToMA20": {"side": "long", "sl_atr": 1.2, "tp_atr": 2.4, "trail_atr": 1.0},
        "BB_BuyDip": {"side": "long",  "sl_atr": 1.2, "tp_atr": 1.6, "trail_atr": 1.0},
        "Hammer": {"side": "long", "sl_atr": 1.5, "tp_atr": 3.0},
        "BullishEngulfing": {"side": "long", "sl_atr": 1.5, "tp_atr": 3.0},
        "BearishEngulfing": {"side": "short", "sl_atr": 1.5, "tp_atr": 3.0},
        "MACD_Cross_Zeroish_Down": {"side": "short", "sl_atr": 1.5, "tp_atr": 3.0},
    },
    "defaults": {"sl_atr": 1.5, "tp_atr": 3.0, "trail_atr": 0.0},
    "costs": {"per_share": 0.0, "per_trade": 0.0, "bps_fee": 0.0, "slippage_bps": 2.0},
    "equity": {"mtm": False},
    # event-study-backed prunes (short side has no edge except BearishEngulfing)
    "disable_patterns": ["RSI_60_Down", "RSI_40_Up", "Doji", "MACD_Cross_Zeroish_Up",
                         "BB_SellFade", "Breakdown20", "Breakdown55"],
    "gates": GATE_CFG["gates"],
}


def run(years: float = 4, cache_dir: str | None = None, split: float = 0.7) -> pd.DataFrame:
    all_trades = []
    for sym in UNIVERSE:
        try:
            prep = prepare_symbol(sym, years=years, cache_dir=cache_dir)
            if prep is None:
                continue
            df_ohlcv, ind_df, dets = prep
            ctx = build_market_context(sym, df_ohlcv["Close"], years=years, cache_dir=cache_dir)
            det_e = enrich_detections(dets, df_ohlcv, ind_df, sym, ctx=ctx)
            try:
                sym_events = _load_events_cached(sym, df_ohlcv, cache_dir)
            except Exception:
                sym_events = None
            det_g = apply_gates(det_e, df_ohlcv, ind_df, BASE_CFG, ctx=ctx, events=sym_events)
            if det_g is None or len(det_g) == 0:
                continue
            split_bar = int(len(df_ohlcv) * split)

            for preset in PRESETS:
                cfg = _deep_update(BASE_CFG, _preset_overrides(preset, BASE_CFG, df_ohlcv, ind_df))
                sig = compile_signals(det_g, df_ohlcv, ind_df, cfg)
                trades, _, _ = simulate(df_ohlcv, sig, ind_df, cfg)
                if trades is None or len(trades) == 0:
                    continue
                t = trades.copy()
                t["symbol"] = sym
                t["preset"] = preset
                t["period"] = np.where(t["entry_bar"] < split_bar, "tune", "valid")
                all_trades.append(t)
        except Exception as e:
            print(f"  {sym}: FAILED ({e})")
    return pd.concat(all_trades, ignore_index=True)


def summarize_wf(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (preset, period), g in trades.groupby(["preset", "period"]):
        wins = g["pnl"] > 0
        gp = float(g.loc[wins, "pnl"].sum())
        gl = -float(g.loc[~wins, "pnl"].sum())
        rows.append({
            "preset": preset, "period": period, "trades": len(g),
            "win_rate": float(wins.mean()),
            "pf": (gp / gl) if gl > 0 else np.inf,
            "expectancy": float(g["pnl"].mean()),
            "total_pnl": float(g["pnl"].sum()),
        })
    out = pd.DataFrame(rows)
    wide = out.pivot(index="preset", columns="period",
                     values=["trades", "win_rate", "pf", "expectancy"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    return wide.sort_values("pf_tune", ascending=False)


if __name__ == "__main__":
    import sys
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    trades = run(cache_dir=cache)
    pd.set_option("display.width", 200)
    print(f"\n{len(trades)} trades across {trades['symbol'].nunique()} symbols, "
          f"{trades['preset'].nunique()} presets")
    print("\n=== Walk-forward: presets ranked by tune-period PF; check valid columns ===")
    print(summarize_wf(trades).round(3).to_string())
    trades.to_csv("walkforward_trades.csv", index=False)
