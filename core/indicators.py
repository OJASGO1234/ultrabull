# core/indicators.py
"""Plain-English "key indicators" — a deterministic read of the current bar's
technical posture, each flag tagged bullish / bearish / neutral with a one-line
why. Heuristics compute facts (see design principles); this is the at-a-glance
layer that sits above the fuller Advice synthesis.

Shared by the analyze view (a "Key indicators" card) and the holdings drawer.
Every threshold here echoes something the 4y event study validated or rejected
— e.g. RSI>70 tested NEGATIVE for fresh longs, the 40-65 band is the only one
with edge — so the explanations can state evidence, not vibes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Flags describing the tape rather than the stock — greyed out and excluded
# from the lean in the stock-only view, mirroring advice.ENV_FACTORS.
ENV_FLAGS = {"Market regime", "Distribution days"}


def _flag(label: str, state: str, detail: str, value: str | None = None) -> dict:
    # state in {"bullish", "bearish", "neutral"}
    return {"label": label, "state": state, "detail": detail, "value": value,
            "axis": "environment" if label in ENV_FLAGS else "setup"}


def build_key_indicators(
    df_ohlcv: pd.DataFrame,
    ind_df: pd.DataFrame,
    ctx: pd.DataFrame | None,
    levels: list[dict] | None,
    use_market_context: bool = True,
) -> dict:
    close = float(df_ohlcv["Close"].iloc[-1])
    atr = float(ind_df["ATR"].iloc[-1]) if np.isfinite(ind_df["ATR"].iloc[-1]) else close * 0.02
    rsi = float(ind_df["RSI"].iloc[-1]) if np.isfinite(ind_df["RSI"].iloc[-1]) else 50.0
    flags: list[dict] = []

    # --- RSI zone ---
    if rsi > 70:
        flags.append(_flag("RSI", "bearish",
                           f"RSI {rsi:.0f} — overbought. Fresh longs bought this hot tested NEGATIVE "
                           f"(−16bps/5d) in our 4-year study; momentum, but a poor entry.", f"{rsi:.0f}"))
    elif 40 <= rsi <= 65:
        flags.append(_flag("RSI", "bullish",
                           f"RSI {rsi:.0f} — the one momentum band with a tested edge (40–65): rising, "
                           f"not yet overheated.", f"{rsi:.0f}"))
    elif rsi < 30:
        flags.append(_flag("RSI", "neutral",
                           f"RSI {rsi:.0f} — oversold. Statistically stretched, but catching a falling knife "
                           f"needs a reversal bar first (a hammer, not just a low RSI).", f"{rsi:.0f}"))
    else:
        flags.append(_flag("RSI", "neutral", f"RSI {rsi:.0f} — neutral zone, no edge either way.", f"{rsi:.0f}"))

    # --- MACD vs signal ---
    if "MACD" in ind_df.columns and "MACD_signal" in ind_df.columns:
        macd = float(ind_df["MACD"].iloc[-1])
        sig = float(ind_df["MACD_signal"].iloc[-1])
        if np.isfinite(macd) and np.isfinite(sig):
            # detect a fresh cross in the last 3 bars for extra weight
            hist = ind_df["MACD"] - ind_df["MACD_signal"]
            crossed_up = len(hist) > 3 and hist.iloc[-1] > 0 and (hist.iloc[-4:-1] <= 0).any()
            crossed_dn = len(hist) > 3 and hist.iloc[-1] < 0 and (hist.iloc[-4:-1] >= 0).any()
            if macd > sig:
                flags.append(_flag("MACD", "bullish",
                                   "MACD is above its signal line" + (" — a fresh bullish cross this week (momentum turning up)."
                                    if crossed_up else " (up-momentum intact).")))
            else:
                flags.append(_flag("MACD", "bearish",
                                   "MACD is below its signal line" + (" — a fresh bearish cross this week (momentum rolling over)."
                                    if crossed_dn else " (down-momentum intact).")))

    # --- stock trend (MA20 vs MA50) ---
    if "MA20" in ind_df.columns and "MA50" in ind_df.columns:
        ma20 = float(ind_df["MA20"].iloc[-1])
        ma50 = float(ind_df["MA50"].iloc[-1])
        if np.isfinite(ma20) and np.isfinite(ma50):
            up = ma20 > ma50
            flags.append(_flag("Trend", "bullish" if up else "bearish",
                               f"20-day average is {'above' if up else 'below'} the 50-day "
                               f"— the stock's own trend is {'up' if up else 'down'}."))

    # --- volume jump (signed by the day's direction) ---
    vol = df_ohlcv["Volume"]
    if len(vol) > 20:
        volx = float(vol.iloc[-1] / max(1e-9, vol.rolling(20).mean().iloc[-1]))
        up_day = len(df_ohlcv) > 1 and df_ohlcv["Close"].iloc[-1] >= df_ohlcv["Close"].iloc[-2]
        if volx >= 2.0:
            flags.append(_flag("Volume", "bullish" if up_day else "bearish",
                               f"{volx:.1f}× the 20-day average volume on a {'up' if up_day else 'down'} day "
                               f"— conviction behind {'buyers' if up_day else 'sellers'}. (Note: 2× spikes alone "
                               f"tested edge-less; read it with the trend.)", f"{volx:.1f}×"))
        elif volx >= 1.5:
            flags.append(_flag("Volume", "neutral",
                               f"{volx:.1f}× the 20-day average — above normal but not a decisive spike.", f"{volx:.1f}×"))
        else:
            flags.append(_flag("Volume", "neutral", f"{volx:.1f}× 20-day average — unremarkable participation.", f"{volx:.1f}×"))

    # --- position vs resistance / support (breakout proximity) ---
    supports = sorted([lv for lv in (levels or []) if lv["kind"] == "support" and lv["price"] <= close],
                      key=lambda lv: -lv["price"])
    resistances = sorted([lv for lv in (levels or []) if lv["kind"] == "resistance" and lv["price"] >= close],
                         key=lambda lv: lv["price"])
    sup = supports[0] if supports else None
    res = resistances[0] if resistances else None
    if res is None and supports:
        flags.append(_flag("Levels", "bullish",
                           "Above every mapped resistance — blue-sky territory near the 52-week high, "
                           "no overhead supply on the chart."))
    elif res is not None:
        d_res_atr = (res["price"] - close) / atr
        d_res_pct = (res["price"] - close) / close * 100
        if d_res_atr <= 0.75:
            flags.append(_flag("Levels", "neutral",
                               f"Pressed against resistance ₹{res['price']:.0f} ({d_res_pct:+.1f}%, {res['touches']} touches) "
                               f"— a close above it is a breakout; getting rejected here is the risk. Watch this level."))
        elif sup and (close - sup["price"]) / atr <= 1.0:
            flags.append(_flag("Levels", "bullish",
                               f"Sitting on support ₹{sup['price']:.0f} ({sup['touches']} touches) — a defined-risk "
                               f"zone (stop just below), with room to ₹{res['price']:.0f}."))
        else:
            flags.append(_flag("Levels", "neutral",
                               f"Mid-range: ₹{sup['price']:.0f} support below, ₹{res['price']:.0f} resistance above "
                               f"({d_res_pct:+.1f}% away)." if sup
                               else f"Below resistance ₹{res['price']:.0f} ({d_res_pct:+.1f}% away), no mapped support beneath."))

    # --- 52-week high proximity ---
    if len(df_ohlcv) >= 60:
        look = min(len(df_ohlcv), 252)
        hi = float(df_ohlcv["High"].iloc[-look:].max())
        off_hi = (close / hi - 1) * 100
        if off_hi >= -2:
            flags.append(_flag("52-week high", "bullish",
                               f"Within {abs(off_hi):.1f}% of its 52-week high — leadership behaviour; "
                               f"stocks making new highs tend to keep leading (in an up-market).", f"{off_hi:+.1f}%"))
        elif off_hi <= -25:
            flags.append(_flag("52-week high", "bearish",
                               f"{abs(off_hi):.0f}% below its 52-week high — deep in a drawdown; needs a base and "
                               f"a reclaim before trend-following works.", f"{off_hi:+.1f}%"))

    # --- market / relative-strength context ---
    if ctx is not None and len(ctx):
        last = ctx.iloc[-1]
        mkt_up = bool(last.get("mkt_up"))
        flags.append(_flag("Market regime", "bullish" if mkt_up else "bearish",
                           "NIFTY is in an uptrend (MA20>MA50) — the backdrop the engine requires for fresh longs."
                           if mkt_up else
                           "NIFTY is in a downtrend — the engine gates fresh longs off here, Hammer reversals "
                           "excepted (validated market filter)."
                           if use_market_context else
                           "NIFTY is in a downtrend. In the default view the engine gates fresh longs off here; "
                           "you are looking at the stock-only view, so it is not being applied."))
        rs20 = last.get("rs_20")
        if rs20 is not None and np.isfinite(rs20):
            rs20 = float(rs20)
            flags.append(_flag("Relative strength", "bullish" if rs20 >= 0 else "bearish",
                               f"{'Leading' if rs20 >= 0 else 'Lagging'} its sector by {abs(rs20) * 100:.1f}% over 20 days "
                               f"— {'money is favouring it over peers' if rs20 >= 0 else 'money is rotating into peers, not this stock'}.",
                               f"{rs20 * 100:+.1f}%"))
        dist = last.get("dist_days")
        if dist is not None and np.isfinite(dist) and float(dist) >= 5:
            flags.append(_flag("Distribution days", "bearish",
                               f"{int(dist)} distribution days in 25 sessions (index down on rising volume — big-fund selling). "
                               f"At 5+, our test showed new buying does badly; the engine pauses fresh buys until it clears."
                               if use_market_context else
                               f"{int(dist)} distribution days in 25 sessions (index down on rising volume — big-fund selling). "
                               f"The default view pauses fresh buys at 5+; this stock-only view does not."))

    # In the stock-only view the tape flags stay VISIBLE but stop counting —
    # otherwise the lean here would contradict the advice card sitting above it,
    # which has already excluded exactly these factors from its score.
    if not use_market_context:
        for f in flags:
            if f["axis"] == "environment":
                f["ignored"] = True

    counted = [f for f in flags if not f.get("ignored")]
    bulls = sum(1 for f in counted if f["state"] == "bullish")
    bears = sum(1 for f in counted if f["state"] == "bearish")
    if bulls > bears + 1:
        lean = "bullish"
    elif bears > bulls + 1:
        lean = "bearish"
    else:
        lean = "mixed"

    return {
        "lean": lean,
        "bullish_count": bulls,
        "bearish_count": bears,
        "flags": flags,
        "note": "A deterministic snapshot of technical posture — facts, not a forecast. "
                "Thresholds echo this system's own 4-year study. Read alongside the Advice card, "
                "not instead of it.",
    }
