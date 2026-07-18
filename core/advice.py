# core/advice.py
"""
Level-aware buy/sell advice for the CURRENT bar.

Combines, with fully transparent scoring:
  - position vs support/resistance (entry quality, chase risk, breakout)
  - regime gates (market, sector, relative strength, stock trend)
  - momentum (RSI band incl. the tested-negative >70 zone, MACD state)
  - volume (20d multiple, signed by the day's direction)
  - recent qualified pattern detections
  - earnings blackout (veto on fresh entries)

Outputs two verdicts — fresh money vs existing holding — plus a concrete plan
whose stop sits BELOW the nearest support (a level break proves the thesis
wrong) and whose target is the next resistance. Every factor is returned with
its contribution so the UI can show WHY.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _factor(name: str, score: float, detail: str) -> dict:
    return {"name": name, "score": round(score, 2), "detail": detail}


def build_advice(
    df_ohlcv: pd.DataFrame,
    ind_df: pd.DataFrame,
    ctx: pd.DataFrame | None,
    levels: list[dict] | None,
    timing: dict | None,
    detections: pd.DataFrame | None,
) -> dict:
    close = float(df_ohlcv["Close"].iloc[-1])
    atr = float(ind_df["ATR"].iloc[-1]) if np.isfinite(ind_df["ATR"].iloc[-1]) else close * 0.02
    rsi = float(ind_df["RSI"].iloc[-1]) if np.isfinite(ind_df["RSI"].iloc[-1]) else 50.0
    macd_up = bool(ind_df["MACD"].iloc[-1] > ind_df["MACD_signal"].iloc[-1]) \
        if "MACD" in ind_df.columns else None
    trend_up = bool(ind_df["MA20"].iloc[-1] > ind_df["MA50"].iloc[-1])

    vol = df_ohlcv["Volume"]
    volx = float(vol.iloc[-1] / max(1e-9, vol.rolling(20).mean().iloc[-1]))
    up_day = bool(df_ohlcv["Close"].iloc[-1] >= df_ohlcv["Close"].iloc[-2]) if len(df_ohlcv) > 1 else True

    mkt_up = sect_up = None
    rs20 = breadth = dist_days = None
    if ctx is not None and len(ctx):
        last = ctx.iloc[-1]
        mkt_up = bool(last.get("mkt_up"))
        sect_up = bool(last.get("sect_up"))
        rs20 = float(last["rs_20"]) if np.isfinite(last.get("rs_20", np.nan)) else None
        breadth = float(last["breadth"]) if np.isfinite(last.get("breadth", np.nan)) else None
        dist_days = float(last["dist_days"]) if np.isfinite(last.get("dist_days", np.nan)) else None

    supports = sorted([lv for lv in (levels or []) if lv["kind"] == "support" and lv["price"] <= close],
                      key=lambda lv: -lv["price"])
    resistances = sorted([lv for lv in (levels or []) if lv["kind"] == "resistance" and lv["price"] >= close],
                         key=lambda lv: lv["price"])
    sup = supports[0] if supports else None
    res = resistances[0] if resistances else None
    res2 = resistances[1] if len(resistances) > 1 else None

    blackout = bool((timing or {}).get("in_earnings_blackout"))

    factors: list[dict] = []

    # --- regime ---
    if mkt_up is not None:
        factors.append(_factor("Market regime", 1.0 if mkt_up else -1.5,
                               "NIFTY uptrend supports longs" if mkt_up
                               else "NIFTY downtrend — the engine gates all fresh longs off here"))
    if sect_up is not None:
        factors.append(_factor("Sector regime", 0.5 if sect_up else -0.5,
                               "sector index trending up" if sect_up else "sector index trending down"))
    if rs20 is not None:
        factors.append(_factor("Relative strength", 1.0 if rs20 >= 0 else -1.0,
                               f"{'leading' if rs20 >= 0 else 'lagging'} its sector by {abs(rs20) * 100:.1f}% over 20d"))
    factors.append(_factor("Stock trend", 1.0 if trend_up else -1.0,
                           "MA20 above MA50" if trend_up else "MA20 below MA50"))

    # --- market health (validated: dist-day ceiling doubled pool edge @10 bars) ---
    if dist_days is not None:
        if dist_days >= 5:
            factors.append(_factor("Distribution days", -1.5,
                                   f"{int(dist_days)} NIFTY distribution days in 25 sessions — institutional selling; the engine gates fresh longs off here"))
        elif dist_days >= 4:
            factors.append(_factor("Distribution days", -0.5,
                                   f"{int(dist_days)}/25 sessions — one more and fresh longs are gated off"))
        else:
            factors.append(_factor("Distribution days", 0.0,
                                   f"{int(dist_days)}/25 sessions — no distribution warning"))
    if breadth is not None:
        factors.append(_factor(
            "Market breadth",
            0.5 if breadth >= 0.55 else (-0.5 if breadth <= 0.35 else 0.0),
            f"{breadth:.0%} of universe above its 50DMA — "
            + ("broad participation" if breadth >= 0.55
               else "narrow, index carried by few names" if breadth <= 0.35
               else "middling")))

    # --- momentum ---
    if rsi > 70:
        factors.append(_factor("RSI", -1.0,
                               f"RSI {rsi:.0f} — overbought; fresh longs here tested NEGATIVE (−16bps/5d) in our study"))
    elif 40 <= rsi <= 65:
        factors.append(_factor("RSI", 1.0, f"RSI {rsi:.0f} — momentum without overheat (the one RSI zone with tested edge)"))
    elif rsi < 35:
        factors.append(_factor("RSI", -0.5, f"RSI {rsi:.0f} — washed out; wait for a reversal bar before catching it"))
    else:
        factors.append(_factor("RSI", 0.0, f"RSI {rsi:.0f} — neutral zone"))
    if macd_up is not None:
        factors.append(_factor("MACD", 0.5 if macd_up else -0.5,
                               "MACD above signal line" if macd_up else "MACD below signal line"))

    # --- volume ---
    if volx >= 1.5:
        factors.append(_factor("Volume", 0.5 if up_day else -0.5,
                               f"{volx:.1f}× 20d volume on an {'up' if up_day else 'down'} day"))
    else:
        factors.append(_factor("Volume", 0.0, f"{volx:.1f}× 20d average — unremarkable"))

    # --- position vs levels ---
    breakout = res is None or close > res["price"]
    if breakout and res is None and supports:
        factors.append(_factor("Levels", 0.5, "above every mapped resistance (blue-sky region near 52w high)"))
    elif res is not None:
        d_sup = (close - sup["price"]) / atr if sup else None
        d_res = (res["price"] - close) / atr
        if sup and d_sup is not None and d_sup <= 1.0:
            factors.append(_factor("Levels", 1.0,
                                   f"sitting on support ₹{sup['price']:.0f} ({sup['touches']} touches) — defined-risk entry zone"))
        elif d_res <= 0.75:
            factors.append(_factor("Levels", -1.0,
                                   f"pressed against resistance ₹{res['price']:.0f} ({res['touches']} touches) — chasing here buys someone's exit; wait for a close above it or a dip"))
        else:
            factors.append(_factor("Levels", 0.0,
                                   f"mid-range between ₹{sup['price']:.0f} and ₹{res['price']:.0f}" if sup
                                   else f"below resistance ₹{res['price']:.0f}, no mapped support beneath"))

    # reward:risk from the levels themselves
    rr = None
    if sup and res:
        risk = close - (sup["price"] - 0.5 * atr)
        reward = res["price"] - close
        if risk > 0:
            rr = reward / risk
            factors.append(_factor("Reward:risk", 0.5 if rr >= 1.5 else (-0.5 if rr < 1.0 else 0.0),
                                   f"{rr:.1f}:1 to next resistance vs stop below support"))

    # --- recent patterns ---
    if detections is not None and len(detections):
        d = detections[detections.get("qualified", False)]
        if len(d):
            recent = d[pd.to_datetime(d["date"]) >= df_ohlcv.index[-4]] if len(df_ohlcv) > 4 else d
            bulls = int((recent["direction"] == "bullish").sum())
            bears = int((recent["direction"] == "bearish").sum())
            if bulls > bears:
                pats = ", ".join(recent.loc[recent["direction"] == "bullish", "pattern"].unique()[:3])
                factors.append(_factor("Recent patterns", 1.0, f"fresh bullish setups: {pats}"))
            elif bears > bulls:
                pats = ", ".join(recent.loc[recent["direction"] == "bearish", "pattern"].unique()[:3])
                factors.append(_factor("Recent patterns", -1.0, f"fresh bearish setups: {pats}"))

    # --- earnings blackout (veto) ---
    if blackout:
        factors.append(_factor("Earnings blackout", -99,
                               "results imminent — gap risk dwarfs any setup edge; no fresh entries"))

    score = float(sum(f["score"] for f in factors if f["score"] > -90))

    # --- verdicts ---
    if blackout:
        fresh = {"verdict": "WAIT", "reason": "earnings blackout — re-assess after results"}
    elif score >= 3.5:
        fresh = {"verdict": "BUY", "reason": "regime, momentum and level position all align"}
    elif score >= 1.5:
        fresh = ({"verdict": "BUY", "reason": "constructive backdrop and price is already at support"}
                 if sup and (close - sup["price"]) / atr <= 1.0
                 else {"verdict": "BUY_ON_DIP",
                       "reason": f"constructive, but entry is better near support ₹{sup['price']:.0f}" if sup
                       else "constructive, but wait for a defined-risk entry"})
    elif score > -1.5:
        fresh = {"verdict": "WAIT", "reason": "signals are mixed — no defined edge at this price"}
    else:
        fresh = {"verdict": "AVOID", "reason": "regime and momentum are against fresh longs"}

    if score <= -3.0:
        holding = {"verdict": "EXIT_OR_REDUCE", "reason": "multiple factors negative — protect capital"}
    elif score <= -1.0:
        holding = {"verdict": "TIGHTEN_STOP",
                   "reason": f"deteriorating — raise stop to just below support ₹{sup['price']:.0f}" if sup
                   else "deteriorating — tighten stop to ~2 ATR below price"}
    elif rsi > 70 and res is not None and (res["price"] - close) / atr <= 0.75:
        holding = {"verdict": "CONSIDER_TRIM", "reason": "overbought into resistance — a partial profit-take is reasonable"}
    else:
        holding = {"verdict": "HOLD", "reason": "no exit trigger; let the position work with the stop in place"}

    # --- plan (levels first, ATR fallback) ---
    stop = round(sup["price"] - 0.5 * atr, 2) if sup else round(close - 2.0 * atr, 2)
    plan = {
        "entry_zone_low": round(sup["price"], 2) if sup else round(close - 0.5 * atr, 2),
        "entry_zone_high": round(min(close, sup["price"] + 0.75 * atr), 2) if sup else round(close, 2),
        "stop": stop,
        "stop_basis": "0.5×ATR below nearest support — a close below that level breaks the setup"
                      if sup else "2×ATR below price (no mapped support beneath)",
        "target1": round(res["price"], 2) if res else round(close + 2.0 * atr, 2),
        "target2": round(res2["price"], 2) if res2 else None,
        "reward_risk": round(rr, 2) if rr else None,
    }

    return {
        "as_of": str(df_ohlcv.index[-1].date()),
        "close": round(close, 2),
        "score": round(score, 2),
        "fresh": fresh,
        "holding": holding,
        "factors": factors,
        "plan": plan,
        "note": ("Rule-based synthesis of the layers above — factor weights are fixed and shown, "
                 "not fitted. It frames risk; it does not predict. Confirm with your fundamental view."),
    }
