# core/interpret.py
"""
Deterministic plain-English readings of the RSI, MACD, and equity-curve charts.

Heuristics, not an LLM: everything worth saying about these charts (zone,
crosses, divergences, momentum direction, drawdown state) is computable from
the same arrays the charts draw — which makes the inference instant, free,
reproducible, and guaranteed consistent with this project's tested findings.
The chat assistant handles open-ended follow-ups.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _last_cross_ago(diff: pd.Series) -> int | None:
    """Bars since the sign of `diff` last flipped (None if never in window)."""
    s = np.sign(diff.dropna().to_numpy())
    if len(s) < 2:
        return None
    flips = np.where(s[1:] != s[:-1])[0]
    return int(len(s) - 2 - flips[-1] + 1) if len(flips) else None


def interpret_rsi(ind_df: pd.DataFrame, close: pd.Series) -> str | None:
    r = ind_df["RSI"].dropna()
    if len(r) < 10:
        return None
    cur = float(r.iloc[-1])
    delta5 = cur - float(r.iloc[-6]) if len(r) > 6 else 0.0
    rising = delta5 > 2
    falling = delta5 < -2
    trend_word = "rising" if rising else ("falling" if falling else "flat")

    if cur > 70:
        zone = (f"RSI {cur:.0f} is overbought — in our 4y study, fresh longs opened here "
                f"underperformed over the next week (−16 bps avg); a pullback or pause is the norm, "
                f"not a reason to panic-sell an existing position")
    elif cur >= 50:
        zone = (f"RSI {cur:.0f} sits in the 50–65 momentum band — the one RSI zone that tested "
                f"mildly positive for forward returns" if cur <= 65
                else f"RSI {cur:.0f} shows strong momentum, approaching overbought")
    elif cur >= 40:
        zone = f"RSI {cur:.0f} is neutral-weak — no momentum edge either way"
    elif cur >= 30:
        zone = f"RSI {cur:.0f} is weak — sellers in control but not yet washed out"
    else:
        zone = (f"RSI {cur:.0f} is washed out — bounces often start here, but our rule is to wait "
                f"for a reversal bar (e.g. a Hammer) rather than catch the falling knife")

    # simple divergence check over the last ~40 bars
    div = ""
    if len(close) >= 40 and len(r) >= 40:
        c = close.iloc[-40:]
        rr = r.iloc[-40:]
        recent, prior = c.iloc[-20:], c.iloc[:-20]
        if recent.max() > prior.max() and rr.loc[recent.idxmax()] < rr.loc[prior.idxmax()] - 2:
            div = " Caution: price made a higher high while RSI made a lower high (bearish divergence — momentum not confirming the advance)."
        elif recent.min() < prior.min() and rr.loc[recent.idxmin()] > rr.loc[prior.idxmin()] + 2:
            div = " Note: price made a lower low while RSI made a higher low (bullish divergence — selling pressure fading)."

    return f"{zone}; currently {trend_word}.{div}"


def interpret_macd(ind_df: pd.DataFrame) -> str | None:
    macd = ind_df["MACD"].dropna()
    sig = ind_df["MACD_signal"].dropna()
    if len(macd) < 10 or len(sig) < 10:
        return None
    m, s = float(macd.iloc[-1]), float(sig.iloc[-1])
    hist = macd - sig
    h = hist.dropna()
    above = m > s
    ago = _last_cross_ago(hist)
    side = "above" if above else "below"
    cross_txt = f"crossed {'up' if above else 'down'} {ago} bar{'s' if ago != 1 else ''} ago" if ago is not None and ago <= 15 else \
        f"has been {side} its signal line for a while"
    mom = ""
    if len(h) > 4:
        widening = abs(float(h.iloc[-1])) > abs(float(h.iloc[-4]))
        if above:
            mom = " and the gap is widening (momentum building)" if widening else " but the gap is narrowing (momentum fading — watch for a down-cross)"
        else:
            mom = " and the gap is widening (downside momentum building)" if widening else " but the gap is narrowing (selling pressure fading)"
    territory = "positive territory (medium-term trend up)" if m > 0 else "negative territory (medium-term trend down)"
    return (f"MACD {cross_txt}{mom}. The line is in {territory}. "
            f"Crosses lag price by design — use them as confirmation of the levels/regime story, not as a standalone trigger.")


def interpret_equity(equity: pd.Series, trades) -> str | None:
    if equity is None or len(equity) < 5:
        return None
    start, last = float(equity.iloc[0]), float(equity.iloc[-1])
    total = last / start - 1
    peak = float(equity.cummax().iloc[-1])
    dd = last / peak - 1
    n = len(trades) if trades is not None else 0

    if n == 0:
        return ("Flat line: the gates blocked every signal in this window, so no trades were taken. "
                "That is the system working — no edge, no position.")

    pnls = [t for t in (trades["pnl"].tolist() if hasattr(trades, "columns") else [x.get("pnl") for x in trades]) if t is not None]
    streak_txt = ""
    if pnls:
        sign = pnls[-1] > 0
        k = 1
        for p in reversed(pnls[:-1]):
            if (p > 0) == sign:
                k += 1
            else:
                break
        if k >= 2:
            streak_txt = f" Most recent run: {k} consecutive {'winners' if sign else 'losers'}."
    dd_txt = " and is at its high-water mark" if dd > -0.001 else f", currently {abs(dd) * 100:.1f}% below its peak"
    return (f"The strategy {'made' if total >= 0 else 'lost'} {abs(total) * 100:.2f}% over this window across {n} trade{'s' if n != 1 else ''}{dd_txt}.{streak_txt} "
            f"Steps happen at trade exits; flat stretches are periods with no open position. "
            f"With {n} trades this is a small sample — judge the process (per-pattern stats, gate behavior), not this single curve.")
