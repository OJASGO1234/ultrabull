# patterns/candles.py
import pandas as pd
from .primitives import candle_parts
from .scoring import base

_EMPTY = pd.DataFrame(columns=["date","pattern","score","strength","regime","notes","params"])

def engulfing(df, ind, ratio=1.05):
    o,h,l,c = df["Open"], df["High"], df["Low"], df["Close"]
    po, ph, pl, pc = o.shift(1), h.shift(1), l.shift(1), c.shift(1)

    bull = (pc < po) & (c > o) & ((c - o).abs() > (po - pc).abs() * ratio) & (h > ph) & (l < pl)
    bear = (pc > po) & (c < o) & ((o - c).abs() > (pc - po).abs() * ratio) & (h > ph) & (l < pl)

    base_score = base("candles")
    reg = ind["_regime"]
    outs = []
    if bull.any():
        outs.append(pd.DataFrame({
            "date": df.index[bull],
            "pattern": "BullishEngulfing",
            "score": base_score + 5,
            "strength": None,
            "regime": reg[bull].values,
            "notes": "Bullish engulfing",
            "params": [{"ratio": ratio}] * int(bull.sum())
        }))
    if bear.any():
        outs.append(pd.DataFrame({
            "date": df.index[bear],
            "pattern": "BearishEngulfing",
            "score": base_score + 5,
            "strength": None,
            "regime": reg[bear].values,
            "notes": "Bearish engulfing",
            "params": [{"ratio": ratio}] * int(bear.sum())
        }))
    return pd.concat(outs, ignore_index=True) if outs else _EMPTY.copy()

def hammer_doji(df, ind, hammer_shadow_ratio=2.0, doji_body_pct=0.10):
    body, rng, upsh, lowsh, body_pct = candle_parts(df)
    hammer = (lowsh >= hammer_shadow_ratio * body) & (upsh <= 0.3 * body)
    doji = (body_pct <= doji_body_pct) & (rng >= rng.rolling(20).median())

    base_score = base("candles")
    reg = ind["_regime"]
    outs = []
    if hammer.any():
        outs.append(pd.DataFrame({
            "date": df.index[hammer],
            "pattern": "Hammer",
            "score": base_score + 5,
            "strength": None,
            "regime": reg[hammer].values,
            "notes": "Hammer (long lower shadow)",
            "params": [{"shadow_ratio": hammer_shadow_ratio}] * int(hammer.sum())
        }))
    if doji.any():
        outs.append(pd.DataFrame({
            "date": df.index[doji],
            "pattern": "Doji",
            "score": base_score,
            "strength": None,
            "regime": reg[doji].values,
            "notes": "Doji (indecision)",
            "params": [{"body_pct": doji_body_pct}] * int(doji.sum())
        }))
    return pd.concat(outs, ignore_index=True) if outs else _EMPTY.copy()
