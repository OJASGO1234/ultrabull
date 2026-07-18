# patterns/pullbacks.py
import pandas as pd
from .primitives import near
from .scoring import base

_EMPTY = pd.DataFrame(columns=["date","pattern","score","strength","regime","notes","params"])

def pullback_ma20(df, ind, atr_tol_k=0.5):
    up = ind["_regime"]=="uptrend"
    dipped = near(df["Low"], ind["MA20"], ind["ATR"], atr_tol_k)
    close_above = df["Close"] > ind["MA20"]
    hit = up & dipped & close_above
    if not hit.any(): return _EMPTY.copy()
    base_score = base("pullback")
    return pd.DataFrame({
        "date": df.index[hit],
        "pattern": "PullbackToMA20",
        "score": base_score + 5,
        "strength": None,
        "regime": ind["_regime"][hit].values,
        "notes": f"Dip to MA20 within {atr_tol_k}×ATR and close above",
        "params": [{"ma":20, "atr_tol_k": atr_tol_k}] * int(hit.sum())
    })

def bb_reversion(df, ind):
    up = ind["_regime"]=="uptrend"
    down = ind["_regime"]=="downtrend"
    touch_low = df["Low"] <= ind["BB_lower"]
    touch_up  = df["High"] >= ind["BB_upper"]

    bbuy = up & touch_low
    bsell = down & touch_up
    base_score = base("pullback")

    outs = []
    if bbuy.any():
        outs.append(pd.DataFrame({
            "date": df.index[bbuy], "pattern": "BB_BuyDip",
            "score": base_score + 5, "strength": None, "regime": ind["_regime"][bbuy].values,
            "notes": "Touched lower band in uptrend", "params": [{"type":"lower"}]*int(bbuy.sum())
        }))
    if bsell.any():
        outs.append(pd.DataFrame({
            "date": df.index[bsell], "pattern": "BB_SellFade",
            "score": base_score + 5, "strength": None, "regime": ind["_regime"][bsell].values,
            "notes": "Touched upper band in downtrend", "params": [{"type":"upper"}]*int(bsell.sum())
        }))
    return pd.concat(outs, ignore_index=True) if outs else _EMPTY.copy()
