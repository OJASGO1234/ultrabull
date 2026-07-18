# patterns/momentum.py
import numpy as np
import pandas as pd
from .scoring import base

_EMPTY = pd.DataFrame(columns=["date","pattern","score","strength","regime","notes","params"])

def rsi_regime_cross(ind, low=40, high=60):
    r = ind["RSI"]
    up_cross = (r.shift(1) < low) & (r >= low)
    dn_cross = (r.shift(1) > high) & (r <= high)
    base_score = base("momentum")
    outs = []
    if up_cross.any():
        outs.append(pd.DataFrame({
            "date": r.index[up_cross], "pattern": f"RSI_{low}_Up",
            "score": base_score + 5, "strength": None, "regime": ind["_regime"][up_cross].values,
            "notes": f"RSI crossed up through {low}", "params": [{"level": low}]*int(up_cross.sum())
        }))
    if dn_cross.any():
        outs.append(pd.DataFrame({
            "date": r.index[dn_cross], "pattern": f"RSI_{high}_Down",
            "score": base_score + 5, "strength": None, "regime": ind["_regime"][dn_cross].values,
            "notes": f"RSI crossed down through {high}", "params": [{"level": high}]*int(dn_cross.sum())
        }))
    return pd.concat(outs, ignore_index=True) if outs else _EMPTY.copy()

def macd_cross_near_zero(ind, eps_atr_pct=0.25):
    macd, sig, atr, close = ind["MACD"], ind["MACD_signal"], ind["ATR"], ind["Close"]
    cross_up = (macd.shift(1) <= sig.shift(1)) & (macd > sig)
    cross_dn = (macd.shift(1) >= sig.shift(1)) & (macd < sig)
    # "near zero" measured in price units: |MACD| within eps_atr_pct fraction of ATR
    near0 = macd.abs() <= eps_atr_pct * atr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    base_score = base("momentum")

    up_hit = cross_up & near0
    dn_hit = cross_dn & near0
    outs = []
    if up_hit.any():
        outs.append(pd.DataFrame({
            "date": macd.index[up_hit], "pattern": "MACD_Cross_Zeroish_Up",
            "score": base_score + 10, "strength": None, "regime": ind["_regime"][up_hit].values,
            "notes": "MACD crossed up near zero", "params": [{"eps_atr_pct": eps_atr_pct}]*int(up_hit.sum())
        }))
    if dn_hit.any():
        outs.append(pd.DataFrame({
            "date": macd.index[dn_hit], "pattern": "MACD_Cross_Zeroish_Down",
            "score": base_score + 10, "strength": None, "regime": ind["_regime"][dn_hit].values,
            "notes": "MACD crossed down near zero", "params": [{"eps_atr_pct": eps_atr_pct}]*int(dn_hit.sum())
        }))
    return pd.concat(outs, ignore_index=True) if outs else _EMPTY.copy()
