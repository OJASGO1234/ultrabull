# patterns/breakouts.py
import pandas as pd
from .scoring import base

_EMPTY = pd.DataFrame(columns=["date","pattern","score","strength","regime","notes","params"])

def breakouts(df, ind, lookbacks=(20,55), vol_k=1.5):
    hi, lo, c, v = df["High"], df["Low"], df["Close"], df["Volume"]
    base_score = base("breakout")
    reg = ind["_regime"]
    vma20 = v.rolling(20).mean().replace(0, 1)

    outs = []
    for N in lookbacks:
        rh = hi.rolling(N).max().shift(1)
        rl = lo.rolling(N).min().shift(1)
        bo = (c > rh) & (v >= vol_k * vma20)
        bd = (c < rl) & (v >= vol_k * vma20)

        if bo.any():
            score = base_score
            outs.append(pd.DataFrame({
                "date": df.index[bo],
                "pattern": f"Breakout{N}",
                "score": score + 10,  # small bump; regime adds context via Phase 2 text
                "strength": None,
                "regime": reg[bo].values,
                "notes": f"Close > {N}d high with volume ≥{vol_k}×",
                "params": [{"lookback": N, "vol_k": vol_k}] * int(bo.sum())
            }))
        if bd.any():
            score = base_score
            outs.append(pd.DataFrame({
                "date": df.index[bd],
                "pattern": f"Breakdown{N}",
                "score": score + 10,
                "strength": None,
                "regime": reg[bd].values,
                "notes": f"Close < {N}d low with volume ≥{vol_k}×",
                "params": [{"lookback": N, "vol_k": vol_k}] * int(bd.sum())
            }))
    return pd.concat(outs, ignore_index=True) if outs else _EMPTY.copy()
