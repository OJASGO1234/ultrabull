# patterns/vcp.py
"""
Simplified Minervini Volatility Contraction Pattern (VCP) setup detector.

The real VCP is discretionary (successively smaller pullbacks left to right).
This captures its measurable core on daily bars:
  - uptrend regime (MA20 > MA50)
  - price within 10% of its 52-week high (base near highs, not a bottom fish)
  - volatility contraction: 10-day mean ATR% at most 0.65x its level 25 bars ago
  - volume dry-up: 10-day mean volume at most 0.8x its level 25 bars ago

Emits `VCP_Setup` (bullish). The tradeable event is usually the breakout FROM
the setup; the detection marks the coiled state itself.
"""
import pandas as pd
from .scoring import base

_EMPTY = pd.DataFrame(columns=["date","pattern","score","strength","regime","notes","params"])


def vcp_setup(df, ind, contraction_k=0.65, vol_dry_k=0.8, near_high_pct=0.10):
    c, v = df["Close"], df["Volume"]
    atrp = (ind["ATR"] / c).rolling(10).mean()
    contraction = atrp <= contraction_k * atrp.shift(25)
    vol10 = v.rolling(10).mean()
    vol_dry = vol10 <= vol_dry_k * vol10.shift(25)
    near_high = c >= (1.0 - near_high_pct) * c.rolling(252, min_periods=60).max()
    up = ind["_regime"] == "uptrend"

    hit = (contraction & vol_dry & near_high & up).fillna(False)
    if not hit.any():
        return _EMPTY.copy()
    return pd.DataFrame({
        "date": df.index[hit],
        "pattern": "VCP_Setup",
        "score": base("breakout") + 5,
        "strength": None,
        "regime": ind["_regime"][hit].values,
        "notes": f"Volatility contraction near 52w high (ATR% ≤{contraction_k}× of 25 bars ago, volume dry-up)",
        "params": [{"contraction_k": contraction_k, "vol_dry_k": vol_dry_k}] * int(hit.sum()),
    })
