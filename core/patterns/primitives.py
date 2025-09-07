# patterns/primitives.py
import numpy as np
import pandas as pd

def candle_parts(df: pd.DataFrame):
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    body  = (c - o).abs()
    rng   = (h - l).clip(lower=1e-12)
    upsh  = h - pd.concat([c, o], axis=1).max(axis=1)
    lowsh = pd.concat([c, o], axis=1).min(axis=1) - l
    body_pct = body / rng
    return body, rng, upsh, lowsh, body_pct

def zscore(s: pd.Series, n: int = 20) -> pd.Series:
    ma = s.rolling(n).mean()
    sd = s.rolling(n).std(ddof=0).replace(0, np.nan)
    return (s - ma) / sd

def near(x: pd.Series, y: pd.Series, atr: pd.Series, k: float) -> pd.Series:
    return (x - y).abs() <= (k * atr)

def regime_from(ind: pd.DataFrame) -> pd.Series:
    # Expect MA20, MA50 in ind
    up = ind["MA20"] > ind["MA50"]
    down = ind["MA20"] < ind["MA50"]
    return pd.Series(np.where(up, "uptrend", np.where(down, "downtrend", "neutral")),
                     index=ind.index, name="_regime")
