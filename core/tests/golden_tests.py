# core/tests/golden_tests.py

# --- imports from our detectors ---
from patterns.candles import hammer_doji
from patterns.breakouts import breakouts
from patterns.cluster import apply_cooldown

import numpy as np
import pandas as pd

# --- minimal synthetic data builders (test-only) ---
def make_df(n=40, start="2025-01-01", base=100.0):
    idx = pd.bdate_range(start, periods=n)
    close = base + np.cumsum(np.random.normal(0, 0.2, size=n))
    open_ = close + np.random.normal(0, 0.05, size=n)
    high  = np.maximum(open_, close) + 0.3
    low   = np.minimum(open_, close) - 0.3
    vol   = np.full(n, 1_000.0)
    return pd.DataFrame({"Open":open_,"High":high,"Low":low,"Close":close,"Volume":vol}, index=idx)

def make_ind_like(df, regime="uptrend"):
    ind = pd.DataFrame(index=df.index)
    ind["_regime"] = regime
    ind["RSI"] = 50.0
    ind["ATR"] = 1.0
    ind["MACD"] = 0.0
    ind["MACD_signal"] = 0.0
    ind["BB_upper"] = df["Close"] + 2.0
    ind["BB_lower"] = df["Close"] - 2.0
    ind["MA20"] = df["Close"].rolling(20).mean().bfill()
    ind["MA50"] = df["Close"].rolling(50).mean().bfill()
    return ind


def golden_tests():
    # 1) Hammer on a known date gets detected
    df = make_df()
    t = 15
    df.loc[df.index[t], ["Open","Close","High","Low"]] = [100.0, 100.2, 100.4, 98.0]
    ind = make_ind_like(df, regime="downtrend")
    ham = hammer_doji(df, ind)
    assert df.index[t] in set(ham["date"]), "Hammer not detected at expected date"

    # 2) Breakout20 with volume spike gets detected
    df = make_df(n=50)
    ind = make_ind_like(df, regime="uptrend")
    df.loc[df.index[:30], "High"]  = 104.0
    df.loc[df.index[:30], "Close"] = 103.5
    df.loc[df.index[30], "Close"]  = 106.0
    df.loc[df.index[30], "High"]   = 106.5
    df.loc[df.index[30], "Volume"] = 3000.0  # 3× avg
    bo = breakouts(df, ind, lookbacks=(20,), vol_k=1.5)
    assert df.index[30] in set(bo["date"]), "Breakout20 not detected at expected date"

    # 3) Cooldown removes near-duplicate hammer hits
    df = make_df()
    ind = make_ind_like(df, regime="downtrend")
    for t in [10, 11]:
        df.loc[df.index[t], ["Open","Close","High","Low"]] = [100.0, 100.2, 100.4, 98.0]
    ham = hammer_doji(df, ind)
    ham_cd = apply_cooldown(ham, bars=3)
    assert len(ham) >= 2 and len(ham_cd) == 1, "Cooldown did not reduce duplicates"

    print("Golden tests: PASS")
