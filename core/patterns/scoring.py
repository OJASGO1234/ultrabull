# patterns/scoring.py
import numpy as np
import pandas as pd

def strength(score: pd.Series) -> pd.Series:
    return np.select([score >= 70, score >= 40], ["strong", "medium"], default="weak")

def base(family: str) -> int:
    return {"candles": 50, "breakout": 60, "pullback": 55, "momentum": 50}.get(family, 50)

def finalize_score(base_score: int, mods: list[int]) -> int:
    s = base_score + sum(mods)
    return int(np.clip(s, 0, 100))
