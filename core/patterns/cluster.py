# patterns/cluster.py
import pandas as pd

def apply_cooldown(dets: pd.DataFrame, bars: int = 3) -> pd.DataFrame:
    if dets.empty: return dets
    dets = dets.sort_values(["pattern","date","score"], ascending=[True,True,False]).reset_index(drop=True)
    out = []
    for pat, g in dets.groupby("pattern", sort=False):
        keep_idx = []
        last_date = None
        for i, row in g.iterrows():
            if last_date is None or (row["date"] - last_date).days >= bars:
                keep_idx.append(i)
                last_date = row["date"]
        out.append(g.loc[keep_idx])
    return pd.concat(out, ignore_index=True).sort_values("date")

def add_strength_and_gate(dets: pd.DataFrame, emit_min: int) -> pd.DataFrame:
    if dets.empty: return dets
    from .scoring import strength
    dets = dets.copy()
    dets["strength"] = strength(dets["score"])
    return dets[dets["score"] >= emit_min].reset_index(drop=True)
