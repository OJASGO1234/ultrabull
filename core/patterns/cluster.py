# patterns/cluster.py
import pandas as pd

def apply_cooldown(dets: pd.DataFrame, bars: int = 3, index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Keep at most one detection per pattern every `bars` TRADING bars.
    Pass the OHLCV DatetimeIndex as `index` so distance is measured in bars,
    not calendar days (weekends/holidays would otherwise shrink the cooldown)."""
    if dets.empty: return dets
    dets = dets.sort_values(["pattern","date","score"], ascending=[True,True,False]).reset_index(drop=True)
    if index is not None:
        pos_lookup = pd.Series(range(len(index)), index=index)
        bar_of = dets["date"].map(pos_lookup)
    else:
        bar_of = None
    out = []
    for pat, g in dets.groupby("pattern", sort=False):
        keep_idx = []
        last = None
        for i, row in g.iterrows():
            if bar_of is not None:
                cur = bar_of.loc[i]
                if pd.isna(cur):  # date not in trading index; keep as-is
                    keep_idx.append(i)
                elif last is None or (cur - last) >= bars:
                    keep_idx.append(i)
                    last = cur
            else:  # fallback: calendar days
                if last is None or (row["date"] - last).days >= bars:
                    keep_idx.append(i)
                    last = row["date"]
        out.append(g.loc[keep_idx])
    return pd.concat(out, ignore_index=True).sort_values("date")

def add_strength_and_gate(dets: pd.DataFrame, emit_min: int) -> pd.DataFrame:
    if dets.empty: return dets
    from .scoring import strength
    dets = dets.copy()
    dets["strength"] = strength(dets["score"])
    return dets[dets["score"] >= emit_min].reset_index(drop=True)
