# patterns/viz.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
# --- helpers ---

def make_signal_addplots(df_index, detections, *, panel=2,
                         only_qualified=True, min_score=60, families=None):
    """Return a list of mpf.addplot objects for a compact signal panel (1 = bull, -1 = bear)."""
    if detections is None or detections.empty:
        return []

    d = detections.copy()
    if only_qualified and "qualified" in d: d = d[d["qualified"]]
    if min_score is not None and "score" in d: d = d[d["score"] >= min_score]
    if families is not None:
        def _fam(p):
            if p.startswith(("Breakout","Breakdown")): return "breakout"
            if p.startswith(("RSI_","MACD_")):         return "momentum"
            if p.startswith(("BB_","Pullback")):       return "pullback"
            return "candles"
        d = d[d["pattern"].map(_fam).isin(families)]
    if d.empty: return []

    # stem values
    bull = pd.Series(index=df_index, dtype=float)
    bear = pd.Series(index=df_index, dtype=float)
    d_dates = pd.to_datetime(d["date"])
    bull.loc[d_dates[d["direction"]=="bullish"]] = 1.0
    bear.loc[d_dates[d["direction"]=="bearish"]] = -1.0

    aps = []
    if bull.notna().any():
        aps.append(mpf.make_addplot(bull, type="scatter", panel=panel, color="#16a34a", markersize=12))
    if bear.notna().any():
        aps.append(mpf.make_addplot(bear, type="scatter", panel=panel, color="#dc2626", markersize=12))
    return aps

def _dates_to_bar_index(df_index: pd.Index, dates: pd.Series | list) -> np.ndarray:
    """Map datetime-like 'dates' to integer bar positions on an mplfinance axis."""
    idx = pd.Index(df_index)
    ts = pd.to_datetime(pd.Series(dates))
    out = np.empty(len(ts), dtype=int)
    for i, d in enumerate(ts):
        try:
            loc = idx.get_loc(d)
            if isinstance(loc, slice):
                out[i] = loc.start
            elif isinstance(loc, (np.ndarray, list)):
                out[i] = int(np.flatnonzero(loc)[0])
            else:
                out[i] = int(loc)
        except KeyError:
            out[i] = idx.get_indexer([d], method="nearest")[0]
    return out

def _family_of(p: str) -> str:
    if p.startswith(("Breakout","Breakdown")): return "breakout"
    if p.startswith(("RSI_","MACD_")):         return "momentum"
    if p.startswith(("BB_","Pullback")):       return "pullback"
    return "candles"

def _color_for(row) -> tuple[str, float]:
    """Return (color, alpha) by direction/alignment."""
    direction = row.get("direction","neutral")
    alignment = row.get("alignment","neutral")
    if direction == "bullish":
        return ("#16a34a", 0.9 if alignment=="aligned" else 0.5)  # green
    if direction == "bearish":
        return ("#dc2626", 0.9 if alignment=="aligned" else 0.5)  # red
    return ("#64748b", 0.6)  # neutral gray

# --- main ---
def overlay_patterns(
    ax,
    detections: pd.DataFrame,
    df_index,
    *,
    only_qualified: bool = True,
    min_score: int | None = 60,
    families: set[str] | None = None,     # e.g., {"breakout","pullback"}
    top_per_day: int = 1,
    y_pos: str = "top",
    draw_lines: bool = True,
    label_rotation: int = 90,
):
    """
    Annotate the price axis with compact pattern markers.
    Pass the SAME index used in mpf.plot via df_index (DatetimeIndex).
    """
    if ax is None or detections is None or detections.empty:
        return ax

    d = detections.copy()
    # optional filters to reduce clutter
    if only_qualified and "qualified" in d:
        d = d[d["qualified"]]
    if min_score is not None and "score" in d:
        d = d[d["score"] >= min_score]
    if families is not None:
        d = d[[ _family_of(p) in families for p in d["pattern"] ]]

    if d.empty:
        return ax

    # keep the top N per day by score
    if "score" in d:
        d = (d.sort_values(["date","score"], ascending=[True, False])
               .groupby("date", as_index=False)
               .head(top_per_day))
    else:
        d = d.drop_duplicates(subset=["date"])

    # map dates -> bar indices
    xs = _dates_to_bar_index(df_index, d["date"])
    y_top, y_bottom = ax.get_ylim()
    y = y_top if y_pos == "top" else y_bottom

    # default short labels
    code_map = {
        "BullishEngulfing": "BE↑", "BearishEngulfing": "BE↓",
        "Hammer": "HM", "Doji": "DJ",
        "Breakout20": "BO20", "Breakout55": "BO55",
        "Breakdown20": "BD20", "Breakdown55": "BD55",
        "PullbackToMA20": "PB20",
        "BB_BuyDip": "BB↓", "BB_SellFade": "BB↑",
        "RSI_40_Up": "RSI40↑", "RSI_60_Down": "RSI60↓",
        "MACD_Cross_Zeroish_Up": "MACD0×↑",
        "MACD_Cross_Zeroish_Down": "MACD0×↓",
    }

    # draw
    for (x, _, row) in zip(xs, d["pattern"], d.itertuples(index=False)):
        color, alpha = _color_for(row._asdict() if hasattr(row, "_asdict") else dict(row._mapping))
        if draw_lines:
            ax.axvline(x, linestyle="--", linewidth=0.6, alpha=0.35*alpha, color=color)
        label = code_map.get(row.pattern, row.pattern)
        ax.text(x, y, label, rotation=label_rotation,
                va="top" if y_pos=="top" else "bottom",
                ha="center", fontsize=7, color=color, alpha=alpha)

    return ax
