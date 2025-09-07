# File: patterns/postprocess.py

import pandas as pd
import numpy as np

def direction_of(pattern: str) -> str:
    """Map a pattern name to directional bias."""
    if pattern.startswith("Breakout"):   return "bullish"
    if pattern.startswith("Breakdown"):  return "bearish"
    if pattern in {"BB_BuyDip","PullbackToMA20","BullishEngulfing","Hammer",
                   "RSI_40_Up","MACD_Cross_Zeroish_Up"}:
        return "bullish"
    if pattern in {"BB_SellFade","BearishEngulfing","RSI_60_Down","MACD_Cross_Zeroish_Down"}:
        return "bearish"
    return "neutral"


def add_direction_alignment(dets: pd.DataFrame) -> pd.DataFrame:
    """Add 'direction' and 'alignment' columns based on pattern and regime."""
    d = dets.copy()
    d["direction"] = d["pattern"].map(direction_of)
    def _align(row):
        if row["direction"] == "neutral":
            return "neutral"
        if (row["direction"] == "bullish" and row["regime"] == "uptrend") or \
           (row["direction"] == "bearish" and row["regime"] == "downtrend"):
            return "aligned"
        return "counter-trend"
    d["alignment"] = d.apply(_align, axis=1)
    return d


def compute_context_features(
    df_ohlcv: pd.DataFrame,
    ind_df: pd.DataFrame,
    vol_lookback: int = 20,
    high_lookback: int = 252
) -> pd.DataFrame:
    """
    Build a per-date feature table:
      rsi, atr, vol_mult{vol_lookback}, off_52w_high_pct
    Expects df_ohlcv columns: ['open','high','low','close','volume'] (any case ok).
    """
    # normalize column names for robustness
    cols = {c.lower(): c for c in df_ohlcv.columns}
    close = df_ohlcv[cols.get("close", "close")]
    volume = df_ohlcv[cols.get("volume", "volume")]

    feat = pd.DataFrame(index=df_ohlcv.index)
    feat["rsi"] = ind_df["RSI"]
    feat["atr"] = ind_df["ATR"]
    feat[f"vol_mult{vol_lookback}"] = volume / volume.rolling(vol_lookback).mean()
    feat["off_52w_high_pct"] = (close / close.rolling(high_lookback, min_periods=1).max() - 1.0) * 100
    return feat


def merge_features(dets: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    """Left-join detections with feature table on date index."""
    return dets.merge(feat, left_on="date", right_index=True, how="left")


def to_english_row(row: pd.Series, symbol: str) -> str:
    """Generate a one-line, plain-English description for a detection row."""
    arrow = {"bullish":"↑","bearish":"↓","neutral":"·"}.get(row.get("direction","neutral"), "·")
    align = row.get("alignment", "neutral")
    rsi = row.get("rsi", np.nan)
    volx = row.get("vol_mult20", row.get("vol_mult", np.nan))
    atr = row.get("atr", np.nan)
    off = row.get("off_52w_high_pct", np.nan)
    rsi_str = "n/a" if pd.isna(rsi) else f"{float(rsi):.1f}"
    vol_str = "n/a" if pd.isna(volx) else f"{float(volx):.1f}×20d"
    atr_str = "n/a" if pd.isna(atr) else f"{float(atr):.2f}"
    off_str = "n/a" if pd.isna(off) else f"{float(off):.1f}%"
    date_str = pd.to_datetime(row["date"]).date().isoformat()
    return (f"{date_str} — {symbol} {row['pattern']} ({arrow}, {align}). "
            f"RSI={rsi_str}, Vol={vol_str}, ATR={atr_str}, off-52w-high={off_str}.")


def add_english(dets: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Add 'english' column with human-readable description for each detection."""
    d = dets.copy()
    d["english"] = d.apply(lambda r: to_english_row(r, symbol), axis=1)
    return d


def qualify_row(row: pd.Series) -> bool:
    """
    Decide if a detection qualifies for a watchlist:
      - strong OR (medium AND aligned)
      - allow counter-trend reversals for Hammer/BullishEngulfing
      - for breakouts/breakdowns, require basic volume confirmation if available
    """
    reversal_ok = row.get("pattern") in {"Hammer","BullishEngulfing"} and row.get("alignment") == "counter-trend"
    strong = row.get("strength") == "strong"
    med_aligned = (row.get("strength") == "medium") and (row.get("alignment") == "aligned")
    vol_ok = True
    pat = str(row.get("pattern",""))
    if (pat.startswith("Breakout") or pat.startswith("Breakdown")):
        vol = row.get("vol_mult20", row.get("vol_mult", np.nan))
        if not pd.isna(vol):
            vol_ok = float(vol) >= 1.2
    return (strong or med_aligned or reversal_ok) and vol_ok


def add_qualification(dets: pd.DataFrame) -> pd.DataFrame:
    """Add boolean 'qualified' column using qualify_row rules."""
    d = dets.copy()
    d["qualified"] = d.apply(qualify_row, axis=1)
    return d


def enrich_detections(
    dets: pd.DataFrame,
    df_ohlcv: pd.DataFrame,
    ind_df: pd.DataFrame,
    symbol: str,
    vol_lookback: int = 20,
    high_lookback: int = 252
) -> pd.DataFrame:
    """
    Convenience wrapper: direction/alignment → features → english → qualification.
    Returns an enriched detections DataFrame.
    """
    d = add_direction_alignment(dets)
    feat = compute_context_features(df_ohlcv, ind_df, vol_lookback=vol_lookback, high_lookback=high_lookback)
    d = merge_features(d, feat)
    d = add_english(d, symbol)
    d = add_qualification(d)
    return d
