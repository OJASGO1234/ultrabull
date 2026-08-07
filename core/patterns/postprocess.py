# File: patterns/postprocess.py

import pandas as pd
import numpy as np

def direction_of(pattern: str) -> str:
    """Map a pattern name to directional bias."""
    if pattern.startswith("Breakout"):   return "bullish"
    if pattern.startswith("Breakdown"):  return "bearish"
    if pattern in {"BB_BuyDip","PullbackToMA20","BullishEngulfing","Hammer",
                   "RSI_40_Up","MACD_Cross_Zeroish_Up","VCP_Setup"}:
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
    if "ADX" in ind_df.columns:
        feat["adx"] = ind_df["ADX"]
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


_REVERSAL_PATTERNS = {"Hammer", "BullishEngulfing"}


def rescore(dets: pd.DataFrame, use_market_context: bool = True) -> pd.DataFrame:
    """
    Rebuild 'score' from the family base plus evidence-based modifiers, so the
    score actually ranks setups instead of being a per-family constant:
      +10/+5  volume surge (>=2x / >=1.5x 20d average)
      +10/-10 regime alignment (reversal patterns get +5 when counter-trend
              instead of the penalty — that's their natural habitat)
      +10/-5  relative strength vs sector agrees / disagrees with direction
      +5      ADX agrees with the pattern family (trending for breakouts,
              calm for mean-reversion)
      +5      breakout within 5% of its 52-week high
    Clipped to [0, 100]; 'strength' is recomputed from the new score.
    """
    from .scoring import base, strength

    d = dets.copy()
    fam = d["pattern"].map(family_of)
    sc = fam.map(base).astype(float)

    volx = pd.to_numeric(d.get("vol_mult20"), errors="coerce")
    sc += np.select([volx >= 2.0, volx >= 1.5], [10.0, 5.0], default=0.0)

    align = d.get("alignment", pd.Series("neutral", index=d.index)).astype(str)
    is_rev = d["pattern"].isin(_REVERSAL_PATTERNS)
    sc += np.select(
        [align.eq("aligned"), align.eq("counter-trend") & is_rev, align.eq("counter-trend")],
        [10.0, 5.0, -10.0], default=0.0)

    # Relative strength is sector- (or, for unmapped symbols, NIFTY-) derived,
    # so it is skipped in the stock-only view. Without this the +10/-5 term rode
    # into stock-only detection scores unannounced — the one market input in
    # that view that nothing declared. The score is descriptive only (no gate
    # reads it; add_qualification is score-free), so this changes what is
    # DISPLAYED, never which trades fire.
    if use_market_context and "rs_20" in d.columns:
        rs = pd.to_numeric(d["rs_20"], errors="coerce")
        direction = d.get("direction", pd.Series("neutral", index=d.index)).astype(str)
        rs_agrees = (direction.eq("bullish") & (rs > 0)) | (direction.eq("bearish") & (rs < 0))
        rs_fights = (direction.eq("bullish") & (rs < 0)) | (direction.eq("bearish") & (rs > 0))
        sc += np.select([rs_agrees, rs_fights], [10.0, -5.0], default=0.0)

    if "adx" in d.columns:
        adx = pd.to_numeric(d["adx"], errors="coerce")
        adx_fit = (fam.eq("breakout") & (adx >= 20)) | (fam.eq("pullback") & (adx <= 32))
        sc += np.where(adx_fit.fillna(False), 5.0, 0.0)

    if "off_52w_high_pct" in d.columns:
        off = pd.to_numeric(d["off_52w_high_pct"], errors="coerce")
        near_high = fam.eq("breakout") & d["pattern"].str.startswith("Breakout") & (off >= -5.0)
        sc += np.where(near_high.fillna(False), 5.0, 0.0)

    d["score"] = np.clip(sc, 0, 100).round(0)
    d["strength"] = strength(d["score"])
    return d


def qualify_row(row: pd.Series) -> bool:
    """
    Decide if a detection qualifies for a watchlist.

    Deliberately score-free: the 35-symbol event study showed the composite
    score does NOT rank forward returns within most patterns (confirmation is
    late), so qualification uses only the discriminators that validated:
      - regime alignment, OR counter-trend reversal for Hammer/BullishEngulfing
      - for breakouts/breakdowns, basic volume confirmation if available
    The score remains as descriptive/reporting output.
    """
    if row.get("direction") == "neutral":
        return False
    reversal_ok = row.get("pattern") in _REVERSAL_PATTERNS and row.get("alignment") == "counter-trend"
    aligned_ok = row.get("alignment") == "aligned"
    vol_ok = True
    pat = str(row.get("pattern",""))
    if (pat.startswith("Breakout") or pat.startswith("Breakdown")):
        vol = row.get("vol_mult20", row.get("vol_mult", np.nan))
        if not pd.isna(vol):
            vol_ok = float(vol) >= 1.2
    return (aligned_ok or reversal_ok) and vol_ok


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
    high_lookback: int = 252,
    ctx: pd.DataFrame | None = None,
    use_market_context: bool = True,
) -> pd.DataFrame:
    """
    Convenience wrapper: direction/alignment → features → rescore → english → qualification.
    Pass `ctx` (market_context.build_market_context) to include relative
    strength in features and scoring.
    Returns an enriched detections DataFrame.
    """
    d = add_direction_alignment(dets)
    feat = compute_context_features(df_ohlcv, ind_df, vol_lookback=vol_lookback, high_lookback=high_lookback)
    if ctx is not None:
        for c in ("rs_20", "rs_60"):
            if c in ctx.columns:
                feat[c] = ctx[c]
    d = merge_features(d, feat)
    d = rescore(d, use_market_context=use_market_context)
    d = add_english(d, symbol)
    d = add_qualification(d)
    return d

def family_of(pattern: str) -> str:
    if pattern.startswith(("Breakout","Breakdown")): return "breakout"
    if pattern.startswith(("RSI_","MACD_")):         return "momentum"
    if pattern.startswith(("BB_","Pullback")):       return "pullback"
    return "candles"

def summarize_signals(d: pd.DataFrame) -> dict:
    """
    Quick sanity readout for one symbol:
      - counts overall & per week
      - counts by family (breakout/pullback/momentum/candles)
      - top examples table
    """
    d = d.copy()
    d["week"] = pd.to_datetime(d["date"]).dt.to_period("W").dt.start_time
    out = {}
    out["total_hits"] = len(d)
    out["per_week_all"] = d.groupby("week").size().mean()
    out["qualified_hits"] = int(d.get("qualified", pd.Series(dtype=bool)).sum())
    out["per_week_qualified"] = (
        d[d.get("qualified", False)].groupby("week").size().mean()
        if "qualified" in d else 0
    )
    # family_of is already in postprocess.py
    out["by_family"] = d.groupby(d["pattern"].map(family_of)).size().to_dict()
    out["top_examples"] = d.sort_values("score", ascending=False)[
        ["date","pattern","score","direction","alignment"]
    ].head(5)
    return out


def warmup_cutoff(ind_df: pd.DataFrame) -> pd.Timestamp:
    """Latest first-valid date across core columns; use to filter early, unreliable hits."""
    cols = ["MA20","MA50","RSI","ATR","BB_upper","BB_lower","MACD","MACD_signal"]
    firsts = [ind_df[c].first_valid_index() for c in cols if c in ind_df.columns]
    firsts = [t for t in firsts if t is not None]
    return max(firsts) if firsts else ind_df.index[0]