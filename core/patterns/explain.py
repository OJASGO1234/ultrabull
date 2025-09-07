# core/patterns/explain.py
import pandas as pd
import numpy as np
import re

# ---- small helpers ------------------------------------------------------------
def _safe_volx(df_ohlcv: pd.DataFrame, dt, lookback: int = 20, min_obs: int = 15):
    """Return volume / avg(volume, lookback) or NaN if too few obs or avg≈0."""
    vol = df_ohlcv["Volume"]
    # Align date to trading index
    idx = df_ohlcv.index
    if dt not in idx:
        pos = idx.get_indexer([pd.Timestamp(dt)], method="nearest")[0]
        dt = idx[pos]
    win = vol.rolling(lookback)
    cnt = float(win.count().loc[dt])
    avg = float(win.mean().loc[dt]) if cnt >= min_obs else np.nan
    if not np.isfinite(avg) or avg == 0.0:
        return np.nan
    return float(vol.loc[dt]) / avg



def _fmt_num(x, n=2):
    if pd.isna(x): return "n/a"
    try: return f"{float(x):.{n}f}"
    except Exception: return "n/a"

def _fmt_pct(x, n=1):
    if pd.isna(x): return "n/a"
    try: return f"{float(x):+.{n}f}%"
    except Exception: return "n/a"

def _family_of(p: str) -> str:
    if p.startswith(("Breakout","Breakdown")): return "breakout"
    if p.startswith(("RSI_","MACD_")):         return "momentum"
    if p.startswith(("BB_","Pullback")):       return "pullback"
    return "candles"

def _regime_at(ind_df: pd.DataFrame, dt) -> str:
    """Compute regime from MA20 vs MA50 on a specific date (fallback to 'neutral')."""
    try:
        ma20 = float(ind_df.loc[dt, "MA20"])
        ma50 = float(ind_df.loc[dt, "MA50"])
        if ma20 > ma50: return "uptrend"
        if ma20 < ma50: return "downtrend"
        return "neutral"
    except Exception:
        # If MAs missing, try to read a regime column if present
        try:
            return str(ind_df.loc[dt, "_regime"])
        except Exception:
            return "neutral"

# ---- core feature line for a given date --------------------------------------
def _day_feature_line(df_ohlcv: pd.DataFrame, ind_df: pd.DataFrame, dt) -> str:
    """Headline stats: close, 1d change, RSI, vol×20d, ATR, off-52w-high, regime."""
    idx = df_ohlcv.index
    if dt not in idx:  # align to nearest trading day in price index
        pos = idx.get_indexer([pd.Timestamp(dt)], method="nearest")[0]
        dt = idx[pos]

    close = df_ohlcv.loc[dt, "Close"]
    # previous trading day for pct change
    prev_idx = idx[idx.get_loc(dt)-1] if idx.get_loc(dt) > 0 else None
    prev_close = df_ohlcv.loc[prev_idx, "Close"] if prev_idx is not None else np.nan
    chg_pct = (close / prev_close - 1.0) * 100 if pd.notna(prev_close) else np.nan

    vol = df_ohlcv["Volume"]
    # volx = (vol / vol.rolling(20).mean()).loc[dt]
    volx = _safe_volx(df_ohlcv, dt, lookback=20, min_obs=15)

    rsi = ind_df.loc[dt, "RSI"] if "RSI" in ind_df else np.nan
    atr = ind_df.loc[dt, "ATR"] if "ATR" in ind_df else np.nan
    high_52w = df_ohlcv["Close"].rolling(252, min_periods=1).max().loc[dt]
    off_high = (close / high_52w - 1.0) * 100 if high_52w else np.nan

    regime = _regime_at(ind_df, dt)

    head = (
        f"**{pd.Timestamp(dt).date().isoformat()} — Close { _fmt_num(close) } "
        f"({_fmt_pct(chg_pct)} d/d)**  \n"
        f"Regime: **{regime}** · RSI={_fmt_num(rsi,1)} · Vol={_fmt_num(volx,1)}×20d · "
        f"ATR={_fmt_num(atr,2)} · off-52w-high={_fmt_pct(off_high,1)}"
    )
    return head

def _coalesce_breaks_for_display(day_hits: pd.DataFrame) -> pd.DataFrame:
    """
    Combine same-day BreakoutN / BreakdownN rows into a single row like
    'Breakout20/55' (keeps the highest-score row as the representative).
    """
    if day_hits.empty or "pattern" not in day_hits:
        return day_hits

    d = day_hits.copy()

    def _combine(dframe: pd.DataFrame, kind: str) -> pd.DataFrame:
        mask = dframe["pattern"].str.match(fr"^{kind}\d+$", na=False)
        if not mask.any():
            return dframe
        # collect N values (e.g., 20, 55) present that day
        Ns = (dframe.loc[mask, "pattern"]
                    .str.extract(r"(\d+)", expand=False)
                    .dropna()
                    .astype(int)
                    .unique())
        Ns = sorted(Ns.tolist())
        if len(Ns) <= 1:
            return dframe
        # representative row = highest score among that kind
        rep = dframe.loc[mask].sort_values("score", ascending=False).iloc[0].copy()
        rep["pattern"] = f"{kind}" + "/".join(map(str, Ns))
        # drop individual rows, append the combined one
        combined = pd.concat([dframe.loc[~mask], rep.to_frame().T], ignore_index=True)
        return combined

    d = _combine(d, "Breakout")
    d = _combine(d, "Breakdown")
    return d

# ---- per-day summary ----------------------------------------------------------
def daily_summary_md(
    detections_enriched: pd.DataFrame,
    df_ohlcv: pd.DataFrame,
    ind_df: pd.DataFrame,
    symbol: str,
    date: pd.Timestamp | str | None = None,
    *,
    top_n: int = 3,
    only_qualified: bool = True
) -> str:
    """
    Return a Markdown summary for one trading day:
      - headline stats (close, change, RSI, vol×avg, ATR, off-52w-high, regime)
      - signal mix counts (bullish/bearish & by family)
      - top N signals with one-line explanations (uses 'english' if present)
    """
    if detections_enriched.empty:
        return f"### {symbol}\n_No detections available._"

    # Resolve the date to the latest trading day by default
    idx = df_ohlcv.index
    day = pd.Timestamp(date).normalize() if date is not None else pd.Timestamp(idx[-1]).normalize()

    # Filter detections for the chosen day (normalize to date)
    d = detections_enriched.copy()
    d["__d"] = pd.to_datetime(d["date"]).dt.normalize()
    day_hits = d[d["__d"] == day].copy()

    if only_qualified and "qualified" in day_hits:
        day_hits = day_hits[day_hits["qualified"]].copy()
    display_hits = _coalesce_breaks_for_display(day_hits)

    # Build headline
    head = f"### {symbol}\n" + _day_feature_line(df_ohlcv, ind_df, day)

    # If no signals that day, early return
    if day_hits.empty:
        return head + "\n\n_No notable patterns detected today._\n\n" \
                      "_Patterns are setups, not predictions. Use stops and position sizing._"

    # Mix by direction
    by_dir = day_hits["direction"].value_counts().to_dict() if "direction" in day_hits else {}
    bull = by_dir.get("bullish", 0); bear = by_dir.get("bearish", 0)

    # By family (breakout/pullback/momentum/candles)
    fam_counts = day_hits["pattern"].map(_family_of).value_counts().to_dict()


    top_source = display_hits
    # Pick top N by score (fall back by pattern order)
    if "score" in top_source:
        top = top_source.sort_values("score", ascending=False).head(top_n)
    else:
        top = top_source.head(top_n)

    # Build markdown lines
    lines = []
    lines.append(f"\n**Signals today:** {bull} bullish · {bear} bearish · by family: " +
                 ", ".join(f"{k}={v}" for k,v in fam_counts.items()))

    lines.append("\n**Top signals**")
    for _, r in top.iterrows():
        if "english" in r and isinstance(r["english"], str) and len(r["english"]) > 0:
            # strip date+symbol at the start if your line already contains them
            lines.append(f"- {r['english']}")
        else:
            # simple fallback
            base = f"{pd.to_datetime(r['date']).date().isoformat()} — {symbol} {r['pattern']}"
            extra = []
            if "direction" in r: extra.append(r["direction"])
            if "alignment" in r: extra.append(r["alignment"])
            if "score" in r:     extra.append(f"score={int(r['score'])}")
            lines.append(f"- {base} ({', '.join(extra)})")

    # Guardrail footer
    lines.append("\n_Patterns are setups, not predictions. Always use defined risk._")

    return head + "\n" + "\n".join(lines)

# ---- recent snapshot (last N trading days) ------------------------------------
def recent_summary_md(
    detections_enriched: pd.DataFrame,
    df_ohlcv: pd.DataFrame,
    ind_df: pd.DataFrame,
    symbol: str,
    days: int = 5,
    *,
    top_n_each: int = 2,
) -> str:
    """Markdown snapshot for the last `days` trading sessions (each day lists top N signals)."""
    if df_ohlcv.empty:
        return f"### {symbol}\n_No price data._"
    idx = df_ohlcv.index[-days:]
    md = [f"## {symbol} — Last {len(idx)} sessions"]
    for dt in idx:
        md.append(daily_summary_md(detections_enriched, df_ohlcv, ind_df, symbol, date=dt, top_n=top_n_each))
    return "\n\n".join(md)
