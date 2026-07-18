# core/market_context.py
"""
Market/sector context for a symbol: index regime + relative strength.

Produces a per-date DataFrame aligned to the stock's trading index with:
  mkt_up / mkt_dn     — NIFTY 50 regime (MA20 vs MA50)
  sect_up / sect_dn   — sector index regime (falls back to NIFTY)
  rs_20, rs_60        — stock return minus sector return over 20/60 bars
"""
from __future__ import annotations
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

MARKET_INDEX = "^NSEI"

# yfinance tickers for NSE sector indices
SECTOR_INDEX = {
    "BANK": "^NSEBANK", "IT": "^CNXIT", "PHARMA": "^CNXPHARMA",
    "AUTO": "^CNXAUTO", "METAL": "^CNXMETAL", "ENERGY": "^CNXENERGY",
    "FMCG": "^CNXFMCG", "INFRA": "^CNXINFRA", "REALTY": "^CNXREALTY",
    "MEDIA": "^CNXMEDIA", "PSUBANK": "^CNXPSUBANK", "FIN": "NIFTY_FIN_SERVICE.NS",
}

# symbol → sector key (extend as the universe grows; unknown symbols use NIFTY)
SYMBOL_SECTOR = {
    "HDFCBANK.NS": "BANK", "ICICIBANK.NS": "BANK", "SBIN.NS": "BANK",
    "AXISBANK.NS": "BANK", "KOTAKBANK.NS": "BANK", "INDUSINDBK.NS": "BANK",
    "BAJFINANCE.NS": "FIN",
    "TCS.NS": "IT", "INFY.NS": "IT", "HCLTECH.NS": "IT", "WIPRO.NS": "IT",
    "TECHM.NS": "IT",
    "SUNPHARMA.NS": "PHARMA", "CIPLA.NS": "PHARMA", "DRREDDY.NS": "PHARMA",
    "SHALBY.NS": "PHARMA",
    "MARUTI.NS": "AUTO", "TATAMOTORS.NS": "AUTO", "M&M.NS": "AUTO",
    "EICHERMOT.NS": "AUTO", "VEDL.NS": "METAL",
    "TATASTEEL.NS": "METAL", "JSWSTEEL.NS": "METAL", "HINDALCO.NS": "METAL",
    "HINDCOPPER.NS": "METAL", "GANESHCO.NS": "METAL",
    "ONGC.NS": "ENERGY", "NTPC.NS": "ENERGY", "POWERGRID.NS": "ENERGY",
    "RELIANCE.NS": "ENERGY", "SWSOLAR.NS": "ENERGY",
    "ITC.NS": "FMCG", "HINDUNILVR.NS": "FMCG",
    "LT.NS": "INFRA", "ADANIPORTS.NS": "INFRA", "ADANIENT.NS": "INFRA",
    "ULTRACEMCO.NS": "INFRA",
}

_index_cache: dict[tuple[str, int], pd.Series] = {}
_BREADTH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "breadth.csv")


_breadth_cache: list = []          # [series] once loaded
_dd_cache: dict[int, pd.Series | None] = {}


def load_breadth() -> pd.Series | None:
    """Universe breadth (% above 50DMA) precomputed by build_breadth.py."""
    if _breadth_cache:
        return _breadth_cache[0]
    try:
        b = pd.read_csv(_BREADTH_PATH, index_col=0, parse_dates=True)["breadth"].astype(float)
    except Exception:
        b = None
    _breadth_cache.append(b)
    return b


def distribution_days(years: float = 1.0, window: int = 25) -> pd.Series | None:
    """O'Neil distribution-day count on NIFTY: sessions where the index fell
    >=0.2% on volume higher than the prior day, rolling `window` sessions.
    4-5+ inside 25 sessions historically precedes corrections."""
    key = int(years * 4)
    if key in _dd_cache:
        return _dd_cache[key]
    end = datetime.now()
    start = end - timedelta(days=int(365 * years) + 60)
    try:
        df = yf.download(MARKET_INDEX, start=start, end=end, auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        close, vol = df["Close"], df["Volume"]
        if isinstance(close, pd.DataFrame):
            close, vol = close.iloc[:, 0], vol.iloc[:, 0]
        close.index = pd.to_datetime(close.index)
        if getattr(close.index, "tz", None) is not None:
            close.index = close.index.tz_localize(None)
            vol.index = close.index
        ret = close.pct_change()
        dd = ((ret <= -0.002) & (vol > vol.shift(1))).astype(float)
        out = dd.rolling(window).sum()
        _dd_cache[key] = out
        return out
    except Exception as e:
        print(f"market_context: distribution_days failed: {e}")
        _dd_cache[key] = None
        return None


def _load_index_close(ticker: str, years: float, cache_dir: str | None = None) -> pd.Series | None:
    key = (ticker, int(years * 4))
    if key in _index_cache:
        return _index_cache[key]
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"idx_{ticker.replace('^','').replace('.','_')}_{years}y.pkl")
        if os.path.exists(path):
            s = pd.read_pickle(path)
            _index_cache[key] = s
            return s
    end = datetime.now()
    start = end - timedelta(days=int(365 * years) + 10)
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index)
        if getattr(close.index, "tz", None) is not None:
            close.index = close.index.tz_localize(None)
        close = close.astype(float)
        if cache_dir:
            close.to_pickle(path)
        _index_cache[key] = close
        return close
    except Exception as e:
        print(f"market_context: failed to load {ticker}: {e}")
        return None


def build_market_context(
    symbol: str,
    stock_close: pd.Series,
    years: float = 1.0,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Return context DataFrame aligned to stock_close.index. Missing data → permissive NaN/True."""
    idx = stock_close.index
    out = pd.DataFrame(index=idx)

    mkt = _load_index_close(MARKET_INDEX, max(years, 1.0) + 0.5, cache_dir)
    sect_key = SYMBOL_SECTOR.get(symbol)
    sect = _load_index_close(SECTOR_INDEX[sect_key], max(years, 1.0) + 0.5, cache_dir) if sect_key else None
    if sect is None:
        sect = mkt

    def _regime(close: pd.Series | None) -> tuple[pd.Series, pd.Series]:
        if close is None:
            t = pd.Series(True, index=idx)
            return t, t.copy()  # permissive: no data, no gate
        c = close.reindex(idx).ffill()
        ma20 = c.rolling(20).mean()
        ma50 = c.rolling(50).mean()
        return (ma20 > ma50).fillna(True), (ma20 < ma50).fillna(True)

    out["mkt_up"], out["mkt_dn"] = _regime(mkt)
    out["sect_up"], out["sect_dn"] = _regime(sect)

    if sect is not None:
        s = sect.reindex(idx).ffill()
        for n in (20, 60):
            out[f"rs_{n}"] = stock_close.pct_change(n) - s.pct_change(n)
    else:
        out["rs_20"] = np.nan
        out["rs_60"] = np.nan

    # market-health extras (best-effort; NaN → gates stay permissive)
    b = load_breadth()
    out["breadth"] = b.reindex(idx).ffill() if b is not None else np.nan
    dd = distribution_days(max(years, 1.0) + 0.3)
    out["dist_days"] = dd.reindex(idx).ffill() if dd is not None else np.nan
    return out
