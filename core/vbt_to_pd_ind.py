import numpy as np
import pandas as pd

# canonical mapping (case-insensitive)
_CANON = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "close",
    "adjclose": "close",
    "volume": "volume",
}
_REQUIRED = ["open","high","low","close","volume"]

def _norm(x) -> str:
    return str(x).strip().lower()

def _pick_series_from_obj(obj, symbol=None):
    """Return a 1D pandas Series from a value that might be Series/DataFrame."""
    val = obj
    if isinstance(val, pd.DataFrame):
        if symbol is not None and symbol in val.columns:
            return val[symbol]
        # fallback: first column
        return val.iloc[:, 0]
    return val

def adx(df_ohlcv: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder's ADX — trend-strength (0-100), direction-agnostic."""
    cols = {str(c).lower(): c for c in df_ohlcv.columns}
    h = df_ohlcv[cols["high"]].astype(float)
    l = df_ohlcv[cols["low"]].astype(float)
    c = df_ohlcv[cols["close"]].astype(float)
    up, dn = h.diff(), -l.diff()
    plus_dm = ((up > dn) & (up > 0)) * up
    minus_dm = ((dn > up) & (dn > 0)) * dn
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_w = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr_w
    mdi = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr_w
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def build_ind_df_from_vbt(df_ohlcv: pd.DataFrame, vbt_ind: dict, symbol: str | None = None) -> pd.DataFrame:
    """
    Builds indicator DataFrame with columns:
      RSI, MA20, MA50, MACD, MACD_signal, ATR, BB_upper, BB_lower
    `df_ohlcv` must already be single-symbol OHLCV with 'Close' column.
    """
    out = pd.DataFrame(index=df_ohlcv.index)

    # RSI
    rsi_obj = vbt_ind["rsi"]
    for name in ("rsi", "rsi_0"):
        if hasattr(rsi_obj, name):
            out["RSI"] = _pick_series_from_obj(getattr(rsi_obj, name), symbol)
            break

    # MA
    for key, col in (("ma20","MA20"), ("ma50","MA50")):
        ma_obj = vbt_ind[key]
        for name in ("ma", "sma", "ma_0", "sma_0"):
            if hasattr(ma_obj, name):
                out[col] = _pick_series_from_obj(getattr(ma_obj, name), symbol)
                break

    # MACD
    macd_obj = vbt_ind["macd"]
    for cand in ("macd", "macd_0"):
        if hasattr(macd_obj, cand):
            out["MACD"] = _pick_series_from_obj(getattr(macd_obj, cand), symbol); break
    for cand in ("signal", "signal_0", "macdsignal"):
        if hasattr(macd_obj, cand):
            out["MACD_signal"] = _pick_series_from_obj(getattr(macd_obj, cand), symbol); break

    # ATR
    atr_obj = vbt_ind["atr"]
    for cand in ("atr", "atr_0"):
        if hasattr(atr_obj, cand):
            out["ATR"] = _pick_series_from_obj(getattr(atr_obj, cand), symbol); break

    # Bollinger Bands
    bb_obj = vbt_ind["bb"]
    for target, cands in (("BB_lower", ("lower","bb_lower","lband","bbl","lb")),
                          ("BB_upper", ("upper","bb_upper","uband","bbu","ub"))):
        for cand in cands:
            if hasattr(bb_obj, cand):
                out[target] = _pick_series_from_obj(getattr(bb_obj, cand), symbol)
                break

    # ADX (computed directly from OHLC; vectorbt has no built-in)
    out["ADX"] = adx(df_ohlcv)

    # sanity
    required = ["RSI","MA20","MA50","MACD","MACD_signal","ATR","BB_upper","BB_lower"]
    missing = [c for c in required if c not in out]
    if missing:
        raise ValueError(f"build_ind_df_from_vbt: missing {missing}")

    return out

def extract_symbol_df(df: pd.DataFrame, symbol: str | None = None, case: str = "lower") -> pd.DataFrame:
    """
    Normalize to a single-symbol OHLCV with columns in desired case (default: lowercase):
      ['open','high','low','close','volume']
    Works for single-level columns or MultiIndex (field, symbol) / (symbol, field).
    Set case='title' or 'upper' if you prefer those.
    """
    # --- slice to one symbol & flatten fields ---
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = {_norm(v) for v in df.columns.get_level_values(0)}
        lvl1 = {_norm(v) for v in df.columns.get_level_values(1)}
        keys = set(_CANON.keys())
        if len(lvl0 & keys) >= 3:
            field_level, sym_level = 0, 1
        elif len(lvl1 & keys) >= 3:
            field_level, sym_level = 1, 0
        else:
            raise ValueError("Could not find OHLCV field level in MultiIndex columns.")

        symbols = df.columns.get_level_values(sym_level).unique().tolist()
        if symbol is None:
            sel_symbol = symbols[0] if symbols else None
        else:
            matches = [s for s in symbols if _norm(s) == _norm(symbol)]
            sel_symbol = matches[0] if matches else None
        if sel_symbol is None:
            raise ValueError("No symbols found / match failed; pass a valid `symbol`.")

        out = df.xs(sel_symbol, axis=1, level=sym_level).copy()
    else:
        out = df.copy()

    # --- rename columns to canonical lowercase ---
    rename_map = {}
    for c in out.columns:
        nc = _norm(c)
        if nc in _CANON:
            rename_map[c] = _CANON[nc]
    out = out.rename(columns=rename_map)

    # allow 'adj close' fallback
    if "close" not in out.columns and "adj close" in { _norm(c) for c in out.columns }:
        # re-run rename to map adj close → close if needed
        rename_map = {c: ("close" if _norm(c) in ("adj close","adjclose") else c) for c in out.columns}
        out = out.rename(columns=rename_map)

    missing = [c for c in _REQUIRED if c not in out.columns]
    if missing:
        raise ValueError(f"extract_symbol_df: missing columns {missing}. Got {list(out.columns)}")

    # order & tz-naive index
    out = out[_REQUIRED]
    if getattr(out.index, "tz", None) is not None:
        try:
            out.index = out.index.tz_convert(None)
        except Exception:
            out.index = out.index.tz_localize(None)

    # --- column case preference ---
    if case == "lower":
        return out
    if case == "title":
        return out.rename(columns=str.title)
    if case == "upper":
        return out.rename(columns=str.upper)
    raise ValueError("case must be one of {'lower','title','upper'}")