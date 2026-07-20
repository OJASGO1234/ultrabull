"""Live quotes via the Groww trading API (data subscription).

Token management is fully hands-off: access tokens are minted from the API
key + TOTP secret in .env (groww_api_key / groww_totp_gen) and rebuilt on
auth failure. Quotes come from the two batch endpoints (get_ltp + get_ohlc =
2 upstream calls regardless of symbol count) behind a ~3s micro-cache so a
page full of pollers doesn't hammer Groww's rate limits.
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/live", tags=["live"])

IST = ZoneInfo("Asia/Kolkata")
CACHE_TTL = 3.0

_lock = threading.Lock()
_client = None
_client_born = 0.0
_cache: dict[str, tuple[float, dict]] = {}


def _new_client():
    from growwapi import GrowwAPI
    import pyotp

    api_key = os.getenv("groww_api_key")
    totp_gen = os.getenv("groww_totp_gen")
    if not api_key or not totp_gen:
        raise HTTPException(status_code=503, detail="Groww API keys not configured (groww_api_key / groww_totp_gen)")
    token = GrowwAPI.get_access_token(api_key, pyotp.TOTP(totp_gen).now())
    return GrowwAPI(token)


def _get_client(force_new: bool = False):
    global _client, _client_born
    with _lock:
        # tokens are day-scoped; rebuild proactively after 8h or on demand
        if force_new or _client is None or time.time() - _client_born > 8 * 3600:
            _client = _new_client()
            _client_born = time.time()
        return _client


def market_open(now: datetime | None = None) -> bool:
    now = now or datetime.now(IST)
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return 9 * 60 <= minutes <= 15 * 60 + 40  # pre-open through a bit past close


def _split(symbol: str) -> tuple[str, str] | None:
    """'BEPL.NS' -> ('NSE', 'BEPL'); '.BO' -> BSE; anything else unsupported."""
    s = symbol.strip().upper()
    if s.endswith(".NS"):
        return "NSE", s[:-3]
    if s.endswith(".BO"):
        return "BSE", s[:-3]
    return None


@router.get("")
def live(symbols: str = Query(..., description="Comma-separated Yahoo symbols")):
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()][:40]
    if not syms:
        return {"quotes": {}, "market_open": market_open(), "source": "groww"}

    key = ",".join(sorted(syms))
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]

    pairs = {s: _split(s) for s in syms}
    exch_syms = tuple(f"{p[0]}_{p[1]}" for p in pairs.values() if p)
    if not exch_syms:
        return {"quotes": {}, "market_open": market_open(), "source": "groww"}

    def _fetch(client):
        ltp = client.get_ltp(exchange_trading_symbols=exch_syms, segment=client.SEGMENT_CASH)
        ohlc = client.get_ohlc(exchange_trading_symbols=exch_syms, segment=client.SEGMENT_CASH)
        return ltp, ohlc

    try:
        try:
            ltp, ohlc = _fetch(_get_client())
        except HTTPException:
            raise
        except Exception:
            ltp, ohlc = _fetch(_get_client(force_new=True))  # stale token? re-auth once
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Groww live data unavailable: {e}")

    now_ts = int(time.time())
    quotes = {}
    for sym, pair in pairs.items():
        if not pair:
            continue
        k = f"{pair[0]}_{pair[1]}"
        price = ltp.get(k)
        bars = ohlc.get(k) or {}
        if price is None:
            continue
        prev = bars.get("close")  # Groww's ohlc.close is the PREVIOUS session close
        quotes[sym] = {
            "ltp": float(price),
            "open": bars.get("open"),
            "high": bars.get("high"),
            "low": bars.get("low"),
            "prev_close": prev,
            "day_change_pct": round(100.0 * (float(price) - prev) / prev, 2) if prev else None,
            "ts": now_ts,
        }

    out = {"quotes": quotes, "market_open": market_open(), "source": "groww"}
    _cache[key] = (time.time(), out)
    if len(_cache) > 16:
        _cache.pop(min(_cache, key=lambda k2: _cache[k2][0]))
    return out
