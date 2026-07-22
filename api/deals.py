"""Bulk / block / short-sell deals from NSE.

Groww's API doesn't expose these, so we read NSE's own JSON endpoint (the one
their website uses). IMPORTANT: these are SEBI end-of-day disclosures — there is
NO live/intraday feed. During the trading day the endpoint serves the previous
completed day's deals; the current day's are published after close. So this is
same-day-after-close fresh at best, not real-time. We cache aggressively since
it only changes once a day.

NSE blocks non-browser requests, so we seed cookies from the homepage first and
send browser headers. Best-effort — any hiccup degrades to an empty result.
"""
from __future__ import annotations

import threading
import time

import requests
from fastapi import APIRouter, Query

router = APIRouter(prefix="/api/deals", tags=["deals"])

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
       "(KHTML, like Gecko) Version/17 Safari/605.1.15")
_HEADERS = {"User-Agent": _UA, "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9", "Referer": "https://www.nseindia.com/market-data/large-deals"}
_URL = "https://www.nseindia.com/api/snapshot-capital-market-largedeal"
CACHE_TTL = 1800.0  # 30 min — the data changes at most once a day

_lock = threading.Lock()
_cache: tuple[float, dict] | None = None


def _f(x):
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _rows(raw: list | None, kind: str) -> list[dict]:
    out = []
    for r in raw or []:
        out.append({
            "type": kind,
            "symbol": (r.get("symbol") or "").strip().upper() or None,
            "name": r.get("name"),
            "client": r.get("clientName"),
            "side": (r.get("buySell") or "").strip().upper() or None,
            "qty": _f(r.get("qty")),
            "price": _f(r.get("watp")),   # weighted average traded price
            "date": r.get("date"),
            "remarks": r.get("remarks") if r.get("remarks") not in ("-", None) else None,
        })
    return out


def _fetch() -> dict:
    """(cached) all large deals from NSE, normalized."""
    global _cache
    with _lock:
        if _cache and time.time() - _cache[0] < CACHE_TTL:
            return _cache[1]
    try:
        s = requests.Session()
        s.get("https://www.nseindia.com", headers=_HEADERS, timeout=12)  # seed cookies
        d = s.get(_URL, headers=_HEADERS, timeout=12).json()
        result = {
            "as_on_date": d.get("as_on_date"),
            "bulk": _rows(d.get("BULK_DEALS_DATA"), "bulk"),
            "block": _rows(d.get("BLOCK_DEALS_DATA"), "block"),
            "short": _rows(d.get("SHORT_DEALS_DATA"), "short"),
            "ok": True,
        }
    except Exception as e:
        result = {"as_on_date": None, "bulk": [], "block": [], "short": [], "ok": False, "error": str(e)}
    with _lock:
        _cache = (time.time(), result)
    return result


@router.get("")
def deals(symbol: str = Query(None, description="Yahoo symbol to filter to, e.g. ITC.NS")):
    """All large deals, or (with ?symbol=) just those for one stock."""
    data = _fetch()
    if not symbol:
        return {**data, "bulk": data["bulk"][:100], "block": data["block"][:50], "short": data["short"][:100]}
    base = symbol.strip().upper().replace(".NS", "").replace(".BO", "")
    match = lambda rows: [r for r in rows if r["symbol"] == base]
    return {
        "symbol": base,
        "as_on_date": data["as_on_date"],
        "ok": data["ok"],
        "bulk": match(data["bulk"]),
        "block": match(data["block"]),
        "short": match(data["short"]),
    }
