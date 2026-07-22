"""Live quotes via the Groww trading API (data subscription).

Token management is fully hands-off: access tokens are minted from the API
key + TOTP secret in .env (groww_api_key / groww_totp_gen) and rebuilt on
auth failure. Quotes come from the two batch endpoints (get_ltp + get_ohlc =
2 upstream calls regardless of symbol count) behind a ~3s micro-cache so a
page full of pollers doesn't hammer Groww's rate limits.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query, WebSocket

from .auth import verify_token
from .sre import yahoo_symbol as _to_yahoo

router = APIRouter(prefix="/api/live", tags=["live"])
groww_router = APIRouter(prefix="/api/groww", tags=["groww"])

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


def ltp_map(symbols: list[str]) -> dict[str, float]:
    """{yahoo_symbol: last_price} for the given symbols via one Groww batch call.
    Best-effort — returns what it can; empty on failure (callers mark-to-market
    with their last known price)."""
    pairs = {s: _split(s) for s in symbols}
    exch = tuple(f"{p[0]}_{p[1]}" for p in pairs.values() if p)
    if not exch:
        return {}
    try:
        try:
            ltp = _get_client().get_ltp(exchange_trading_symbols=exch, segment="CASH")
        except Exception:
            ltp = _get_client(force_new=True).get_ltp(exchange_trading_symbols=exch, segment="CASH")
    except Exception:
        return {}
    out: dict[str, float] = {}
    for s, p in pairs.items():
        if p:
            v = ltp.get(f"{p[0]}_{p[1]}")
            if v is not None:
                out[s] = float(v)
    return out


_depth_cache: dict[str, tuple[float, dict]] = {}


@router.get("/depth")
def depth(symbol: str = Query(..., description="Yahoo symbol, e.g. ITC.NS")):
    """Level-2 order book (5 bids / 5 asks) + total buy/sell quantity from
    Groww. Empty/thin outside market hours. ~2.5s micro-cache so the order book
    and the order-flow gauge can both poll without doubling Groww calls."""
    sym = symbol.strip().upper()
    hit = _depth_cache.get(sym)
    if hit and time.time() - hit[0] < 2.5:
        return hit[1]
    pair = _split(sym)
    if not pair:
        raise HTTPException(status_code=422, detail="need an NSE/BSE symbol (.NS/.BO)")
    exch, base = pair
    try:
        try:
            q = _get_client().get_quote(trading_symbol=base, exchange=exch, segment="CASH")
        except Exception:
            q = _get_client(force_new=True).get_quote(trading_symbol=base, exchange=exch, segment="CASH")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Groww depth unavailable: {e}")

    d = q.get("depth") or {}
    bids = [{"price": b.get("price"), "qty": b.get("quantity"), "orders": b.get("orderCount")}
            for b in (d.get("buy") or [])]
    asks = [{"price": a.get("price"), "qty": a.get("quantity"), "orders": a.get("orderCount")}
            for a in (d.get("sell") or [])]
    best_bid = bids[0]["price"] if bids else None
    best_ask = asks[0]["price"] if asks else None
    spread = round(best_ask - best_bid, 2) if (best_bid is not None and best_ask is not None) else None
    out = {
        "symbol": sym,
        "ltp": q.get("last_price"),
        "bids": bids, "asks": asks,
        "best_bid": best_bid, "best_ask": best_ask, "spread": spread,
        "spread_pct": round(100 * spread / best_bid, 3) if (spread is not None and best_bid) else None,
        "total_buy_qty": q.get("total_buy_quantity"),
        "total_sell_qty": q.get("total_sell_quantity"),
        "market_open": market_open(),
    }
    _depth_cache[sym] = (time.time(), out)
    if len(_depth_cache) > 32:
        _depth_cache.pop(min(_depth_cache, key=lambda k: _depth_cache[k][0]))
    return out


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


# ---------- websocket: true realtime via GrowwFeed (NATS push) ----------
# One shared feed + consume thread for the whole process; each websocket
# client just reads the feed's snapshot for its symbols and pushes deltas.

_feed = None
_feed_lock = threading.Lock()
_feed_subscribed: set[str] = set()
_symbol_tokens: dict[str, tuple[str, str]] = {}  # yahoo symbol -> (exchange, exchange_token)


def _ensure_feed(symbols: list[str]) -> dict[str, tuple[str, str]]:
    """Subscribe any new symbols on the shared feed; return symbol->key map."""
    global _feed
    client = _get_client()
    with _feed_lock:
        if _feed is None:
            from growwapi import GrowwFeed
            _feed = GrowwFeed(client)
            threading.Thread(target=_feed.consume, daemon=True, name="groww-feed").start()
        fresh = []
        mapping: dict[str, tuple[str, str]] = {}
        for s in symbols:
            pair = _split(s)
            if not pair:
                continue
            if s not in _symbol_tokens:
                try:
                    inst = client.get_instrument_by_exchange_and_trading_symbol(
                        exchange=pair[0], trading_symbol=pair[1])
                except Exception:
                    continue
                if not inst:
                    continue
                _symbol_tokens[s] = (pair[0], str(inst["exchange_token"]))
            mapping[s] = _symbol_tokens[s]
            exch, etok = _symbol_tokens[s]
            key = f"{exch}:{etok}"
            if key not in _feed_subscribed and len(_feed_subscribed) < 100:
                fresh.append({"exchange": exch, "segment": "CASH", "exchange_token": etok})
                _feed_subscribed.add(key)
        if fresh:
            _feed.subscribe_ltp(fresh, on_data_received=lambda *_a, **_k: None)
        return mapping


@router.websocket("/ws")
async def live_ws(websocket: WebSocket):
    params = websocket.query_params
    if os.getenv("ULTRABULL_AUTH", "0") == "1":
        if verify_token(params.get("token") or "") is None:
            await websocket.close(code=4401)
            return
    symbols = [s.strip().upper() for s in (params.get("symbols") or "").split(",") if s.strip()][:40]
    try:
        mapping = await asyncio.to_thread(_ensure_feed, symbols)
    except Exception:
        mapping = {}
    if not mapping:
        await websocket.close(code=4400)
        return
    await websocket.accept()
    last_sent: dict[str, float] = {}
    quiet = 0
    try:
        while True:
            snap = _feed.get_ltp() or {}
            out = {}
            for sym, (exch, etok) in mapping.items():
                d = ((snap.get(exch) or {}).get("CASH") or {}).get(etok)
                if not d or not d.get("ltp"):
                    continue
                ltp = float(d["ltp"])
                if last_sent.get(sym) != ltp:
                    last_sent[sym] = ltp
                    ts_ms = d.get("tsInMillis") or 0
                    out[sym] = {"ltp": ltp, "ts": int(ts_ms / 1000) if ts_ms else int(time.time())}
            if out:
                quiet = 0
                await websocket.send_json({"type": "tick", "quotes": out})
            else:
                quiet += 1
                if quiet >= 25:  # ~15s heartbeat keeps closed-market connections honest
                    quiet = 0
                    await websocket.send_json({"type": "hb", "market_open": market_open()})
            await asyncio.sleep(0.6)
    except Exception:
        return  # client went away (or feed hiccup) — polling fallback covers it


# ---------- Groww holdings (merged into the holdings page) ----------

@groww_router.get("/holdings")
def groww_holdings():
    """The user's Groww demat holdings, normalized to the same shape as the
    SRE rows so the frontend can merge them into one table."""
    def _fetch(client):
        return client.get_holdings_for_user(timeout=15)

    try:
        try:
            resp = _fetch(_get_client())
        except HTTPException:
            raise
        except Exception:
            resp = _fetch(_get_client(force_new=True))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Groww holdings unavailable: {e}")

    rows = []
    for h in resp.get("holdings", []):
        qty = float(h.get("quantity") or 0)
        if qty <= 0:
            continue
        sym = (h.get("trading_symbol") or "").strip().upper() or None
        avg = float(h.get("average_price") or 0) or None
        invested = round(qty * avg, 2) if avg else None
        rows.append({
            "broker": "Groww",
            "symbol": sym,
            "series": None,
            "yahoo_symbol": _to_yahoo(sym, "NSE"),
            "name": None,
            "exchange": "NSE",
            "isin": h.get("isin"),
            "token_id": None,
            "quantity": qty,
            "avg_price": avg,
            "ltp": None,
            "prev_close": None,
            "pledged_qty": float(h.get("pledge_quantity") or 0) or None,
            "invested": invested,
            "current_value": None,
            "pnl": None,
            "pnl_pct": None,
            "day_pnl": None,
            "day_change_pct": None,
            "raw": h,
        })
    return {"holdings": rows, "count": len(rows)}
