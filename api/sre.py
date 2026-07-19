"""SRE broker proxy (SkyRaise "MoneyMaker" web terminal).

SRE has no public API, so we replay the exact requests its own web frontend
makes: POST with an empty JSON body, a short-lived bearer token, and
browser-identical headers (the terminal's backend checks Origin/Referer).

The token is pasted by the user into our web UI and kept in sessionStorage
there; every call forwards it in the POST body — the server stores nothing,
and the token never appears in a URL or server log. Tokens expire ~30 min
after issue (JWT `exp`), so 401s are surfaced distinctly for the UI to
prompt a re-paste.

Read-only by design: holdings, scrip details, company news. No order
endpoints — the app analyzes, the user executes.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time

import sqlite3

import requests
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

SRE_BASE = "https://trading.sre.co.in:29002/MoneyMakerApi/v1"
SRE_ORIGIN = "https://trading.sre.co.in:28001"
SRE_TIMEOUT = 20.0
# Broker terminals sometimes run on certs a stock CA bundle rejects;
# SRE_VERIFY_SSL=0 is the documented escape hatch.
SRE_VERIFY_SSL = os.getenv("SRE_VERIFY_SSL", "1") != "0"

router = APIRouter(prefix="/api/sre", tags=["sre"])


def _headers(token: str) -> dict:
    return {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-IN,en-GB;q=0.9,en;q=0.8",
        "Authorization": f"Bearer {token}",
        "Cache-Control": "no-cache",
        "Content-Type": "application/json",
        "Expires": "0",
        "Pragma": "no-cache",
        "Origin": SRE_ORIGIN,
        "Referer": SRE_ORIGIN + "/",
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
            "(KHTML, like Gecko) Version/26.5 Safari/605.1.15"
        ),
    }


def _clean_token(token: str) -> str:
    # tolerate a pasted "Bearer eyJ..." or surrounding quotes/whitespace
    return re.sub(r"^\s*Bearer\s+", "", token.strip().strip('"').strip(), flags=re.I)


def _call(path: str, params: dict, token: str):
    token = _clean_token(token)
    if not token:
        raise HTTPException(status_code=401, detail="No SRE token — paste one from the trading terminal.")
    try:
        resp = requests.post(
            f"{SRE_BASE}/{path}", params=params, data="{}",
            headers=_headers(token), timeout=SRE_TIMEOUT, verify=SRE_VERIFY_SSL,
        )
    except requests.exceptions.SSLError:
        raise HTTPException(
            status_code=502,
            detail="TLS verification against trading.sre.co.in failed — if its cert is "
                   "non-standard, restart the API with SRE_VERIFY_SSL=0.",
        )
    except requests.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach SRE ({type(e).__name__}) — check network/VPN.",
        )
    if resp.status_code in (401, 403):
        raise HTTPException(
            status_code=401,
            detail="SRE rejected the token (expired or invalid) — tokens last ~30 min; "
                   "paste a fresh one from the terminal.",
        )
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"SRE returned HTTP {resp.status_code}: {resp.text[:300]}")
    try:
        return resp.json()
    except ValueError:
        raise HTTPException(status_code=502, detail=f"SRE returned non-JSON: {resp.text[:300]}")


def token_info(token: str) -> dict | None:
    """Decode the JWT payload (no signature check — purely informational) so the
    UI can show whose token it is and when it dies."""
    try:
        payload_b64 = _clean_token(token).split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        exp = int(payload.get("exp", 0))
        return {
            "client": payload.get("name"),
            "expires_at": time.strftime("%H:%M:%S", time.localtime(exp)) if exp else None,
            "seconds_left": max(0, exp - int(time.time())) if exp else None,
            "expired": bool(exp and exp <= time.time()),
        }
    except Exception:
        return None


# ---------- response normalization (SRE's field names are undocumented) ----------

def _norm_key(k) -> str:
    return re.sub(r"[^a-z0-9]", "", str(k).lower())


def _f(x) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).replace(",", "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _extract_rows(payload) -> list[dict]:
    """Find the list-of-records inside whatever envelope SRE wraps it in."""
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if not isinstance(payload, dict):
        return []
    # well-known wrapper keys first, then breadth-first over nested dicts
    by_key = {_norm_key(k): v for k, v in payload.items()}
    for k in ("data", "result", "holdings", "holdingsdata", "d", "response", "rows", "list"):
        v = by_key.get(k)
        if isinstance(v, list):
            return [r for r in v if isinstance(r, dict)]
        if isinstance(v, dict):
            found = _extract_rows(v)
            if found:
                return found
    queue = [payload]
    for _ in range(3):
        nxt = []
        for d in queue:
            for v in d.values():
                if isinstance(v, list) and v and all(isinstance(r, dict) for r in v):
                    return v
                if isinstance(v, dict):
                    nxt.append(v)
        queue = nxt
        if not queue:
            break
    return []


def _pick(row_norm: dict, *names):
    for n in names:
        v = row_norm.get(n)
        if v is not None and v != "":
            return v
    return None


# NSE series suffixes some terminals glue onto the symbol ("BEPL-EQ")
_SERIES_RE = re.compile(r"-(EQ|BE|BZ|SM|ST|IT|GS)$", re.I)


def yahoo_symbol(symbol: str | None, exchange: str | None) -> str | None:
    if not symbol:
        return None
    base = _SERIES_RE.sub("", str(symbol).strip().upper())
    if not base:
        return None
    suffix = ".BO" if exchange and "BSE" in str(exchange).upper() else ".NS"
    return base + suffix


def normalize_holding(row: dict) -> dict:
    """Map one SRE holdings record to our shape.

    Observed /Holdings schema (2026-07): isin, position (total qty), value
    (current ₹), free_qty/collateral_qty/pledged_qty, nse_name "ATUL EQ" /
    bse_name "ATUL A", nse_token/bse_token, nse_ltp/bse_ltp, buy_price (avg,
    as a STRING), profit (unrealized ₹), todays_profit, close (0 = absent).
    Generic candidate names are kept as fallbacks in case other segments or
    app versions differ."""
    n = {_norm_key(k): v for k, v in row.items()}

    nse_name = _pick(n, "nsename")
    bse_name = _pick(n, "bsename")
    symbol = series = exchange = None
    if nse_name or bse_name:
        # "ATUL EQ" -> symbol ATUL, series EQ; prefer the NSE listing
        parts = str(nse_name or bse_name).strip().split()
        symbol = parts[0].upper() if parts else None
        series = parts[1].upper() if len(parts) > 1 else None
        exchange = "NSE" if nse_name else "BSE"
    else:
        symbol = _pick(n, "tradingsymbol", "symbol", "scripname", "nsesymbol",
                       "scripsymbol", "symbolname", "scrip", "sym")
        exchange = _pick(n, "exchange", "exch", "exchangesegment", "segment", "market")

    qty = _f(_pick(n, "position", "netqty", "netquantity", "quantity", "qty",
                   "holdingqty", "holdingquantity", "totalqty", "freeqty", "dpqty"))
    avg = _f(_pick(n, "buyprice", "avgprice", "buyavgprice", "averageprice", "avgrate",
                   "buyavg", "avgcostprice", "averagerate", "costprice"))
    if exchange == "BSE":
        ltp = _f(_pick(n, "bseltp", "nseltp", "ltp", "lasttradedprice", "lastprice",
                       "currentprice", "cmp", "marketprice"))
    else:
        ltp = _f(_pick(n, "nseltp", "bseltp", "ltp", "lasttradedprice", "lastprice",
                       "currentprice", "cmp", "marketprice"))
    # SRE sends close=0 when it has no previous close — 0 is never a real price
    prev_close = _f(_pick(n, "prevclose", "previousclose", "previouscloseprice",
                          "closeprice", "close")) or None

    nse_token = _pick(n, "nsetoken")
    bse_token = _pick(n, "bsetoken")
    if nse_token:
        token_id = f"NseCm{nse_token}"  # ScripDetails wants e.g. TokenID=NseCm419
    elif bse_token:
        token_id = f"BseCm{bse_token}"
    else:
        token_id = _pick(n, "tokenid", "scriptoken", "securitytoken",
                         "instrumenttoken", "scripcode", "tokenno", "token")

    out = {
        "symbol": symbol,
        "series": series,
        "yahoo_symbol": yahoo_symbol(symbol, exchange),
        "name": _pick(n, "companyname", "scripfullname", "securityname", "company", "name"),
        "exchange": exchange,
        "isin": _pick(n, "isin", "isinno", "isincode"),
        "token_id": token_id,
        "quantity": qty,
        "avg_price": avg,
        "ltp": ltp,
        "prev_close": prev_close,
        "pledged_qty": _f(_pick(n, "pledgedqty", "pledgeqty", "pledgedquantity")),
        "invested": None,
        "current_value": _f(_pick(n, "value", "currentvalue", "marketvalue", "holdingvalue")),
        "pnl": _f(_pick(n, "profit", "unrealizedpnl", "unrealisedpnl", "pnl")),
        "pnl_pct": None,
        "day_pnl": _f(_pick(n, "todaysprofit", "dayprofit", "todayspnl")),
        "day_change_pct": None,
        "raw": row,
    }
    if qty is not None and avg is not None:
        out["invested"] = round(qty * avg, 2)
    if out["current_value"] is None and qty is not None and ltp is not None:
        out["current_value"] = round(qty * ltp, 2)
    if out["pnl"] is None and out["invested"] is not None and out["current_value"] is not None:
        out["pnl"] = round(out["current_value"] - out["invested"], 2)
    if out["pnl"] is not None:
        out["pnl"] = round(out["pnl"], 2)
        if out["invested"]:
            out["pnl_pct"] = round(100.0 * out["pnl"] / out["invested"], 2)
    if ltp is not None and prev_close:
        out["day_change_pct"] = round(100.0 * (ltp - prev_close) / prev_close, 2)
    return out


# ---------- endpoints (token travels in the POST body, never the URL) ----------

class SreAuth(BaseModel):
    client_id: str
    token: str


class SreScripReq(SreAuth):
    token_id: str


class SreNewsReq(SreAuth):
    symbol: str


# ---------- per-user SRE credential store (so the UI comes preset) ----------
# The SRE token expires ~30 min after issue, so what this really preserves is
# the client ID plus the last token — fresh enough tokens auto-connect, stale
# ones just leave the fields prefilled for a quick re-paste.

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CREDS_DB = os.path.join(_ROOT, "data_cache", "sre_creds.db")


def _creds_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(CREDS_DB), exist_ok=True)
    conn = sqlite3.connect(CREDS_DB, timeout=10.0)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS sre_creds ("
        "  username TEXT PRIMARY KEY,"
        "  client_id TEXT NOT NULL,"
        "  token TEXT NOT NULL,"
        "  updated_at TEXT NOT NULL)"
    )
    return conn


def _cred_user(request: Request) -> str:
    # auth middleware sets state.user in production; local dev runs open
    return getattr(request.state, "user", None) or "local"


class SreCreds(BaseModel):
    client_id: str
    token: str


@router.get("/creds")
def get_creds(request: Request):
    conn = _creds_db()
    try:
        row = conn.execute(
            "SELECT client_id, token FROM sre_creds WHERE username = ?",
            (_cred_user(request),),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return {"client_id": None, "token": None, "token_info": None}
    return {"client_id": row[0], "token": row[1], "token_info": token_info(row[1])}


@router.put("/creds")
def put_creds(creds: SreCreds, request: Request):
    conn = _creds_db()
    try:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO sre_creds (username, client_id, token, updated_at) "
                "VALUES (?, ?, ?, datetime('now'))",
                (_cred_user(request), creds.client_id.strip(), _clean_token(creds.token)),
            )
    finally:
        conn.close()
    return {"ok": True}


@router.post("/holdings")
def holdings(auth: SreAuth):
    raw = _call("Holdings", {"ClientID": auth.client_id.strip()}, auth.token)
    rows = [normalize_holding(r) for r in _extract_rows(raw)]
    return {
        "holdings": rows,
        "count": len(rows),
        "token": token_info(auth.token),
        "raw": raw,
    }


@router.post("/scrip")
def scrip_details(req: SreScripReq):
    """Live quote/details for one instrument (TokenID like 'NseCm419')."""
    return {"raw": _call("ScripDetails", {"TokenID": req.token_id.strip()}, req.token)}


@router.post("/news")
def company_news(req: SreNewsReq):
    raw = _call(
        "AccordCompanyNews",
        {"ClientID": req.client_id.strip(), "SymbolOrCode": req.symbol.strip().upper()},
        req.token,
    )
    return {"items": _extract_rows(raw), "raw": raw}
