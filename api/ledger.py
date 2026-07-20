"""Trade ledger — accumulate executed orders day by day so we can compute
TRUE multi-lot XIRR (exact cashflows), not the single-buy-date approximation.

Groww's order API is current-trading-day scoped: it can't hand us the history
of shares already held, but it does expose the day's executed orders. So a
daily poll (systemd timer on the server, after close) records each fill; over
time the ledger becomes a complete transaction history for anything traded
from the day polling started.

    python -m api.ledger poll     # fetch today's executed Groww orders
    python -m api.ledger show     # dump the ledger

Dedup is by (broker, order_id) so re-polling the same day is idempotent.
"""
from __future__ import annotations

import os
import re
import sqlite3
import sys
from datetime import date as _date_cls

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER_DB = os.path.join(ROOT, "data_cache", "trade_ledger.db")


def _db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(LEDGER_DB), exist_ok=True)
    conn = sqlite3.connect(LEDGER_DB, timeout=10.0)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS trades ("
        "  broker TEXT NOT NULL,"
        "  order_id TEXT NOT NULL,"
        "  yahoo_symbol TEXT,"
        "  symbol TEXT,"
        "  side TEXT NOT NULL,"          # BUY / SELL
        "  quantity REAL NOT NULL,"
        "  price REAL NOT NULL,"
        "  trade_date TEXT NOT NULL,"    # YYYY-MM-DD
        "  recorded_at TEXT NOT NULL,"
        "  PRIMARY KEY (broker, order_id))"
    )
    return conn


def _norm(k) -> str:
    return re.sub(r"[^a-z0-9]", "", str(k).lower())


def _f(x):
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _pick(row: dict, *names):
    n = {_norm(k): v for k, v in row.items()}
    for name in names:
        v = n.get(name)
        if v not in (None, "", 0):
            return v
    return None


def _extract_trade(o: dict) -> dict | None:
    """Map one Groww order record to a ledger row. Defensive against the
    undocumented field spellings — only records fully-executed orders."""
    status = str(_pick(o, "orderstatus", "status") or "").upper()
    if status and not any(s in status for s in ("EXECUTED", "COMPLETE", "FILLED", "TRADED")):
        return None
    qty = _f(_pick(o, "filledquantity", "executedquantity", "quantity", "qty"))
    price = _f(_pick(o, "averagefillprice", "avgfillprice", "averageprice", "avgprice",
                     "tradeprice", "price"))
    if not qty or not price or qty <= 0 or price <= 0:
        return None
    side = str(_pick(o, "transactiontype", "side", "buysell") or "").upper()
    side = "BUY" if side.startswith("B") else "SELL" if side.startswith("S") else side or "BUY"
    order_id = _pick(o, "growworderid", "orderid", "orderreferenceid", "id")
    if not order_id:
        return None
    sym = _pick(o, "tradingsymbol", "symbol")
    dt_raw = _pick(o, "exchangetime", "tradedate", "createdat", "orderdatetime", "creationtime")
    trade_date = None
    if dt_raw:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", str(dt_raw))
        if m:
            trade_date = m.group(1)
    trade_date = trade_date or _date_cls.today().isoformat()
    yh = None
    if sym:
        base = re.sub(r"-(EQ|BE)$", "", str(sym).strip().upper())
        yh = base + ".NS" if base else None
    return {"order_id": str(order_id), "yahoo_symbol": yh, "symbol": str(sym).upper() if sym else None,
            "side": side, "quantity": qty, "price": price, "trade_date": trade_date}


def record_trades(broker: str, orders: list[dict]) -> int:
    conn = _db()
    now = _date_cls.today().isoformat()
    added = 0
    try:
        with conn:
            for o in orders:
                t = _extract_trade(o)
                if not t:
                    continue
                cur = conn.execute(
                    "INSERT OR IGNORE INTO trades (broker, order_id, yahoo_symbol, symbol, side, "
                    "quantity, price, trade_date, recorded_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    (broker, t["order_id"], t["yahoo_symbol"], t["symbol"], t["side"],
                     t["quantity"], t["price"], t["trade_date"], now),
                )
                added += cur.rowcount
    finally:
        conn.close()
    return added


def poll_groww() -> dict:
    """Fetch the day's executed Groww orders and append to the ledger."""
    from .live import _get_client
    client = _get_client()
    orders = []
    try:
        seg = getattr(client, "SEGMENT_CASH", "CASH")
        page = 0
        while page < 20:
            resp = client.get_order_list(page=page, page_size=50, segment=seg)
            batch = resp.get("order_list") or resp.get("orders") or []
            if not batch:
                break
            orders.extend(batch)
            if len(batch) < 50:
                break
            page += 1
    except Exception as e:
        return {"ok": False, "error": str(e), "polled": 0, "added": 0}
    added = record_trades("Groww", orders)
    return {"ok": True, "polled": len(orders), "added": added}


def ledger_flows(yahoo_symbols: set[str] | None = None) -> dict[str, list[tuple[str, float, float]]]:
    """Per-symbol list of (trade_date, signed_qty, price). BUY qty positive,
    SELL negative — the raw material for cashflow XIRR."""
    conn = _db()
    try:
        rows = conn.execute(
            "SELECT yahoo_symbol, side, quantity, price, trade_date FROM trades "
            "WHERE yahoo_symbol IS NOT NULL ORDER BY trade_date"
        ).fetchall()
    finally:
        conn.close()
    out: dict[str, list] = {}
    for yh, side, qty, price, dt in rows:
        if yahoo_symbols is not None and yh not in yahoo_symbols:
            continue
        signed = qty if side == "BUY" else -qty
        out.setdefault(yh, []).append((dt, signed, price))
    return out


def has_ledger() -> bool:
    conn = _db()
    try:
        return conn.execute("SELECT 1 FROM trades LIMIT 1").fetchone() is not None
    finally:
        conn.close()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "poll"
    if cmd == "poll":
        print(poll_groww())
    elif cmd == "show":
        conn = _db()
        for r in conn.execute("SELECT trade_date, broker, symbol, side, quantity, price FROM trades ORDER BY trade_date"):
            print(r)
        conn.close()
    else:
        print("usage: python -m api.ledger [poll|show]")
