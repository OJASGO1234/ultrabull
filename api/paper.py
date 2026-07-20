"""Paper-trading account — forward, out-of-sample validation with real live
prices and no real money. Equity / ETF / REIT delivery only (no F&O).

Cash is the source of truth. Buy/sell fill at the live price (or a supplied
one); user-settable brokerage% and tax% are charged on each transaction's gross
value and baked into the cost basis / netted from proceeds so realized P&L is
exact. Positions mark-to-market against the live Groww feed. Daily equity
snapshots build the performance curve; a kill-line flags drawdown breaches.

Everything persists to data_cache/paper.db, scoped per portal user.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import date as _date_cls, datetime

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/paper", tags=["paper"])

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_DB = os.path.join(ROOT, "data_cache", "paper.db")

DEFAULT_CAPITAL = 1_000_000.0
DEFAULT_BROKERAGE_PCT = 0.0   # many Indian brokers: zero on delivery
DEFAULT_TAX_PCT = 0.12        # rough all-in: STT + exchange + GST + stamp, both sides
DEFAULT_KILL_DRAWDOWN_PCT = 15.0


def _db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(PAPER_DB), exist_ok=True)
    conn = sqlite3.connect(PAPER_DB, timeout=10.0)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS paper_account (
          username TEXT PRIMARY KEY,
          starting_capital REAL NOT NULL,
          cash REAL NOT NULL,
          realized_pnl REAL NOT NULL DEFAULT 0,
          brokerage_pct REAL NOT NULL,
          tax_pct REAL NOT NULL,
          kill_drawdown_pct REAL NOT NULL,
          created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS paper_positions (
          username TEXT NOT NULL,
          symbol TEXT NOT NULL,
          yahoo_symbol TEXT,
          qty REAL NOT NULL,
          cost_basis REAL NOT NULL,
          opened_at TEXT NOT NULL,
          stop REAL, target REAL,
          PRIMARY KEY (username, symbol));
        CREATE TABLE IF NOT EXISTS paper_trades (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          username TEXT NOT NULL,
          ts TEXT NOT NULL,
          symbol TEXT NOT NULL,
          yahoo_symbol TEXT,
          side TEXT NOT NULL,
          qty REAL NOT NULL,
          price REAL NOT NULL,
          gross REAL NOT NULL,
          charges REAL NOT NULL,
          realized_pnl REAL,
          note TEXT);
        CREATE TABLE IF NOT EXISTS paper_equity (
          username TEXT NOT NULL,
          d TEXT NOT NULL,
          equity REAL NOT NULL,
          cash REAL NOT NULL,
          holdings REAL NOT NULL,
          PRIMARY KEY (username, d));
        """
    )
    return conn


def _user(request: Request) -> str:
    return getattr(request.state, "user", None) or "local"


def _account(conn: sqlite3.Connection, user: str) -> dict:
    row = conn.execute(
        "SELECT starting_capital, cash, realized_pnl, brokerage_pct, tax_pct, kill_drawdown_pct, created_at "
        "FROM paper_account WHERE username = ?", (user,)).fetchone()
    if row is None:
        with conn:
            conn.execute(
                "INSERT INTO paper_account (username, starting_capital, cash, realized_pnl, "
                "brokerage_pct, tax_pct, kill_drawdown_pct, created_at) VALUES (?,?,?,0,?,?,?,?)",
                (user, DEFAULT_CAPITAL, DEFAULT_CAPITAL, DEFAULT_BROKERAGE_PCT, DEFAULT_TAX_PCT,
                 DEFAULT_KILL_DRAWDOWN_PCT, datetime.now().isoformat(timespec="seconds")))
        return _account(conn, user)
    keys = ["starting_capital", "cash", "realized_pnl", "brokerage_pct", "tax_pct",
            "kill_drawdown_pct", "created_at"]
    return dict(zip(keys, row))


def _charges(gross: float, acct: dict) -> float:
    return round(gross * (acct["brokerage_pct"] + acct["tax_pct"]) / 100.0, 2)


def _snapshot(conn: sqlite3.Connection, user: str, equity: float, cash: float, holdings: float) -> None:
    with conn:
        conn.execute("INSERT OR REPLACE INTO paper_equity (username, d, equity, cash, holdings) VALUES (?,?,?,?,?)",
                     (user, _date_cls.today().isoformat(), round(equity, 2), round(cash, 2), round(holdings, 2)))


# ---------- models ----------

class ConfigReq(BaseModel):
    starting_capital: float | None = None
    brokerage_pct: float | None = None
    tax_pct: float | None = None
    kill_drawdown_pct: float | None = None


class TradeReq(BaseModel):
    symbol: str
    yahoo_symbol: str | None = None
    side: str  # BUY / SELL
    qty: float
    price: float | None = None  # None -> fill at live price
    stop: float | None = None
    target: float | None = None
    note: str | None = None


# ---------- state assembly ----------

def _state(conn: sqlite3.Connection, user: str) -> dict:
    from .live import ltp_map, market_open

    acct = _account(conn, user)
    positions = conn.execute(
        "SELECT symbol, yahoo_symbol, qty, cost_basis, opened_at, stop, target "
        "FROM paper_positions WHERE username = ? ORDER BY symbol", (user,)).fetchall()
    ysyms = [p[1] for p in positions if p[1]]
    live = ltp_map(ysyms) if ysyms else {}

    pos_out, holdings = [], 0.0
    for sym, ysym, qty, cost_basis, opened, stop, target in positions:
        avg = cost_basis / qty if qty else 0.0
        ltp = live.get(ysym)
        mkt_val = qty * ltp if ltp is not None else cost_basis
        holdings += mkt_val
        pos_out.append({
            "symbol": sym, "yahoo_symbol": ysym, "qty": qty,
            "avg_cost": round(avg, 2), "cost_basis": round(cost_basis, 2),
            "ltp": ltp, "market_value": round(mkt_val, 2),
            "unrealized_pnl": round(mkt_val - cost_basis, 2) if ltp is not None else None,
            "unrealized_pct": round(100 * (mkt_val - cost_basis) / cost_basis, 2) if (ltp is not None and cost_basis) else None,
            "opened_at": opened, "stop": stop, "target": target,
            "stop_hit": ltp is not None and stop is not None and ltp <= stop,
            "target_hit": ltp is not None and target is not None and ltp >= target,
        })

    equity = acct["cash"] + holdings
    invested = acct["starting_capital"]
    total_pnl = equity - invested
    # performance
    trades = conn.execute(
        "SELECT side, realized_pnl FROM paper_trades WHERE username = ? AND side='SELL'", (user,)).fetchall()
    closed = [t[1] for t in trades if t[1] is not None]
    wins = [p for p in closed if p > 0]
    eq_rows = conn.execute("SELECT equity FROM paper_equity WHERE username = ? ORDER BY d", (user,)).fetchall()
    curve = [r[0] for r in eq_rows] + [equity]
    peak, max_dd = curve[0] if curve else invested, 0.0
    for e in curve:
        peak = max(peak, e)
        if peak > 0:
            max_dd = max(max_dd, (peak - e) / peak * 100)
    kill = acct["kill_drawdown_pct"]
    cur_peak = max(curve) if curve else invested
    cur_drawdown = round((cur_peak - equity) / cur_peak * 100, 2) if cur_peak > 0 else 0.0

    return {
        "config": {"starting_capital": acct["starting_capital"], "brokerage_pct": acct["brokerage_pct"],
                   "tax_pct": acct["tax_pct"], "kill_drawdown_pct": kill, "created_at": acct["created_at"]},
        "cash": round(acct["cash"], 2),
        "holdings_value": round(holdings, 2),
        "equity": round(equity, 2),
        "starting_capital": invested,
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(100 * total_pnl / invested, 2) if invested else None,
        "realized_pnl": round(acct["realized_pnl"], 2),
        "unrealized_pnl": round(holdings - sum(p["cost_basis"] for p in pos_out), 2),
        "positions": pos_out,
        "market_open": market_open(),
        "stats": {
            "closed_trades": len(closed),
            "win_rate": round(len(wins) / len(closed), 3) if closed else None,
            "avg_win": round(sum(wins) / len(wins), 2) if wins else None,
            "avg_loss": round(sum(p for p in closed if p <= 0) / max(1, len(closed) - len(wins)), 2) if len(closed) > len(wins) else None,
            "max_drawdown_pct": round(max_dd, 2),
            "current_drawdown_pct": cur_drawdown,
            "kill_breached": cur_drawdown >= kill,
        },
    }


# ---------- endpoints ----------

@router.get("")
def paper_state(request: Request):
    conn = _db()
    try:
        st = _state(conn, _user(request))
        _snapshot(conn, _user(request), st["equity"], st["cash"], st["holdings_value"])
        return st
    finally:
        conn.close()


@router.post("/config")
def paper_config(req: ConfigReq, request: Request):
    user = _user(request)
    conn = _db()
    try:
        acct = _account(conn, user)
        # starting_capital only settable while the account is untouched (no trades)
        has_trades = conn.execute("SELECT 1 FROM paper_trades WHERE username=? LIMIT 1", (user,)).fetchone()
        with conn:
            if req.starting_capital is not None and not has_trades:
                conn.execute("UPDATE paper_account SET starting_capital=?, cash=? WHERE username=?",
                             (req.starting_capital, req.starting_capital, user))
            if req.brokerage_pct is not None:
                conn.execute("UPDATE paper_account SET brokerage_pct=? WHERE username=?", (max(0.0, req.brokerage_pct), user))
            if req.tax_pct is not None:
                conn.execute("UPDATE paper_account SET tax_pct=? WHERE username=?", (max(0.0, req.tax_pct), user))
            if req.kill_drawdown_pct is not None:
                conn.execute("UPDATE paper_account SET kill_drawdown_pct=? WHERE username=?", (max(0.0, req.kill_drawdown_pct), user))
        return {"ok": True, "capital_locked": bool(has_trades) and req.starting_capital is not None}
    finally:
        conn.close()


@router.post("/trade")
def paper_trade(req: TradeReq, request: Request):
    user = _user(request)
    side = req.side.strip().upper()
    if side not in ("BUY", "SELL"):
        raise HTTPException(status_code=422, detail="side must be BUY or SELL")
    if req.qty <= 0:
        raise HTTPException(status_code=422, detail="qty must be positive")
    sym = req.symbol.strip().upper()
    ysym = (req.yahoo_symbol or "").strip().upper() or None

    conn = _db()
    try:
        acct = _account(conn, user)
        price = req.price
        if price is None:
            from .live import ltp_map
            if not ysym:
                raise HTTPException(status_code=422, detail="need a price or a yahoo_symbol for a live fill")
            price = ltp_map([ysym]).get(ysym)
            if price is None:
                raise HTTPException(status_code=502, detail="no live price — supply a price or retry in market hours")
        price = float(price)
        gross = req.qty * price
        charges = _charges(gross, acct)

        pos = conn.execute("SELECT qty, cost_basis FROM paper_positions WHERE username=? AND symbol=?",
                           (user, sym)).fetchone()
        realized = None
        with conn:
            if side == "BUY":
                if acct["cash"] < gross + charges:
                    raise HTTPException(status_code=422, detail=f"insufficient paper cash (need ₹{gross + charges:,.0f}, have ₹{acct['cash']:,.0f})")
                new_qty = (pos[0] if pos else 0) + req.qty
                new_basis = (pos[1] if pos else 0) + gross + charges
                conn.execute(
                    "INSERT INTO paper_positions (username,symbol,yahoo_symbol,qty,cost_basis,opened_at,stop,target) "
                    "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(username,symbol) DO UPDATE SET "
                    "qty=?, cost_basis=?, yahoo_symbol=COALESCE(?,yahoo_symbol), stop=COALESCE(?,stop), target=COALESCE(?,target)",
                    (user, sym, ysym, new_qty, new_basis, datetime.now().isoformat(timespec="seconds"),
                     req.stop, req.target, new_qty, new_basis, ysym, req.stop, req.target))
                conn.execute("UPDATE paper_account SET cash=cash-? WHERE username=?", (gross + charges, user))
            else:  # SELL
                if not pos or pos[0] < req.qty:
                    raise HTTPException(status_code=422, detail=f"can't sell {req.qty}; hold {pos[0] if pos else 0}")
                avg = pos[1] / pos[0]
                proceeds = gross - charges
                realized = round(proceeds - avg * req.qty, 2)
                rem_qty = pos[0] - req.qty
                rem_basis = pos[1] - avg * req.qty
                if rem_qty <= 1e-9:
                    conn.execute("DELETE FROM paper_positions WHERE username=? AND symbol=?", (user, sym))
                else:
                    conn.execute("UPDATE paper_positions SET qty=?, cost_basis=? WHERE username=? AND symbol=?",
                                 (rem_qty, rem_basis, user, sym))
                conn.execute("UPDATE paper_account SET cash=cash+?, realized_pnl=realized_pnl+? WHERE username=?",
                             (proceeds, realized, user))
            conn.execute(
                "INSERT INTO paper_trades (username,ts,symbol,yahoo_symbol,side,qty,price,gross,charges,realized_pnl,note) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (user, datetime.now().isoformat(timespec="seconds"), sym, ysym, side, req.qty, price,
                 round(gross, 2), charges, realized, req.note))
        st = _state(conn, user)
        _snapshot(conn, user, st["equity"], st["cash"], st["holdings_value"])
        return {"ok": True, "price": price, "charges": charges, "realized_pnl": realized, "state": st}
    finally:
        conn.close()


@router.get("/history")
def paper_history(request: Request):
    user = _user(request)
    conn = _db()
    try:
        eq = conn.execute("SELECT d, equity FROM paper_equity WHERE username=? ORDER BY d", (user,)).fetchall()
        tr = conn.execute(
            "SELECT ts, symbol, side, qty, price, charges, realized_pnl, note FROM paper_trades "
            "WHERE username=? ORDER BY id DESC LIMIT 200", (user,)).fetchall()
        return {
            "equity_curve": [{"date": d, "equity": e} for d, e in eq],
            "trades": [{"ts": t[0], "symbol": t[1], "side": t[2], "qty": t[3], "price": t[4],
                        "charges": t[5], "realized_pnl": t[6], "note": t[7]} for t in tr],
        }
    finally:
        conn.close()


def snapshot_all() -> dict:
    """Record today's EOD equity for every paper account (daily timer calls
    this so the performance curve builds even on days the tab isn't opened)."""
    conn = _db()
    try:
        users = [r[0] for r in conn.execute("SELECT username FROM paper_account").fetchall()]
        for u in users:
            st = _state(conn, u)
            _snapshot(conn, u, st["equity"], st["cash"], st["holdings_value"])
        return {"ok": True, "accounts": len(users)}
    finally:
        conn.close()


@router.post("/reset")
def paper_reset(request: Request):
    user = _user(request)
    conn = _db()
    try:
        with conn:
            for tbl in ("paper_positions", "paper_trades", "paper_equity", "paper_account"):
                conn.execute(f"DELETE FROM {tbl} WHERE username=?", (user,))
        _account(conn, user)  # recreate at defaults
        return {"ok": True}
    finally:
        conn.close()


if __name__ == "__main__":
    print(snapshot_all())
