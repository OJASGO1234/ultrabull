"""Point-in-time paper-trading bridge for the momentum strategy.

The gap this closes: api/paper.py is a fully manual, live-price paper ledger
with no connection to any strategy's signals, and api/main.py's
`signal_snapshots` table (see `/api/signal/{symbol}`) keeps only the LATEST
digest per symbol (`INSERT OR REPLACE`, no history) -- neither can answer
"what did the momentum strategy recommend on a given day, and what actually
happened when that recommendation was acted on?".

This module adds the minimum piece needed for that loop:

  1. An IMMUTABLE snapshot (`momentum_recommendations`, in the SAME sqlite
     file api/paper.py already uses -- api.paper.PAPER_DB -- as a new,
     additive table; api/paper.py's own schema/tables are untouched):
     records what core/strategies/momentum_strategy.py's live screen
     (`screen_universe_now`, the same function GET /api/momentum/screen
     calls) said about one symbol on one date. `suggested_entry` is written
     ONCE at creation and never updated afterwards.

  2. An execute step that fills the recommendation as an ACTUAL paper trade
     through the EXISTING api/paper.py ledger (`_fill`, imported and called
     unmodified) at whatever price is supplied by the caller -- the next
     session's real open, a live LTP, or a manually-typed fill. That price
     is recorded as `actual_entry`, a column separate from
     `suggested_entry`; the two are never merged or overwritten into each
     other, matching the "suggested vs actual" requirement directly.

  3. A read endpoint that joins each recommendation back to the
     paper_trades/paper_positions rows `_fill` created for it (same sqlite
     file, so this is a plain SQL join) -- so a recommendation's outcome
     (still open vs. closed, and its P&L) can be graded against what was
     originally suggested.

What this deliberately does NOT do (see core/MOMENTUM_STRATEGY.md for the
full picture): it does not run itself on a schedule. A human (or, if wired
up later, a cron job -- see api/main.py's existing `/api/audit/run`-style
pattern for the closest precedent in this repo) must call POST .../snapshot
at end of day and POST .../execute the next session. Automating that
schedule is the remaining piece for a fully hands-off loop, and is called
out explicitly rather than pretended to already exist.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from .paper import _db as _paper_db, _fill as _paper_fill, _user as _paper_user

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(ROOT, "core")
import sys  # noqa: E402
if CORE not in sys.path:
    sys.path.insert(0, CORE)

from strategies.momentum_strategy import MomentumConfig, screen_universe_now  # noqa: E402
from momentum import LOOKBACK_BARS as _MOMENTUM_LOOKBACK_BARS  # noqa: E402
from event_study import UNIVERSE as _MOMENTUM_UNIVERSE  # noqa: E402

router = APIRouter(prefix="/api/momentum/paper", tags=["momentum-paper"])

_MOMENTUM_CACHE_DIR = os.path.join(ROOT, "data_cache", "momentum")


def _db() -> sqlite3.Connection:
    conn = _paper_db()  # creates/opens the SAME file as api/paper.py; its tables are untouched here
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS momentum_recommendations (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          username TEXT NOT NULL,
          created_at TEXT NOT NULL,
          symbol TEXT NOT NULL,
          yahoo_symbol TEXT NOT NULL,
          strategy TEXT NOT NULL,
          lookback TEXT NOT NULL,
          as_of_date TEXT NOT NULL,
          suggested_entry REAL NOT NULL,
          sl_est REAL, tp_est REAL, sl_atr REAL, tp_atr REAL,
          absolute_momentum REAL, relative_momentum REAL, rank INTEGER,
          reasons TEXT,
          status TEXT NOT NULL DEFAULT 'pending',
          executed_at TEXT, actual_entry REAL, paper_trade_id INTEGER
        );
        """
    )
    # Additive migration for api/momentum_monitor.py's close-tracking (a
    # recommendation that reaches status='closed' records how/when/at what
    # price -- see that module for the monitoring cycle that writes these).
    # ALTER TABLE ADD COLUMN is a no-op-safe, non-destructive sqlite
    # migration; re-running it every call (like the CREATE TABLE IF NOT
    # EXISTS above) is intentional so a pre-existing db file picks up the
    # new columns without a separate migration step.
    existing_cols = {r[1] for r in conn.execute("PRAGMA table_info(momentum_recommendations)").fetchall()}
    with conn:
        for col, decl in (
            ("closed_at", "TEXT"), ("exit_price", "REAL"),
            ("exit_reason", "TEXT"), ("close_realized_pnl", "REAL"),
        ):
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE momentum_recommendations ADD COLUMN {col} {decl}")
    return conn


class SnapshotReq(BaseModel):
    symbol: str
    lookback: str = "6M"
    years: float = 2.0
    top_n: int = 10
    use_relative: bool = True
    use_technical: bool = True
    use_market_regime: bool = True
    symbols: str | None = None  # comma-separated universe override; default the shared UNIVERSE


class ExecuteReq(BaseModel):
    actual_entry: float          # the REAL fill price (e.g. next session's open) -- distinct from suggested_entry
    qty: float
    note: str | None = None


def _row_to_dict(r: sqlite3.Row | tuple, cols: list[str]) -> dict:
    return dict(zip(cols, r))


_REC_COLS = [
    "id", "username", "created_at", "symbol", "yahoo_symbol", "strategy", "lookback",
    "as_of_date", "suggested_entry", "sl_est", "tp_est", "sl_atr", "tp_atr",
    "absolute_momentum", "relative_momentum", "rank", "reasons", "status",
    "executed_at", "actual_entry", "paper_trade_id",
    "closed_at", "exit_price", "exit_reason", "close_realized_pnl",
]


def _record_recommendation(
    conn: sqlite3.Connection, user: str, sym: str, cfg: MomentumConfig,
    lookback: str, row: dict, as_of_date: str, config_dict: dict,
) -> dict:
    """Insert an IMMUTABLE recommendation row if one doesn't already exist
    for (user, symbol, strategy, as_of_date) -- the idempotency key used
    throughout this module and by api/momentum_monitor.py's cycle, so
    calling this twice for the same signal never creates a duplicate.
    `suggested_entry` (and every other field) is written exactly once here
    and never updated by any other function in this module. Shared by
    POST /snapshot (one symbol, human-triggered) and the monitoring
    cycle (a whole universe scan, machine-triggered) -- same code path,
    so the two can never silently diverge in what counts as a
    recommendation."""
    existing = conn.execute(
        "SELECT id FROM momentum_recommendations WHERE username=? AND symbol=? AND strategy=? AND as_of_date=?",
        (user, sym, cfg.tag, as_of_date),
    ).fetchone()
    if existing:
        return {"ok": True, "id": existing[0], "already_existed": True,
                "suggested_entry": row["entry_est"], "as_of_date": as_of_date}

    created_at = datetime.now().isoformat(timespec="seconds")
    reasons = json.dumps({
        "rank": row["rank"], "absolute_momentum": row["absolute_momentum"],
        "relative_momentum": row["relative_momentum"], "technical_confirmed": row["technical_confirmed"],
        "config": config_dict,
    })
    with conn:
        cur = conn.execute(
            "INSERT INTO momentum_recommendations (username, created_at, symbol, yahoo_symbol, strategy, "
            "lookback, as_of_date, suggested_entry, sl_est, tp_est, sl_atr, tp_atr, absolute_momentum, "
            "relative_momentum, rank, reasons, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'pending')",
            (user, created_at, sym, sym, cfg.tag, lookback, as_of_date, row["entry_est"],
             row["sl_est"], row["tp_est"], row["sl_atr"], row["tp_atr"], row["absolute_momentum"],
             row["relative_momentum"], row["rank"], reasons),
        )
    return {"ok": True, "id": cur.lastrowid, "already_existed": False,
            "suggested_entry": row["entry_est"], "as_of_date": as_of_date}


@router.post("/snapshot")
def create_snapshot(req: SnapshotReq, request: Request):
    """Recommendation -> immutable snapshot. Runs the SAME live cross-
    sectional screen GET /api/momentum/screen uses (core.strategies.
    momentum_strategy.screen_universe_now); if `symbol` is not CURRENTLY a
    fully-qualifying momentum candidate (selected + technically confirmed),
    nothing is written -- this table only ever records real signals, never
    a placeholder for "checked and nothing qualified"."""
    user = _paper_user(request)
    sym = req.symbol.strip().upper()
    if req.lookback not in _MOMENTUM_LOOKBACK_BARS:
        raise HTTPException(400, f"lookback must be one of {list(_MOMENTUM_LOOKBACK_BARS)}")

    universe = [s.strip().upper() for s in req.symbols.split(",") if s.strip()] if req.symbols else _MOMENTUM_UNIVERSE
    if sym not in universe:
        universe = universe + [sym]  # always screen the symbol being snapshotted, even if outside the default universe

    cfg = MomentumConfig(
        lookback_label=req.lookback, top_n=req.top_n,
        use_relative_momentum=req.use_relative,
        use_technical_confirmation=req.use_technical,
        use_market_regime=req.use_market_regime,
    )
    screen = screen_universe_now(universe, req.years, cfg, cache_dir=_MOMENTUM_CACHE_DIR)
    if screen["as_of"] is None:
        raise HTTPException(502, "No valid momentum data for this symbol/universe right now")

    row = next((r for r in screen["rows"] if r["symbol"] == sym), None)
    if row is None:
        return {"ok": False,
                "reason": (f"{sym} is not currently ranked as of {screen['as_of'].date()} -- either its data "
                           f"didn't load, or it fails the eligibility filters (min_absolute_momentum/"
                           f"min_relative_momentum), so it never entered the cross-sectional ranking at all"),
                "row": None}
    if row["signal"] != "long":
        return {"ok": False, "reason": f"{sym} is not currently a qualifying momentum candidate "
                                        f"(selected={row['selected']}, technical_confirmed={row['technical_confirmed']})",
                "row": row}

    as_of_date = str(screen["as_of"].date())
    conn = _db()
    try:
        config_dict = {"top_n": req.top_n, "use_relative": req.use_relative,
                        "use_technical": req.use_technical, "use_market_regime": req.use_market_regime}
        return _record_recommendation(conn, user, sym, cfg, req.lookback, row, as_of_date, config_dict)
    finally:
        conn.close()


@router.get("/recommendations")
def list_recommendations(request: Request, status: str | None = None):
    user = _paper_user(request)
    conn = _db()
    try:
        if status:
            rows = conn.execute(
                f"SELECT {','.join(_REC_COLS)} FROM momentum_recommendations WHERE username=? AND status=? "
                "ORDER BY id DESC", (user, status),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {','.join(_REC_COLS)} FROM momentum_recommendations WHERE username=? ORDER BY id DESC",
                (user,),
            ).fetchall()
        return {"recommendations": [_row_to_dict(r, _REC_COLS) for r in rows]}
    finally:
        conn.close()


def _execute_recommendation_row(
    conn: sqlite3.Connection, user: str, rec_id: int, rec_d: dict,
    actual_entry: float, qty: float, note: str | None,
) -> dict:
    """Fill a pending recommendation as an ACTUAL paper trade through the
    EXISTING api/paper.py ledger (`_fill`, imported unmodified). `actual_entry`
    is stored in its own column -- `suggested_entry` (written once at
    snapshot time by `_record_recommendation`) is never touched, so the two
    remain independently inspectable forever. Raises ValueError (insufficient
    cash, etc., surfaced by `_paper_fill`) or RuntimeError (wrong status) --
    callers decide how to turn those into an HTTP response or a skip."""
    if rec_d["status"] != "pending":
        raise RuntimeError(f"recommendation is already {rec_d['status']}")

    _paper_fill(conn, user, rec_d["symbol"], rec_d["yahoo_symbol"], "BUY", qty,
                actual_entry, rec_d["sl_est"], rec_d["tp_est"],
                note or f"momentum snapshot #{rec_id} ({rec_d['strategy']}, {rec_d['as_of_date']})")
    trade_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    executed_at = datetime.now().isoformat(timespec="seconds")
    with conn:
        conn.execute(
            "UPDATE momentum_recommendations SET status='executed', executed_at=?, actual_entry=?, "
            "paper_trade_id=? WHERE id=?",
            (executed_at, actual_entry, trade_id, rec_id),
        )
    gap_pct = round(100.0 * (actual_entry - rec_d["suggested_entry"]) / rec_d["suggested_entry"], 2) \
        if rec_d["suggested_entry"] else None
    return {"ok": True, "paper_trade_id": trade_id, "suggested_entry": rec_d["suggested_entry"],
            "actual_entry": actual_entry, "qty": qty, "executed_at": executed_at, "slippage_pct": gap_pct}


@router.post("/recommendations/{rec_id}/execute")
def execute_recommendation(rec_id: int, req: ExecuteReq, request: Request):
    """Fill a pending recommendation as an ACTUAL paper trade -- see
    `_execute_recommendation_row` (shared with api/momentum_monitor.py's
    cycle) for what actually happens."""
    user = _paper_user(request)
    if req.qty <= 0:
        raise HTTPException(422, "qty must be positive")
    if req.actual_entry <= 0:
        raise HTTPException(422, "actual_entry must be positive")

    conn = _db()
    try:
        rec = conn.execute(
            f"SELECT {','.join(_REC_COLS)} FROM momentum_recommendations WHERE id=? AND username=?",
            (rec_id, user),
        ).fetchone()
        if rec is None:
            raise HTTPException(404, "no such recommendation")
        rec_d = _row_to_dict(rec, _REC_COLS)
        try:
            return _execute_recommendation_row(conn, user, rec_id, rec_d, req.actual_entry, req.qty, req.note)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        except ValueError as e:
            raise HTTPException(422, str(e))
    finally:
        conn.close()


@router.post("/recommendations/{rec_id}/cancel")
def cancel_recommendation(rec_id: int, request: Request):
    user = _paper_user(request)
    conn = _db()
    try:
        with conn:
            n = conn.execute(
                "UPDATE momentum_recommendations SET status='cancelled' WHERE id=? AND username=? AND status='pending'",
                (rec_id, user),
            ).rowcount
        if not n:
            raise HTTPException(404, "no such pending recommendation")
        return {"ok": True}
    finally:
        conn.close()


@router.get("/recommendations/{rec_id}/outcome")
def recommendation_outcome(rec_id: int, request: Request):
    """Grade a recommendation: join it back to the paper_trades/paper_positions
    rows its execution created (same sqlite file), reporting suggested vs
    actual entry and, once the position is closed, realized P&L."""
    user = _paper_user(request)
    conn = _db()
    try:
        rec = conn.execute(
            f"SELECT {','.join(_REC_COLS)} FROM momentum_recommendations WHERE id=? AND username=?",
            (rec_id, user),
        ).fetchone()
        if rec is None:
            raise HTTPException(404, "no such recommendation")
        rec_d = _row_to_dict(rec, _REC_COLS)
        out = {"recommendation": rec_d, "position": None, "closing_trade": None}
        if rec_d["status"] != "executed":
            return out

        pos = conn.execute(
            "SELECT symbol, qty, cost_basis, opened_at, stop, target FROM paper_positions "
            "WHERE username=? AND symbol=?", (user, rec_d["symbol"]),
        ).fetchone()
        if pos:
            qty, cost_basis = pos[1], pos[2]
            out["position"] = {
                "symbol": pos[0], "qty": qty, "avg_cost": round(cost_basis / qty, 2) if qty else None,
                "opened_at": pos[3], "stop": pos[4], "target": pos[5],
            }
        # A SELL trade on this symbol placed AFTER the entry fill closed (all
        # or part of) the position this recommendation opened.
        closing = conn.execute(
            "SELECT id, ts, qty, price, realized_pnl FROM paper_trades WHERE username=? AND symbol=? "
            "AND side='SELL' AND id > ? ORDER BY id ASC", (user, rec_d["symbol"], rec_d["paper_trade_id"] or 0),
        ).fetchall()
        if closing:
            out["closing_trade"] = [
                {"id": c[0], "ts": c[1], "qty": c[2], "price": c[3], "realized_pnl": c[4]} for c in closing
            ]
        return out
    finally:
        conn.close()
