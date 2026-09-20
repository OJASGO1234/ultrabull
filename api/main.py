"""FastAPI backend exposing the ultrabull analysis pipeline as JSON.

Run from the repo root:
    venv/bin/uvicorn api.main:app --port 8000
"""
import os
import sys

# Headless matplotlib before anything in core/ imports pyplot.
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(ROOT, "core")
if CORE not in sys.path:
    sys.path.insert(0, CORE)

# Load repo-root .env (ANTHROPIC_API_KEY for the chat assistant, etc.)
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

import copy
import json
import math
import sqlite3
import threading
import time
from datetime import date as _date_cls
from typing import Literal

import anthropic
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from stock_explorer import load_data, compute_indicators, load_events
from market_context import (
    MARKET_INDEX, SECTOR_INDEX, SYMBOL_SECTOR,
    build_market_context, _load_index_close,
)
from levels import find_levels
from outlook import build_outlook
from advice import build_advice
from indicators import build_key_indicators
from fundamentals import fetch_fundamentals
from interpret import interpret_rsi, interpret_macd, interpret_equity
from idea_lab import evaluate_rule, backtest_strategy, FEATURES as IDEA_FEATURES

IDEA_CACHE_DIR = os.path.join(ROOT, "data_cache")
from vbt_to_pd_ind import build_ind_df_from_vbt, extract_symbol_df
from momentum import (
    LOOKBACK_BARS as MOMENTUM_LOOKBACK_BARS,
    LOOKBACK_MONTHS as MOMENTUM_LOOKBACK_MONTHS,
    compute_multi_period_momentum,
    technical_confirmation as momentum_technical_confirmation,
)
from patterns.config import CFG
from patterns.pipeline import run_all_patterns
from patterns.postprocess import enrich_detections, warmup_cutoff, summarize_signals
from patterns.explain import daily_summary_md
from rules.gates import apply_gates
from rules.signals import compile_signals
from sim.engine import simulate
from sim.metrics import summarize, by_tag, audit_exits
from pipeline import cfg as PIPELINE_BASE_CFG
from stock_analysis import build_stock_analysis

from .sre import router as sre_router
from .auth import make_token, verify_token, verify_user
from .live import router as live_router, groww_router
from .paper import router as paper_router
from .deals import router as deals_router
from .momentum_paper import router as momentum_paper_router

app = FastAPI(title="Ultrabull API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sre_router)
app.include_router(live_router)
app.include_router(groww_router)
app.include_router(paper_router)
app.include_router(deals_router)
app.include_router(momentum_paper_router)


@app.on_event("startup")
def _start_paper_executor():
    """Background loop that fills pending paper orders when live prices hit."""
    from .paper import start_executor
    start_executor()

# ---------- auth (enforced only when ULTRABULL_AUTH=1 — i.e. in production) ----------

AUTH_ENABLED = os.getenv("ULTRABULL_AUTH", "0") == "1"
AUTH_PUBLIC_PATHS = {"/api/health", "/api/auth/login"}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path
    if AUTH_ENABLED and path.startswith("/api") and path not in AUTH_PUBLIC_PATHS:
        authz = request.headers.get("authorization", "")
        token = authz[7:] if authz.lower().startswith("bearer ") else ""
        user = verify_token(token)
        if user is None:
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)
        request.state.user = user
    return await call_next(request)


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/auth/login")
def login(req: LoginRequest):
    if not verify_user(req.username, req.password):
        time.sleep(0.5)  # blunt brute-force throttle
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token, exp = make_token(req.username.strip().lower())
    return {"token": token, "username": req.username.strip().lower(), "expires_at": exp}


@app.get("/api/auth/me")
def auth_me(request: Request):
    """Who am I / is auth even on. The frontend calls this at boot to decide
    whether to show the login screen."""
    return {"auth_required": AUTH_ENABLED, "username": getattr(request.state, "user", None)}


def build_cfg() -> dict:
    """The module-level cfg in core/pipeline.py plus the mutations its
    __main__ block applies before simulating — kept identical so API results
    match a CLI `python pipeline.py` run."""
    cfg = copy.deepcopy(PIPELINE_BASE_CFG)
    cfg.setdefault("costs", {})
    cfg["costs"].update({
        "per_share": 0.0,
        "per_trade": 0.0,
        "bps_fee": 0.0,
        "slippage_bps": 2.0,
    })
    cfg["execution"]["max_hold_bars"] = 10
    cfg["patterns"]["BB_BuyDip"].update({"tp_atr": 1.6, "trail_atr": 1.0})
    cfg["patterns"]["BB_SellFade"].update({"tp_atr": 1.6, "trail_atr": 1.2})
    cfg.setdefault("gates", {})
    cfg["gates"]["bb_touch_eps"] = 0.0
    cfg["gates"]["pullback_ma_tol"] = 0.01
    cfg["execution"].update({"enter_at": "Open", "delay_bars": 1})
    # Evidence-backed prunes (35-symbol 4y event study): RSI crosses, MACD_Up,
    # Doji, and the whole short side except BearishEngulfing have no edge.
    cfg["disable_patterns"] = [
        "RSI_60_Down", "RSI_40_Up", "Doji", "MACD_Cross_Zeroish_Up",
        "BB_SellFade", "Breakdown20", "Breakdown55",
        "VCP_Setup",  # context only: the coil itself tested edge-less; the trade is the breakout from it
    ]
    # steward-applied runtime disables (Phase 1.5) — human-approved, reversible,
    # merged additively so the evidence-based base prunes are never re-enabled here
    try:
        from .steward import load_overrides
        cfg["disable_patterns"] = sorted(set(cfg["disable_patterns"]) | set(load_overrides()["disabled"]))
    except Exception:
        pass
    cfg["gates"]["market"] = {
        "enabled": True,
        "require_market_regime": True,
        "require_sector_regime": False,
        "rs_col": "rs_20",
        "breakout_rs_min": 0.0,
        "breakdown_rs_max": 0.0,
        # distribution-day ceiling validated (pool excess @10: +15.8 -> +29.3 bps);
        # breadth_min tested and left OFF (redundant with NIFTY regime on this universe)
        "dist_days_max": 5.0,
        "breadth_min": 0.0,
    }
    cfg["gates"]["events"] = {
        "enabled": True,
        "earnings_blackout_before": 3,
        "earnings_blackout_after": 1,
    }
    cfg["gates"]["adx"] = {"breakout_min": 0.0, "meanrev_max": 0.0}  # evaluated & rejected
    cfg["gates"]["liquidity"] = {"min_turnover_20d": 2e7}
    cfg.setdefault("equity", {})
    cfg["equity"]["mtm"] = True
    cfg["equity"]["mtm_haircut_bps"] = 2.0
    return cfg


def stock_only_cfg(cfg: dict) -> dict:
    """`cfg` with the market-timing filters lifted — the config behind the
    stock-only view.

    What this ACTUALLY changes as a backtest gate is narrower than it looks:
    build_cfg already ships require_sector_regime=False and breadth_min=0.0
    (both tested and left off), so only two gates really flip — the NIFTY
    regime requirement and the distribution-day ceiling. The sector-regime and
    breadth entries below are set defensively, not because they were on.

    Separately, and this is the part users see: build_advice drops all four
    market factors (regime, sector, breadth, distribution days) from the SCORE
    in this view. So the toggle changes the advice card more than it changes
    the backtest. Do not describe the two as if they were the same lift.

    ON:  relative strength vs sector (stock selection, separately validated),
         earnings blackout, liquidity floor, and every stock-level gate.

    This exists so the UI can show what the analysis looks like without the
    market-timing layer — NOT because that view is equally supported. The
    regime gate is strongly validated (pool excess -6.9 -> +9.6 bps at 10
    bars). The distribution-day ceiling is weaker than its original headline:
    core/dist_day_study.py (Jul 2026) finds recent-12m separation only at 5-10
    bars (block-bootstrap p 0.07/0.06), none in the prior 3 years, and a sign
    flip at 1 bar — kept as shipped because the evidence cannot move it either
    way, not because it was re-confirmed.
    """
    c = copy.deepcopy(cfg)
    c.setdefault("gates", {}).setdefault("market", {})
    c["gates"]["market"].update({
        "enabled": True,            # keep the block alive so RS still applies
        "require_market_regime": False,
        "require_sector_regime": False,
        "dist_days_max": 0.0,       # 0 disables the ceiling
        "breadth_min": 0.0,
    })
    return c


# ---------- JSON-safe serialization helpers ----------

def _metrics_json(metrics: dict) -> dict:
    """Backtest summary -> JSON-safe dict (NumPy floats break FastAPI)."""
    metrics = metrics or {}
    return {
        "trades": int(metrics.get("trades", 0)),
        "win_rate": _num(metrics.get("win_rate")),
        "avg_win": _num(metrics.get("avg_win")),
        "avg_loss": _num(metrics.get("avg_loss")),
        "profit_factor": _num_or_inf(metrics.get("profit_factor")),
        "expectancy": _num(metrics.get("expectancy")),
        "median_hold_bars": _num(metrics.get("median_hold_bars")),
        "return_total": _num(metrics.get("return_total")),
        "max_drawdown": _num(metrics.get("max_drawdown")),
        "sharpe_daily": _num(metrics.get("sharpe_daily")),
    }


def _by_pattern_json(by_pat) -> list[dict]:
    """Per-pattern backtest breakdown -> JSON-safe rows."""
    if by_pat is None or not len(by_pat):
        return []
    return [{
        "tag": tag,
        "trades": int(r["trades"]),
        "win_rate": _num(r["win_rate"]),
        "total_pnl": _num(r["total_pnl"]),
        "avg_pnl": _num(r["avg_pnl"]),
        "profit_factor": _num_or_inf(r["profit_factor"]),
    } for tag, r in by_pat.iterrows()]


def _num(x):
    """float | None; NaN/Inf -> None (JSON has no NaN)."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _num_or_inf(x):
    """Like _num but preserves +inf as the string 'inf' (profit factor)."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    if math.isinf(f):
        return "inf"
    return f


def _date(ts) -> str:
    return pd.Timestamp(ts).date().isoformat()


def _series_points(s: pd.Series) -> list[dict]:
    out = []
    for ts, v in s.items():
        f = _num(v)
        if f is not None:
            out.append({"time": _date(ts), "value": f})
    return out


def _ohlcv_rows(df: pd.DataFrame) -> list[dict]:
    out = []
    for ts, r in df.iterrows():
        o, h, l, c = _num(r["Open"]), _num(r["High"]), _num(r["Low"]), _num(r["Close"])
        if None in (o, h, l, c):
            continue
        out.append({
            "time": _date(ts),
            "open": o, "high": h, "low": l, "close": c,
            "volume": _num(r["Volume"]) or 0.0,
        })
    return out


def _ts_intraday(ts) -> int:
    """Naive IST bar timestamp -> UTC epoch seconds (lightweight-charts time)."""
    return int(pd.Timestamp(ts).tz_localize("Asia/Kolkata").timestamp())


def _series_points_t(s: pd.Series, tf) -> list[dict]:
    out = []
    for ts, v in s.items():
        f = _num(v)
        if f is not None:
            out.append({"time": tf(ts), "value": f})
    return out


def _ohlcv_rows_t(df: pd.DataFrame, tf) -> list[dict]:
    out = []
    for ts, r in df.iterrows():
        o, h, l, c = _num(r["Open"]), _num(r["High"]), _num(r["Low"]), _num(r["Close"])
        if None in (o, h, l, c):
            continue
        out.append({"time": tf(ts), "open": o, "high": h, "low": l, "close": c,
                    "volume": _num(r["Volume"]) or 0.0})
    return out


def _bar_date(df_ohlcv: pd.DataFrame, bar: int) -> str | None:
    if bar is None or bar < 0 or bar >= len(df_ohlcv):
        return None
    return _date(df_ohlcv.index[int(bar)])


# ---------- event/timing helpers ----------

def _compute_timing(symbol: str, events: pd.DataFrame | None, cfg: dict) -> dict:
    """Forward-looking timing intelligence: where are we in the results cycle,
    is the earnings blackout active, upcoming ex-dividend. Best-effort — any
    yfinance hiccup degrades to nulls, never fails the request."""
    today = pd.Timestamp(_date_cls.today())
    egate = (cfg.get("gates") or {}).get("events") or {}
    blackout_before = int(egate.get("earnings_blackout_before", 3))

    out = {
        "next_earnings": None,
        "trading_days_to_next_earnings": None,
        "in_earnings_blackout": False,
        "last_earnings": None,
        "days_since_last_earnings": None,
        "results_overdue": False,
        "next_ex_dividend": None,
        "last_dividend": None,
        "last_split": None,
        "earnings_data_stale": False,
        # deterministic projection from the company's own reporting cadence,
        # used when no confirmed future date exists ("heuristics compute facts")
        "expected_results_window": None,
    }

    # Past events from the tidy frame (already clipped to the price window)
    if events is not None and len(events):
        et = events.copy()
        et["date"] = pd.to_datetime(et["date"])
        past = et[et["date"] <= today]
        for typ, key in (("EARNINGS", "last_earnings"), ("DIVIDEND", "last_dividend"), ("SPLIT", "last_split")):
            rows = past[past["type"].astype(str).str.upper() == typ]
            if len(rows):
                r = rows.iloc[-1]
                out[key] = {"date": _date(r["date"]), "label": str(r.get("label", typ))}

    # Future earnings via yfinance (the tidy frame only covers the past window)
    try:
        ed = yf.Ticker(symbol).get_earnings_dates(limit=12)
        if ed is not None and len(ed):
            idx = pd.to_datetime(ed.index)
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_localize(None)
            idx = idx.normalize()
            future = sorted(d for d in idx if d > today)
            past = sorted(d for d in idx if d <= today)
            if past:
                prev = out["last_earnings"]
                # keep whichever known past date is fresher
                if prev is None or past[-1] > pd.Timestamp(prev["date"]):
                    out["last_earnings"] = {"date": _date(past[-1]), "label": "E"}
            if future:
                nxt = future[0]
                tdays = int(np.busday_count(today.date(), nxt.date()))
                out["next_earnings"] = _date(nxt)
                out["trading_days_to_next_earnings"] = tdays
                out["in_earnings_blackout"] = tdays <= blackout_before
            elif len(past) >= 3:
                # No confirmed date — project one from the historical cadence
                # (Indian companies report ~quarterly; use THIS company's median
                # gap, not a blanket 91 days). Skip when Yahoo's coverage is so
                # stale the projection would be guesswork.
                gaps = [(past[i + 1] - past[i]).days for i in range(len(past) - 1)]
                med = float(np.median([g for g in gaps if 30 <= g <= 200] or gaps))
                if 30 <= med <= 200 and (today - past[-1]).days <= 400:
                    expected = past[-1] + pd.Timedelta(days=med)
                    while expected < today - pd.Timedelta(days=14):
                        expected += pd.Timedelta(days=med)  # roll to the next cycle
                    out["expected_results_window"] = {
                        "start": _date(expected - pd.Timedelta(days=7)),
                        "end": _date(expected + pd.Timedelta(days=7)),
                        "basis": f"projected from a median {med:.0f}-day gap across "
                                 f"{len(past)} past results (not an announced date)",
                    }
    except Exception:
        pass

    if out["last_earnings"] is not None:
        days_since = int((today - pd.Timestamp(out["last_earnings"]["date"])).days)
        out["days_since_last_earnings"] = days_since
        # Indian listed companies report quarterly (~91d cadence); if no known
        # future date and the last one is 80-200 days old, results are due soon.
        # Beyond ~200 days Yahoo's coverage for this name is simply stale
        # (common on NSE midcaps) — flag it instead of claiming "overdue".
        out["earnings_data_stale"] = days_since > 200
        out["results_overdue"] = (
            out["next_earnings"] is None and 80 <= days_since <= 200
        )

    try:
        cal = yf.Ticker(symbol).calendar or {}
        exd = cal.get("Ex-Dividend Date")
        if exd is not None:
            exd_ts = pd.Timestamp(exd)
            if exd_ts >= today:
                out["next_ex_dividend"] = _date(exd_ts)
    except Exception:
        pass

    return out


def _context_snapshot(symbol: str, ctx: pd.DataFrame | None, ind_df: pd.DataFrame) -> dict:
    out = {
        "market_regime": None, "sector_regime": None, "sector": SYMBOL_SECTOR.get(symbol),
        "rs_20": None, "rs_60": None, "adx": None, "breadth": None, "dist_days": None,
        # True when no sector index is mapped, so rs_* is measured against NIFTY
        # itself — the one way market data still reaches the stock-only view's
        # RS gate and setup score. Surfaced so the UI can stop calling it "sector".
        "rs_vs_market": SYMBOL_SECTOR.get(symbol) is None,
    }
    if ctx is not None and len(ctx):
        last = ctx.iloc[-1]
        out["market_regime"] = "uptrend" if bool(last.get("mkt_up")) else ("downtrend" if bool(last.get("mkt_dn")) else "neutral")
        out["sector_regime"] = "uptrend" if bool(last.get("sect_up")) else ("downtrend" if bool(last.get("sect_dn")) else "neutral")
        out["rs_20"] = _num(last.get("rs_20"))
        out["rs_60"] = _num(last.get("rs_60"))
        out["breadth"] = _num(last.get("breadth"))
        out["dist_days"] = _num(last.get("dist_days"))
    if "ADX" in ind_df.columns and len(ind_df):
        out["adx"] = _num(ind_df["ADX"].iloc[-1])
    return out


# ---------- pipeline runner ----------

def run_analysis(symbol: str, years: float, start: str | None = None, end: str | None = None) -> dict:
    df = load_data(symbol, years, start=start, end=end)
    # yfinance can return an in-progress session as a row with NaN OHLC (the
    # bar hasn't settled yet) while still reporting partial volume -- trim
    # that unsettled tail here, at the single point everything downstream
    # (indicators, patterns, advice, outlook, momentum) branches from, so
    # every one of those calculations keeps its existing assumption that
    # `close` is always a real number. Pre-existing, unrelated to the
    # momentum strategy added below; without this, ANY symbol's analysis
    # (not just momentum) 500s on a NaN reaching JSON serialization
    # (e.g. core/advice.py's plan levels) whenever the fetch lands mid-bar.
    close_col = df["close"] if len(df) else None
    if isinstance(close_col, pd.DataFrame):  # MultiIndex columns (yfinance, single symbol) -> squeeze
        close_col = close_col.iloc[:, 0]
    if close_col is not None and close_col.isna().any():
        last_valid = close_col.last_valid_index()
        if last_valid is not None:
            df = df.loc[:last_valid]
    ind = compute_indicators(df)

    # context/index loaders fetch from today backwards — make sure they reach
    # back to the window start even for historical custom ranges
    if start:
        years = max(years, (pd.Timestamp.now() - pd.Timestamp(start)).days / 365.0)

    df_ohlcv = extract_symbol_df(df, symbol=symbol, case="title")
    ind_df = build_ind_df_from_vbt(df_ohlcv, ind)

    empty_dets = pd.DataFrame(
        columns=["date", "pattern", "score", "strength", "regime", "notes", "params"]
    )
    try:
        detections = run_all_patterns(df_ohlcv, ind_df, CFG)
    except ValueError:  # pd.concat with zero non-empty frames
        detections = empty_dets

    cfg = build_cfg()

    # market/sector context + corporate events — best-effort, never fatal
    try:
        ctx = build_market_context(symbol, df_ohlcv["Close"], years=max(years, 1.0))
    except Exception:
        ctx = None
    try:
        events = load_events(symbol, df_ohlcv.rename(columns=str.lower))
    except Exception:
        events = pd.DataFrame(columns=["date", "type", "label"])

    if detections is None or detections.empty:
        det_enriched = empty_dets.assign(direction=None, alignment=None, qualified=False)
        signals = pd.DataFrame(columns=["bar_idx", "side", "tag"])
        trades = pd.DataFrame()
        equity = pd.Series(cfg["risk"]["capital"], index=df_ohlcv.index, dtype=float)
        sig_summary = {}
    else:
        det_enriched = enrich_detections(detections, df_ohlcv, ind_df, symbol, ctx=ctx)
        cut = warmup_cutoff(ind_df)
        det_enriched = det_enriched[det_enriched["date"] >= cut]
        sig_summary = summarize_signals(det_enriched)

        det_gated = apply_gates(det_enriched, df_ohlcv, ind_df, cfg, ctx=ctx, events=events)
        signals = compile_signals(det_gated, df_ohlcv, ind_df, cfg)
        trades, equity, _fills = simulate(df_ohlcv, signals, ind_df, cfg)

    metrics = summarize(trades, equity)
    by_pat = by_tag(trades)
    audit_bad = audit_exits(trades, cfg)
    summary_md = daily_summary_md(det_enriched, df_ohlcv, ind_df, symbol)

    # ---- serialize ----
    macd_hist = ind_df["MACD"] - ind_df["MACD_signal"]

    # plan levels (entry reference / stop / target in ₹) per detection, from the
    # same per-pattern ATR multiples the simulator trades with
    close_by_date = df_ohlcv["Close"]

    def _plan(pattern: str, direction, date=None, bar: int | None = None,
              sl_atr=None, tp_atr=None, side=None):
        p_cfg = (cfg.get("patterns") or {}).get(pattern, {})
        side = side or p_cfg.get("side") or ("short" if direction == "bearish" else "long")
        sl_k = float(sl_atr if sl_atr is not None else p_cfg.get("sl_atr", cfg["defaults"]["sl_atr"]))
        tp_k = float(tp_atr if tp_atr is not None else p_cfg.get("tp_atr", cfg["defaults"]["tp_atr"]))
        try:
            if bar is not None:
                ref = float(df_ohlcv["Close"].iloc[bar])
                a = float(ind_df["ATR"].iloc[bar])
            else:
                ref = float(close_by_date.loc[date])
                a = float(ind_df["ATR"].loc[date])
        except (KeyError, IndexError):
            return {"side": side, "sl_atr": sl_k, "tp_atr": tp_k,
                    "ref_close": None, "sl_est": None, "tp_est": None}
        if not (math.isfinite(ref) and math.isfinite(a)):
            return {"side": side, "sl_atr": sl_k, "tp_atr": tp_k,
                    "ref_close": None, "sl_est": None, "tp_est": None}
        sgn = 1.0 if side == "long" else -1.0
        return {
            "side": side, "sl_atr": sl_k, "tp_atr": tp_k,
            "ref_close": round(ref, 2),
            "sl_est": round(ref - sgn * sl_k * a, 2),
            "tp_est": round(ref + sgn * tp_k * a, 2),
        }

    def _detections_rows(dets) -> list[dict]:
        rows = []
        for r in dets.itertuples(index=False):
            plan = _plan(getattr(r, "pattern"), getattr(r, "direction", None),
                         date=getattr(r, "date"))
            rows.append({
                "date": _date(getattr(r, "date")),
                "pattern": getattr(r, "pattern"),
                "score": _num(getattr(r, "score", None)),
                "strength": getattr(r, "strength", None),
                "regime": getattr(r, "regime", None),
                "direction": getattr(r, "direction", None),
                "alignment": getattr(r, "alignment", None),
                "qualified": bool(getattr(r, "qualified", False)),
                "english": getattr(r, "english", None),
                "side": plan["side"],
                "ref_close": plan["ref_close"],
                "sl_est": plan["sl_est"],
                "tp_est": plan["tp_est"],
            })
        return rows

    detections_json = _detections_rows(det_enriched)

    # Both views serialize signals/trades the same way — the pools differ, not
    # the shape.
    def _signals_rows(sig) -> list[dict]:
        rows = []
        for r in sig.itertuples(index=False):
            bar = int(getattr(r, "bar_idx"))
            plan = _plan(getattr(r, "tag"), None, bar=bar,
                         sl_atr=getattr(r, "sl_atr", None), tp_atr=getattr(r, "tp_atr", None),
                         side=getattr(r, "side", None))
            rows.append({
                "bar_idx": bar,
                "date": _bar_date(df_ohlcv, bar),
                "side": getattr(r, "side"),
                "tag": getattr(r, "tag"),
                "score": _num(getattr(r, "score", None)),
                "priority": _num(getattr(r, "priority", None)),
                "enter_at": getattr(r, "enter_at", None),
                "delay_bars": _num(getattr(r, "delay_bars", None)),
                "sl_atr": _num(getattr(r, "sl_atr", None)),
                "tp_atr": _num(getattr(r, "tp_atr", None)),
                "trail_atr": _num(getattr(r, "trail_atr", None)),
                "ref_close": plan["ref_close"],
                "sl_est": plan["sl_est"],
                "tp_est": plan["tp_est"],
            })
        return rows

    signals_json = _signals_rows(signals)

    def _trades_rows(trd) -> list[dict]:
        rows = []
        if not len(trd):
            return rows
        for r in trd.itertuples(index=False):
            rows.append({
                "entry_bar": int(r.entry_bar),
                "exit_bar": int(r.exit_bar),
                "entry_date": _bar_date(df_ohlcv, int(r.entry_bar)),
                "exit_date": _bar_date(df_ohlcv, int(r.exit_bar)),
                "side": r.side,
                "tag": r.tag,
                "entry_price": _num(r.entry_price),
                "exit_price": _num(r.exit_price),
                "size_shares": _num(r.size_shares),
                "sl_price": _num(r.sl_price),
                "tp_price": _num(r.tp_price),
                "pnl": _num(r.pnl),
                "fees": _num(r.fees),
                "reason": r.reason,
            })
        return rows

    trades_json = _trades_rows(trades)

    by_pattern_json = _by_pattern_json(by_pat)

    n_checked = int((trades["reason"].isin(["SL", "TP"])).sum()) if len(trades) else 0
    audit_json = {
        "checked": n_checked,
        "mismatch_count": int(len(audit_bad)) if audit_bad is not None else 0,
        "ok": (audit_bad is None) or (len(audit_bad) == 0),
    }

    signal_summary_json = {}
    if sig_summary:
        signal_summary_json = {
            "total_hits": _num(sig_summary.get("total_hits")),
            "per_week_all": _num(sig_summary.get("per_week_all")),
            "qualified_hits": _num(sig_summary.get("qualified_hits")),
            "per_week_qualified": _num(sig_summary.get("per_week_qualified")),
            "by_family": {k: int(v) for k, v in (sig_summary.get("by_family") or {}).items()},
        }

    events_json = []
    if events is not None and len(events):
        for r in events.itertuples(index=False):
            events_json.append({
                "date": _date(getattr(r, "date")),
                "type": str(getattr(r, "type", "")).upper(),
                "label": str(getattr(r, "label", "")),
            })

    timing_json = _compute_timing(symbol, events, cfg)
    context_json = _context_snapshot(symbol, ctx, ind_df)

    # support/resistance from the most recent ~1y of bars, plus outlook
    try:
        look = min(len(df_ohlcv), 260)
        raw_levels = find_levels(df_ohlcv.iloc[-look:], ind_df.iloc[-look:])
    except Exception:
        raw_levels = []
    levels_json = [{
        "price": lv["price"],
        "kind": lv["kind"],
        "touches": lv["touches"],
        "strength": _num(lv["strength"]),
        "last_touch": _date(lv["last_touch"]),
        "source": lv["source"],
        "distance_pct": _num(lv["distance_pct"]),
    } for lv in raw_levels]

    try:
        outlook_json = build_outlook(ind_df, ctx, raw_levels, float(df_ohlcv["Close"].iloc[-1]))
    except Exception:
        outlook_json = None

    try:
        advice_json = build_advice(df_ohlcv, ind_df, ctx, raw_levels, timing_json, det_enriched)
    except Exception:
        advice_json = None

    # ---- stock-only view: the same analysis with the market-timing filters
    # lifted, computed here so the UI toggle is instant and both views come
    # from one data fetch. Never the default; see stock_only_cfg().
    try:
        so_outlook_json = build_outlook(ind_df, ctx, raw_levels,
                                        float(df_ohlcv["Close"].iloc[-1]),
                                        use_market_context=False)
    except Exception:
        so_outlook_json = None

    try:
        so_advice_json = build_advice(df_ohlcv, ind_df, ctx, raw_levels, timing_json,
                                      det_enriched, use_market_context=False)
    except Exception:
        so_advice_json = None

    # …and what lifting them costs in this symbol's own backtest. The whole
    # backtest is recomputed, not just the headline metrics, so the UI can show
    # one coherent story per view instead of stock-only verdicts sitting above
    # a market-aware equity curve.
    so_metrics = so_by_pattern = None
    so_signals_json: list[dict] = []
    so_trades_json: list[dict] = []
    so_equity_points: list[dict] = []
    so_detections_json: list[dict] = detections_json
    try:
        if detections is not None and not detections.empty:
            so_cfg = stock_only_cfg(cfg)
            # Re-enrich without the sector-relative term so this view's detection
            # SCORES are market-free too. Gating is unaffected (the RS gate reads
            # ctx directly, not this frame) — only the displayed score changes.
            so_det_enriched = enrich_detections(detections, df_ohlcv, ind_df, symbol,
                                                ctx=ctx, use_market_context=False)
            so_det_enriched = so_det_enriched[so_det_enriched["date"] >= cut]
            so_gated = apply_gates(so_det_enriched, df_ohlcv, ind_df, so_cfg, ctx=ctx, events=events)
            so_signals = compile_signals(so_gated, df_ohlcv, ind_df, so_cfg)
            so_trades, so_equity, _ = simulate(df_ohlcv, so_signals, ind_df, so_cfg)
            so_metrics = summarize(so_trades, so_equity)
            so_by_pattern = by_tag(so_trades)
            so_signals_json = _signals_rows(so_signals)
            so_trades_json = _trades_rows(so_trades)
            so_equity_points = _series_points(so_equity)
            so_detections_json = _detections_rows(so_det_enriched)
    except Exception:
        so_metrics = so_by_pattern = None
        so_signals_json, so_trades_json, so_equity_points = [], [], []
        so_detections_json = detections_json

    try:
        key_indicators_json = build_key_indicators(df_ohlcv, ind_df, ctx, raw_levels)
    except Exception:
        key_indicators_json = None

    try:
        so_key_indicators_json = build_key_indicators(df_ohlcv, ind_df, ctx, raw_levels,
                                                      use_market_context=False)
    except Exception:
        so_key_indicators_json = None

    try:
        fundamentals_json = fetch_fundamentals(symbol)
    except Exception:
        fundamentals_json = None

    # deterministic plain-English chart readings (see core/interpret.py)
    try:
        interpretations_json = {
            "rsi": interpret_rsi(ind_df, df_ohlcv["Close"]),
            "macd": interpret_macd(ind_df),
            "equity": interpret_equity(equity, trades),
        }
    except Exception:
        interpretations_json = None

    # NIFTY + sector index closes aligned to the stock's trading days, for
    # rebased overlay on the price chart
    def _index_series(ticker: str) -> list[dict]:
        s = _load_index_close(ticker, max(years, 1.0) + 0.5)
        if s is None:
            return []
        return _series_points(s.reindex(df_ohlcv.index).ffill())

    indexes_json = []
    try:
        pts = _index_series(MARKET_INDEX)
        if pts:
            indexes_json.append({"name": "NIFTY 50", "points": pts})
        sect_key = SYMBOL_SECTOR.get(symbol)
        if sect_key and sect_key in SECTOR_INDEX:
            pts = _index_series(SECTOR_INDEX[sect_key])
            if pts:
                indexes_json.append({"name": f"NIFTY {sect_key}", "points": pts})
    except Exception:
        pass

    metrics_json = _metrics_json(metrics)

    # Both views of the same run. Top-level advice/outlook/metrics stay the
    # market-aware ones so chat, steward and paper keep their existing contract.
    views_json = {
        "market_aware": {
            "label": "Market-aware (default)",
            "advice": advice_json,
            "outlook": outlook_json,
            "key_indicators": key_indicators_json,
            "metrics": metrics_json,
            "by_pattern": by_pattern_json,
            "signals": signals_json,
            "trades": trades_json,
            "equity": _series_points(equity),
            "detections": detections_json,
        },
        "stock_only": {
            "label": "Stock-only (market context ignored)",
            "advice": so_advice_json,
            "outlook": so_outlook_json,
            "key_indicators": so_key_indicators_json,
            "metrics": _metrics_json(so_metrics) if so_metrics is not None else None,
            "by_pattern": _by_pattern_json(so_by_pattern),
            "signals": so_signals_json,
            "trades": so_trades_json,
            "equity": so_equity_points,
            "detections": so_detections_json,
            # Split deliberately: conflating "gates that stopped trades in the
            # backtest" with "factors that moved the advice score" would claim
            # more was lifted than actually was.
            "lifted_gates": [
                "NIFTY market-regime requirement",
                "distribution-day ceiling (never applied to Hammer even by default)",
            ],
            "lifted_score_factors": [
                "NIFTY market regime",
                "sector regime",
                "market breadth",
                "distribution days",
            ],
            "already_off": [
                "sector-regime gate and breadth floor were NOT active as backtest "
                "gates in the default view either — both were tested and left off",
            ],
            "kept_filters": [
                ("relative strength vs sector — STILL A HARD GATE: breakouts must have "
                 "rs_20 >= 0. Unlike breadth and the distribution-day ceiling, the RS "
                 "threshold has no 'off' guard, so 0.0 means 'must be leading', not "
                 "'disabled'" + (
                     f". No sector index is mapped for {symbol}, so this is measured "
                     "against NIFTY itself — market data, in a view that says it ignores "
                     "the market"
                     if SYMBOL_SECTOR.get(symbol) is None else
                     f" (benchmark: NIFTY {SYMBOL_SECTOR[symbol]})")),
                "relative strength also scores into the setup axis and the base-rate "
                "lookup key (rs=+/-), so it moves this view's score and outlook too",
                "earnings blackout",
                "liquidity floor",
                "all stock-level pattern gates",
            ],
            "warning": (
                "The market-regime gate lifted here is strongly validated: removing it "
                "took pooled excess from +9.6 bps back to -6.9 bps at 10 bars. The "
                "distribution-day ceiling is weaker — a Jul-2026 recheck "
                "(core/dist_day_study.py) found its recent-12-month separation "
                "directionally right at 5-10 bars (+74/+167 bps, one-sided block-bootstrap "
                "p 0.07 and 0.06) but absent in the prior 3 years and REVERSED at 1 bar, "
                "so it was left as shipped rather than re-tuned. It binds hard at the "
                "moment — 85.8% of the last 120 sessions vs 50.9% across 4 years — partly "
                "because the threshold sits at the 4-year median and partly because the "
                "count has genuinely shifted right (median 6 over the last 120 sessions). "
                "This view is also more permissive by construction: it removes up to 4.0 "
                "points of drag against only 2.0 of lift, which softens the holding-side "
                "exit call as well as loosening fresh buys. Use it to understand the stock "
                "in isolation, not as a second opinion of equal standing."
            ),
        },
    }

    indicators_json = {
        "ma20": _series_points(ind_df["MA20"]),
        "ma50": _series_points(ind_df["MA50"]),
        "bb_upper": _series_points(ind_df["BB_upper"]),
        "bb_lower": _series_points(ind_df["BB_lower"]),
        "rsi": _series_points(ind_df["RSI"]),
        "atr": _series_points(ind_df["ATR"]),
        "macd": _series_points(ind_df["MACD"]),
        "macd_signal": _series_points(ind_df["MACD_signal"]),
        "macd_hist": _series_points(macd_hist),
        "adx": _series_points(ind_df["ADX"]) if "ADX" in ind_df.columns else [],
    }
    last_date = _date(df_ohlcv.index[-1])

    # Momentum / relative-strength diagnostics (new strategy, see
    # core/momentum.py + core/strategies/momentum_strategy.py). This is a
    # per-symbol DIAGNOSTIC panel only -- absolute/relative momentum at each
    # of 3M/6M/9M/12M plus the same technical-confirmation flags the
    # strategy trades on -- not a cross-sectional rank (ranking requires
    # comparing against the whole universe on the same date; see
    # GET /api/momentum/screen for that). Reuses the SAME NIFTY series
    # `ctx` above already fetched (market_context's in-memory index cache),
    # so this costs no extra network call.
    try:
        bench_close = _load_index_close(MARKET_INDEX, max(years, 1.0) + 0.5)
        mom_df = compute_multi_period_momentum(df_ohlcv["Close"], bench_close)
        mom_tech = momentum_technical_confirmation(df_ohlcv["Close"], ind_df)
        # yfinance can return TODAY's row with a NaN close for a not-yet-settled
        # session (see stock_explorer.load_data); anchor on the last row with a
        # real close so the panel doesn't go blank for the whole trading day.
        valid_idx = df_ohlcv["Close"].last_valid_index()
        mom_row = mom_df.loc[valid_idx] if valid_idx is not None else mom_df.iloc[-1]
        tech_row = mom_tech.loc[valid_idx] if valid_idx is not None else mom_tech.iloc[-1]
        momentum_json = {
            "benchmark": MARKET_INDEX,
            "as_of": _date(valid_idx) if valid_idx is not None else last_date,
            "lookbacks": {
                label: {
                    "months": MOMENTUM_LOOKBACK_MONTHS[label],
                    "bars": MOMENTUM_LOOKBACK_BARS[label],
                    "absolute_momentum": _num(mom_row.get(f"momentum_{label}")),
                    "relative_momentum": _num(mom_row.get(f"relative_momentum_{label}")),
                }
                for label in MOMENTUM_LOOKBACK_BARS
            },
            "technical_confirmation": {
                "close_above_ma20": bool(tech_row["close_above_ma20"]),
                "close_above_ma50": bool(tech_row["close_above_ma50"]),
                "ma20_above_ma50": bool(tech_row["ma20_above_ma50"]),
                "confirmed": bool(tech_row["confirmed"]),
            },
            "note": (
                "Per-symbol diagnostic values, not a cross-sectional rank or a "
                "trading recommendation by themselves. The momentum strategy's "
                "backtested track record (if any) is in "
                "core/momentum_experiment_results.csv, produced by "
                "core/research_momentum_experiment.py -- it has not been "
                "validated as profitable and should not be read as advice."
            ),
        }
    except Exception:
        momentum_json = None

    # Unified, typed view of everything above for the AI recommendation layer
    # (and any other future consumer) to read from one object instead of ~10
    # loose locals. Purely additive: every key below this point already
    # existed before this field was added. See core/stock_analysis.py --
    # thin adapter only, computes nothing new.
    try:
        stock_analysis_obj = build_stock_analysis(
            symbol=symbol,
            as_of=last_date,
            fundamentals_json=fundamentals_json,
            indicators_json=indicators_json,
            key_indicators_json=key_indicators_json,
            detections_json=detections_json,
            context_json=context_json,
            signals_json=signals_json,
            advice_json=advice_json,
            outlook_json=outlook_json,
            levels_json=levels_json,
            metrics_json=metrics_json,
        )
        stock_analysis_json = stock_analysis_obj.to_dict()
    except Exception:
        stock_analysis_json = None

    return {
        "views": views_json,
        "symbol": symbol,
        "years": years,
        "bars": len(df_ohlcv),
        "last_date": last_date,
        "ohlcv": _ohlcv_rows(df_ohlcv),
        "indicators": indicators_json,
        "events": events_json,
        "context": context_json,
        "timing": timing_json,
        "levels": levels_json,
        "outlook": outlook_json,
        "advice": advice_json,
        "key_indicators": key_indicators_json,
        "fundamentals": fundamentals_json,
        "interpretations": interpretations_json,
        "indexes": indexes_json,
        "detections": detections_json,
        "signals": signals_json,
        "trades": trades_json,
        "equity": _series_points(equity),
        "metrics": metrics_json,
        "by_pattern": by_pattern_json,
        "audit": audit_json,
        "signal_summary": signal_summary_json,
        "summary_md": summary_md,
        "stock_analysis": stock_analysis_json,
        "momentum": momentum_json,
    }


# ---------- TTL cache (single-flight per key) ----------

CACHE_TTL_SECONDS = float(os.getenv("ULTRABULL_CACHE_TTL", "900"))  # 15 min default
CACHE_MAX_ENTRIES = 32

_cache: dict[tuple[str, float], tuple[float, dict]] = {}  # key -> (stored_at, result)
_cache_lock = threading.Lock()
_inflight: dict[tuple[str, float], threading.Lock] = {}


def _cache_get(key: tuple[str, float]) -> dict | None:
    with _cache_lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < CACHE_TTL_SECONDS:
            return hit[1]
        return None


def _cache_put(key: tuple[str, float], result: dict) -> None:
    with _cache_lock:
        now = time.time()
        for k in [k for k, (ts, _) in _cache.items() if now - ts >= CACHE_TTL_SECONDS]:
            del _cache[k]
        if len(_cache) >= CACHE_MAX_ENTRIES:
            oldest = min(_cache, key=lambda k: _cache[k][0])
            del _cache[oldest]
        _cache[key] = (now, result)


def cached_analysis(symbol: str, years: float, start: str | None = None,
                    end: str | None = None, force: bool = False) -> tuple[dict, bool]:
    """Return (result, was_cached). Concurrent misses on the same key run the
    pipeline once; the rest wait on the per-key lock and read the fresh entry.
    force=True skips cache reads (still writes) — a guaranteed recompute."""
    key = (symbol, years, start or "", end or "")
    cached = None if force else _cache_get(key)
    if cached is not None:
        return cached, True

    with _cache_lock:
        flight = _inflight.setdefault(key, threading.Lock())
    with flight:
        cached = None if force else _cache_get(key)  # a racer may have filled it while we waited
        if cached is not None:
            return cached, True
        try:
            result = run_analysis(symbol, years, start=start, end=end)
            _cache_put(key, result)
            return result, False
        finally:
            with _cache_lock:
                _inflight.pop(key, None)


@app.get("/api/health")
def health():
    return {"status": "ok"}


# ---------- chat assistant ----------

CHAT_MODEL = "claude-opus-4-8"

CHAT_SYSTEM = """You are the analysis assistant inside Ultrabull, a technical-analysis \
web app for NSE/BSE stocks. The user sees a dashboard with: a level-aware buy/sell \
synthesis, an empirical outlook (state-conditioned base rates), timing context (market/\
sector regime, relative strength, earnings cycle, breadth, distribution days), a price \
chart with support/resistance levels, backtest results, and pattern detections.

You receive the CURRENT analysis payload as JSON. Ground every answer in it — quote the \
actual numbers (prices, percentages, dates) rather than speaking generically. If the \
payload doesn't contain what's needed, say so plainly instead of guessing.

The analysis exists in TWO views. The top-level `advice`/`outlook`/`key_indicators`/\
`metrics`/`trades`/`signals`/`by_pattern` fields ARE the market-aware view (it is not \
repeated under `views`, to keep this payload small). `views.stock_only` holds the other \
one, and `active_view` says which the user is looking at. ANSWER FROM THAT ONE — when \
`active_view` is `stock_only`, ignore the top-level fields and quote `views.stock_only.*`, \
or you will contradict what is on the user's screen.

`market_aware` is the default and the supported one. `stock_only` removes the four \
market factors (NIFTY regime, sector regime, breadth, distribution days) from the advice \
SCORE, and lifts two actual backtest gates: the NIFTY regime requirement and the \
distribution-day ceiling. Be precise about that difference — the sector-regime gate and \
breadth floor were already off as gates in BOTH views, so do not tell the user they were \
switched off. It keeps relative strength, the earnings blackout and the liquidity floor.

If the user is on `stock_only`, answer their question there, but do not present its \
verdict as equally supported: it drops up to 4.0 points of drag against only 2.0 of \
lift, so it loosens fresh-buy calls AND softens holding-side exit calls. The view's own \
`warning` field carries the measured cost — use its numbers, don't invent your own.

Each advice object splits its score into `setup_score` (this stock) and `env_score` (the \
tape); when a setup is strong but the environment is vetoing it, say exactly that and \
quote `blockers[].clears_when` so the user knows what would change the answer. An empty \
`blockers` list means the tape is NOT what is holding the verdict back — do not invent \
one.

Project-validated knowledge you should apply (from this system's own 35-symbol, 4-year \
event studies):
- Validated edges: market-regime gate, relative-strength-vs-sector gate, earnings \
blackout (−3/+1 days), distribution-day ceiling (<5 in 25 sessions), Hammer (long) and \
BearishEngulfing (short) patterns. Breakouts have edge only when gated, over ~2+ weeks.
- Tested and rejected: RSI>50 as blanket confirmation (only the 50-65 band has mild \
edge; RSI>70 is NEGATIVE for fresh longs), 2x volume spikes as "institutional signals", \
blanket P/E<20 screens, ADX entry gates, composite-score ranking, VCP coil as an entry, \
PEAD drift on NSE large caps.
- Honest limits: no technical state predicted 1-week direction above ~58% in 4 years of \
data; day-ahead direction is a coin flip; edges are non-stationary (the walk-forward \
showed the best in-sample patterns degrading most recently).

Style — write for a smart person who is NOT a market professional:
- Plain English first. Any term of art (distribution day, base rate, relative strength, \
breadth, drawdown, reward:risk, divergence, t-stat) gets a one-line meaning in brackets \
the FIRST time you use it, e.g. "5 distribution days (days the index fell on higher \
volume than the day before — the footprint of large funds selling)".
- Say "54 of every 100 similar days closed higher" rather than "p_up 54%". Never expose \
raw field names or state keys (p_up, rs_20, mkt=dn) — translate them.
- Explain WHY each factor matters in one clause, not just its value: "lagging its \
sector by 10% (money is rotating into its peers, not this stock)".
- Match length to the question: quick question → one short paragraph; analysis request \
→ structure with headings and a table.
The user does their own fundamental research — your job is the technical/timing side. \
You are not a financial adviser and must not promise outcomes; frame everything as \
evidence and risk, and say "the data doesn't support that" when it doesn't. Refuse \
requests to place orders — this app analyzes, the user decides and executes.

You have two backtest tools over the project's 4y, 35-symbol NSE universe — USE them \
rather than speculating when the user proposes something testable:
- `test_rule` measures forward-return edge at 5/10/21 bars (t-stat, hit rate, excess bps). \
Reach for it on "does X predict anything?" / "is X bullish?" questions.
- `backtest_strategy` simulates actual trades with an entry rule AND an exit rule \
(stop/target in ATR multiples + max hold), returning win rate, avg win/loss, profit \
factor, expectancy, and exit mix. Reach for it on "what if I buy when X, stop at Y, \
target Z?" or when the user wants win-rate/profit-factor.
Report results honestly: sample size, and for test_rule treat |t| < 2 as inconclusive. \
ALWAYS compare the recent-12-months split against the full sample — an edge that vanished \
recently is not an edge. For backtest_strategy, remind that it measures edge (1-unit \
sizing, no fees/slippage), not rupee P&L, and that a high win rate with a tight target \
and wide stop can still lose money — read profit factor and expectancy together. Small n \
or thin coverage means say so."""


TEST_RULE_TOOL = {
    "name": "test_rule",
    "description": (
        "Backtest a trading idea against the 4-year, 35-symbol NSE universe using the "
        "project's event-study methodology. Express the idea as 1-6 numeric conditions "
        "that must ALL hold on a bar; the tool returns pooled forward-return excess "
        "(vs each symbol's own drift) at 5/10/21 bars with n, t-stat, hit rate, and a "
        "recent-12-months stability split. Call it whenever the user proposes a rule, "
        "heuristic, or 'does X work?' question that maps to these features. "
        "Feature units: percentages are in % (e.g. ret_20_pct > 10 means +10%); "
        "vol_mult20 is a multiple (2 = twice average volume); rsi/adx are 0-100; "
        "macd_above_signal and mkt_up are 0/1 flags; dist_days is a count; "
        "breadth_pct is 0-100."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Short label for the idea"},
            "direction": {"type": "string", "enum": ["long", "short"]},
            "conditions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 6,
                "items": {
                    "type": "object",
                    "properties": {
                        "feature": {"type": "string", "enum": IDEA_FEATURES},
                        "op": {"type": "string", "enum": ["<", "<=", ">", ">=", "=="]},
                        "value": {"type": "number"},
                    },
                    "required": ["feature", "op", "value"],
                },
            },
        },
        "required": ["direction", "conditions"],
    },
}


BACKTEST_TOOL = {
    "name": "backtest_strategy",
    "description": (
        "Trade-level backtest of a complete strategy (entry rule + exit rule) across the "
        "4-year, 35-symbol NSE universe. Unlike test_rule (which measures forward-return edge "
        "at fixed horizons), this simulates actual trades: enter at the next bar's open when "
        "ALL entry conditions hold, then exit at a stop (sl_atr x ATR against you), a target "
        "(tp_atr x ATR in your favour), or after max_hold bars — whichever comes first. Returns "
        "win rate, avg win/loss, profit factor, expectancy, exit mix, and a recent-12-months "
        "stability split. Use when the user asks 'what if I buy when X, stop at Y, target Z' or "
        "wants win-rate/profit-factor rather than a t-stat. Same feature units as test_rule. "
        "1-unit sizing, no fees/slippage — it measures edge, not rupee P&L; say so."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Short label for the strategy"},
            "direction": {"type": "string", "enum": ["long", "short"]},
            "conditions": {
                "type": "array", "minItems": 1, "maxItems": 6,
                "items": {
                    "type": "object",
                    "properties": {
                        "feature": {"type": "string", "enum": IDEA_FEATURES},
                        "op": {"type": "string", "enum": ["<", "<=", ">", ">=", "=="]},
                        "value": {"type": "number"},
                    },
                    "required": ["feature", "op", "value"],
                },
            },
            "sl_atr": {"type": "number", "description": "Stop distance in ATR multiples (e.g. 1.5). Default 1.5"},
            "tp_atr": {"type": "number", "description": "Target distance in ATR multiples (e.g. 3.0). Default 3.0"},
            "max_hold": {"type": "integer", "description": "Max bars to hold before timing out (1-60). Default 10"},
        },
        "required": ["direction", "conditions"],
    },
}


class ChatMessage(BaseModel):
    role: str
    content: str
    # optional pasted image on a user turn: base64 data URL ("data:image/png;base64,...")
    image: str | None = None


class ChatRequest(BaseModel):
    symbol: str
    years: float = 0.5
    start: str | None = None
    end: str | None = None
    messages: list[ChatMessage]
    # Which analysis view the user is looking at. Constrained so a client typo
    # 422s instead of silently degrading to "no view stated".
    view: Literal["market_aware", "stock_only"] = "market_aware"


def _chat_context(result: dict, view: str = "market_aware") -> str:
    """Compact analysis context: everything except the heavy per-bar series.

    Both views ship in the payload; `active_view` tells Claude which one the
    user is actually looking at so its answer matches their screen.
    """
    slim = {k: v for k, v in result.items() if k not in ("ohlcv", "indicators", "equity")}
    slim["detections"] = [d for d in slim.get("detections", []) if d.get("qualified")][-15:]
    slim["signals"] = slim.get("signals", [])[-15:]
    slim["trades"] = slim.get("trades", [])[-15:]
    slim["last_close"] = result["ohlcv"][-1]["close"] if result.get("ohlcv") else None
    slim.pop("summary_md", None)

    # `views` must be slimmed the same way the top level is, or every chat turn
    # re-sends the whole analysis twice. market_aware is dropped outright — it is
    # byte-for-byte the top-level fields — and the remaining view loses its
    # per-bar equity series and long tables. Left untouched this doubled the
    # context (+111%, ~25k uncached input tokens per message).
    views = result.get("views") or {}
    if views:
        slim["views"] = {
            name: {**{k: val for k, val in v.items() if k != "equity"},
                   "signals": (v.get("signals") or [])[-15:],
                   "trades": (v.get("trades") or [])[-15:]}
            for name, v in views.items() if name != "market_aware" and v
        }
        if views.get(view):
            slim["active_view"] = view
    return json.dumps(slim, default=str)


@app.post("/api/chat")
def chat(req: ChatRequest):
    if not req.messages or req.messages[-1].role != "user":
        raise HTTPException(status_code=422, detail="Last message must be from the user")
    symbol = req.symbol.strip().upper()
    try:
        result, _ = cached_analysis(symbol, req.years, req.start, req.end)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    client = anthropic.Anthropic()
    def _msg(m: ChatMessage) -> dict:
        # attach a pasted image as a vision block on a user turn
        if m.role == "user" and m.image and m.image.startswith("data:image/"):
            try:
                header, b64 = m.image.split(",", 1)
                media_type = header.split(";")[0].split(":")[1]
                if media_type in ("image/png", "image/jpeg", "image/gif", "image/webp") and len(b64) < 7_000_000:
                    return {"role": "user", "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                        {"type": "text", "text": m.content or "What do you make of this?"},
                    ]}
            except Exception:
                pass
        return {"role": m.role, "content": m.content}

    msgs = [_msg(m) for m in req.messages[-20:]]
    system_blocks = [
        {"type": "text", "text": CHAT_SYSTEM, "cache_control": {"type": "ephemeral"}},
        {"type": "text",
         "text": f"Current analysis payload for {symbol}:\n{_chat_context(result, req.view)}"},
    ]
    try:
        # short agentic loop: Claude may call test_rule to backtest user ideas
        for _ in range(4):
            response = client.messages.create(
                model=CHAT_MODEL,
                max_tokens=8192,
                system=system_blocks,
                tools=[TEST_RULE_TOOL, BACKTEST_TOOL],
                messages=msgs,
            )
            if response.stop_reason != "tool_use":
                break
            msgs.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                try:
                    if block.name == "backtest_strategy":
                        res = backtest_strategy(
                            conditions=block.input["conditions"],
                            direction=block.input.get("direction", "long"),
                            sl_atr=block.input.get("sl_atr", 1.5),
                            tp_atr=block.input.get("tp_atr", 3.0),
                            max_hold=block.input.get("max_hold", 10),
                            name=block.input.get("name", "strategy"),
                            cache_dir=IDEA_CACHE_DIR,
                        )
                    else:
                        res = evaluate_rule(
                            conditions=block.input["conditions"],
                            direction=block.input.get("direction", "long"),
                            name=block.input.get("name", "idea"),
                            cache_dir=IDEA_CACHE_DIR,
                        )
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                         "content": json.dumps(res, default=str)})
                except Exception as e:
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                         "content": f"{block.name} failed: {e}", "is_error": True})
            msgs.append({"role": "user", "content": tool_results})
    except (anthropic.AuthenticationError, TypeError):
        # TypeError = SDK found no credential source at all (raised at request time)
        raise HTTPException(
            status_code=503,
            detail="Chat needs an Anthropic API key: set ANTHROPIC_API_KEY in the "
                   "server environment (or run `ant auth login`) and restart the API.",
        )
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limited by the Claude API — try again shortly.")
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Claude API error ({e.status_code}): {e.message}")
    except anthropic.APIConnectionError:
        raise HTTPException(status_code=503, detail="Could not reach the Claude API — check network.")

    reply = "".join(b.text for b in response.content if b.type == "text")
    return {"reply": reply, "model": response.model}


class IdeaCondition(BaseModel):
    feature: str
    op: str
    value: float


class IdeaRequest(BaseModel):
    direction: str = "long"
    name: str = "idea"
    conditions: list[IdeaCondition]


@app.post("/api/idea")
def idea(req: IdeaRequest):
    """Direct rule-testing endpoint (same engine the chat tool uses)."""
    try:
        return evaluate_rule(
            conditions=[c.model_dump() for c in req.conditions],
            direction=req.direction, name=req.name, cache_dir=IDEA_CACHE_DIR,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


def _view_digest(result: dict, view: str) -> dict:
    """The market-context-sensitive slice of a view (score/verdicts/factors/
    key indicators) — everything else in the compact signal (plan, levels,
    timing) is level-based and identical across views."""
    v = ((result.get("views") or {}).get(view)) or {}
    advice = v.get("advice") or (result.get("advice") if view == "market_aware" else None) or {}
    outlook = v.get("outlook") or (result.get("outlook") if view == "market_aware" else None) or {}
    h1w = ((outlook or {}).get("horizons") or {}).get("1w") or {}
    factors = [f for f in (advice.get("factors") or []) if abs(f.get("score") or 0) > 0]
    factors.sort(key=lambda f: -abs(f.get("score") or 0))
    return {
        "score": _num(advice.get("score")),
        "fresh": advice.get("fresh"),
        "holding": advice.get("holding"),
        "factors": factors[:4],
        "market_context_applied": advice.get("market_context_applied"),
        "outlook_1w": {"p_up": _num(h1w.get("p_up")), "verdict": h1w.get("verdict"),
                       "n": h1w.get("n"), "fallback": h1w.get("fallback")},
        "key_indicators": v.get("key_indicators") or (result.get("key_indicators") if view == "market_aware" else None),
    }


def _compact_signal(result: dict) -> dict:
    """Row-sized digest of a full analysis for the holdings table: verdicts,
    plan (stop/target), nearest support/resistance with distance, breakout
    state, and the 1-week base rate. `views` carries the market-aware and
    stock-only verdicts side by side (mirroring /api/analyze's views_json) so
    the holdings table can toggle without a second fetch or an inconsistent
    snapshot store."""
    advice = result.get("advice") or {}
    plan = advice.get("plan") or {}
    close = advice.get("close")
    if close is None and result.get("ohlcv"):
        close = result["ohlcv"][-1]["close"]

    levels = result.get("levels") or []
    sup = res = None
    if close is not None:
        below = [lv for lv in levels if lv["kind"] == "support" and lv["price"] <= close]
        above = [lv for lv in levels if lv["kind"] == "resistance" and lv["price"] >= close]
        sup = max(below, key=lambda lv: lv["price"], default=None)
        res = min(above, key=lambda lv: lv["price"], default=None)

    def _level(lv):
        if lv is None:
            return None
        return {"price": _num(lv["price"]), "distance_pct": _num(lv.get("distance_pct")),
                "touches": lv.get("touches"), "strength": _num(lv.get("strength"))}

    # most recent qualified breakout-family detection within ~3 weeks of the last bar
    recent_breakout = None
    try:
        last_d = _date_cls.fromisoformat(result["last_date"])
        for d in reversed(result.get("detections") or []):
            if not d.get("qualified"):
                continue
            if "breakout" not in str(d.get("pattern", "")).lower():
                continue
            if (last_d - _date_cls.fromisoformat(d["date"])).days <= 21:
                recent_breakout = {"pattern": d["pattern"], "date": d["date"],
                                   "english": d.get("english")}
            break
    except (ValueError, KeyError):
        pass

    views = {name: _view_digest(result, name) for name in (result.get("views") or {"market_aware": None}).keys()}
    if not views:
        views = {"market_aware": _view_digest(result, "market_aware")}
    default = views.get("market_aware") or next(iter(views.values()))

    return {
        "symbol": result["symbol"],
        "last_date": result.get("last_date"),
        "close": _num(close),
        "views": views,
        # kept for back-compat with any caller still reading top-level
        # verdict fields — always the market-aware view, same as before.
        "score": default.get("score"),
        "fresh": default.get("fresh"),
        "holding": default.get("holding"),
        "stop": _num(plan.get("stop")),
        "stop_basis": plan.get("stop_basis"),
        "target": _num(plan.get("target1")),
        "target2": _num(plan.get("target2")),
        "reward_risk": _num(plan.get("reward_risk")),
        "if_breakout": (lambda ib: None if not ib else {
            "trigger": _num(ib.get("trigger")),
            "target": _num(ib.get("target")),
            "target_basis": ib.get("target_basis"),
            "stop_after": _num(ib.get("stop_after")),
            "note": ib.get("note"),
        })(plan.get("if_breakout")),
        "entry_zone": [_num(plan.get("entry_zone_low")), _num(plan.get("entry_zone_high"))],
        "support": _level(sup),
        "resistance": _level(res),
        "breakout": {
            "above_all_resistance": bool(close is not None and res is None and levels),
            "recent_breakout": recent_breakout,
        },
        "factors": default.get("factors") or [],
        "outlook_1w": default.get("outlook_1w"),
        "key_indicators": default.get("key_indicators"),
        "in_earnings_blackout": bool((result.get("timing") or {}).get("in_earnings_blackout")),
        "next_earnings": (result.get("timing") or {}).get("next_earnings"),
        "expected_results_window": (result.get("timing") or {}).get("expected_results_window"),
        "results_overdue": bool((result.get("timing") or {}).get("results_overdue")),
        "market_regime": (result.get("context") or {}).get("market_regime"),
    }


# ---------- signal snapshot store (SQLite, survives restarts) ----------
# The TTL cache above holds full payloads for minutes; this holds the compact
# per-symbol digest as-of its analysis date, so the holdings table renders the
# last known verdicts instantly and only recomputes when the user asks.

SIGNAL_DB = os.path.join(ROOT, "data_cache", "signal_snapshots.db")


def _signal_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(SIGNAL_DB), exist_ok=True)
    conn = sqlite3.connect(SIGNAL_DB, timeout=10.0)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS signal_snapshots ("
        "  symbol TEXT PRIMARY KEY,"
        "  stored_at TEXT NOT NULL,"
        "  payload TEXT NOT NULL)"
    )
    return conn


def _snapshot_put(symbol: str, snap: dict) -> str:
    stored_at = pd.Timestamp.now().isoformat(timespec="seconds")
    conn = _signal_db()
    try:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO signal_snapshots (symbol, stored_at, payload) VALUES (?, ?, ?)",
                (symbol, stored_at, json.dumps(snap)),
            )
    finally:
        conn.close()
    return stored_at


@app.get("/api/signal/{symbol}")
def signal(
    symbol: str,
    years: float = Query(1.0, gt=0.0, le=10.0),
    start: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    end: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    force: bool = Query(False, description="Skip the TTL cache — guaranteed fresh recompute"),
):
    """Compute (TTL-cached unless force=1) the analysis for one symbol, persist
    the compact digest to the snapshot store, and return it."""
    symbol = symbol.strip().upper()
    if not symbol:
        raise HTTPException(status_code=422, detail="Empty symbol")
    if start and end and start >= end:
        raise HTTPException(status_code=422, detail="start must be before end")
    try:
        result, was_cached = cached_analysis(symbol, years, start, end, force=force)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    snap = _compact_signal(result)
    snap["range"] = {"years": None if start else years, "start": start, "end": end}
    stored_at = _snapshot_put(symbol, snap)
    return {**snap, "stored_at": stored_at, "cached": was_cached}


# ---------- holding buy-dates + XIRR (brokers don't expose purchase dates) ----------

HOLD_DATES_DB = os.path.join(ROOT, "data_cache", "hold_dates.db")


def _hold_dates_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(HOLD_DATES_DB), exist_ok=True)
    conn = sqlite3.connect(HOLD_DATES_DB, timeout=10.0)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS hold_dates ("
        "  username TEXT NOT NULL,"
        "  key TEXT NOT NULL,"           # broker:symbol
        "  buy_date TEXT NOT NULL,"      # YYYY-MM-DD
        "  PRIMARY KEY (username, key))"
    )
    return conn


def _xirr(flows: list[tuple[_date_cls, float]], guess: float = 0.15) -> float | None:
    """Annualized IRR of dated cashflows (Newton's method, day-count basis).
    flows: [(date, amount)] with outflows negative, inflows positive."""
    if len(flows) < 2:
        return None
    t0 = min(f[0] for f in flows)
    years = [(d - t0).days / 365.0 for d, _ in flows]
    amts = [a for _, a in flows]
    if not (any(a < 0 for a in amts) and any(a > 0 for a in amts)):
        return None
    rate = guess
    for _ in range(100):
        f = sum(a / (1 + rate) ** y for a, y in zip(amts, years))
        df = sum(-y * a / (1 + rate) ** (y + 1) for a, y in zip(amts, years))
        if abs(df) < 1e-9:
            break
        new = rate - f / df
        if new <= -0.999:
            new = (rate - 0.999) / 2
        if abs(new - rate) < 1e-7:
            return new
        rate = new
    return rate if -0.999 < rate < 100 else None


class HoldDate(BaseModel):
    key: str  # "broker:SYMBOL"
    buy_date: str  # YYYY-MM-DD


@app.get("/api/holddates")
def get_hold_dates(request: Request):
    user = getattr(request.state, "user", None) or "local"
    conn = _hold_dates_db()
    try:
        rows = conn.execute("SELECT key, buy_date FROM hold_dates WHERE username = ?", (user,)).fetchall()
    finally:
        conn.close()
    return {"dates": {k: v for k, v in rows}}


@app.put("/api/holddates")
def put_hold_date(req: HoldDate, request: Request):
    user = getattr(request.state, "user", None) or "local"
    try:
        _date_cls.fromisoformat(req.buy_date)
    except ValueError:
        raise HTTPException(status_code=422, detail="buy_date must be YYYY-MM-DD")
    conn = _hold_dates_db()
    try:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO hold_dates (username, key, buy_date) VALUES (?, ?, ?)",
                (user, req.key.strip(), req.buy_date),
            )
    finally:
        conn.close()
    return {"ok": True}


class XirrPosition(BaseModel):
    key: str
    yahoo_symbol: str | None = None
    invested: float
    current_value: float
    ltp: float | None = None
    buy_date: str | None = None


class XirrRequest(BaseModel):
    positions: list[XirrPosition]


@app.post("/api/xirr")
def xirr(req: XirrRequest, request: Request):
    """Portfolio XIRR. Exact per-symbol cashflows come from the trade ledger
    when it has that symbol's history (each BUY/SELL a dated flow); otherwise
    it falls back to the manually-set single buy date. Positions with neither
    are excluded from XIRR but still counted in the absolute totals."""
    from .ledger import ledger_flows

    user = getattr(request.state, "user", None) or "local"
    conn = _hold_dates_db()
    try:
        stored = dict(conn.execute("SELECT key, buy_date FROM hold_dates WHERE username = ?", (user,)).fetchall())
    finally:
        conn.close()

    wanted = {p.yahoo_symbol for p in req.positions if p.yahoo_symbol}
    led = ledger_flows(wanted) if wanted else {}

    today = _date_cls.today()
    flows: list[tuple[_date_cls, float]] = []
    per_key: dict[str, dict] = {}
    dated_invested = dated_value = 0.0

    for p in req.positions:
        entry = {"annualized_pct": None, "buy_date": None, "days_held": None, "source": None}
        sym_flows = led.get(p.yahoo_symbol) if p.yahoo_symbol else None

        if sym_flows:
            # exact: replay every ledger fill, terminal value = current holding
            pos_flows: list[tuple[_date_cls, float]] = []
            for dt, signed_qty, price in sym_flows:
                try:
                    d = _date_cls.fromisoformat(dt)
                except ValueError:
                    continue
                pos_flows.append((d, -signed_qty * price))  # BUY -> outflow, SELL -> inflow
            term = p.current_value
            if pos_flows and term is not None:
                pos_flows.append((today, float(term)))
                r = _xirr(pos_flows)
                entry.update({
                    "source": "ledger",
                    "annualized_pct": round(r * 100, 2) if r is not None else None,
                    "buy_date": min(f[0] for f in pos_flows).isoformat(),
                })
                flows.extend((d, a) for d, a in pos_flows[:-1])  # merge fills into portfolio
                dated_invested += p.invested
                dated_value += term
                per_key[p.key] = entry
                continue

        bd = p.buy_date or stored.get(p.key)
        entry["buy_date"] = bd
        if bd:
            try:
                d = _date_cls.fromisoformat(bd)
            except ValueError:
                d = None
            if d and d < today and p.invested > 0:
                days = (today - d).days
                entry["days_held"] = days
                entry["source"] = "manual"
                if days >= 1:
                    yrs = days / 365.0
                    entry["annualized_pct"] = round(((p.current_value / p.invested) ** (1 / yrs) - 1) * 100, 2)
                flows.append((d, -float(p.invested)))
                dated_invested += p.invested
                dated_value += p.current_value
        per_key[p.key] = entry

    portfolio_xirr = None
    if flows:
        flows.append((today, float(dated_value)))
        r = _xirr(flows)
        portfolio_xirr = round(r * 100, 2) if r is not None else None

    return {
        "portfolio_xirr_pct": portfolio_xirr,
        "dated_invested": round(dated_invested, 2),
        "dated_current": round(dated_value, 2),
        "positions": per_key,
    }


@app.post("/api/ledger/poll")
def ledger_poll():
    """Manually trigger a Groww order poll (the systemd timer runs this daily)."""
    from .ledger import poll_groww
    return poll_groww()


# ---------- validation audit (health loop — served read-only; timer recomputes) ----------

@app.get("/api/audit")
def audit_latest():
    """Most recent validation audit (calibration + pattern/gate decay + coverage)."""
    from audit import latest_audit
    rep = latest_audit()
    if rep is None:
        return {"available": False,
                "detail": "No audit yet — the weekly timer or POST /api/audit/run produces one."}
    return {"available": True, "report": rep}


@app.get("/api/audit/history")
def audit_trend(limit: int = Query(26, ge=1, le=104)):
    from audit import audit_history
    return {"history": audit_history(limit)}


def _held_symbols() -> list[str]:
    """Yahoo symbols the user actually holds (Groww), for the coverage check."""
    syms: set[str] = set()
    try:
        from .live import groww_holdings
        for r in groww_holdings().get("holdings", []):
            if r.get("yahoo_symbol"):
                syms.add(r["yahoo_symbol"])
    except Exception:
        pass
    return sorted(syms)


@app.post("/api/audit/run")
def audit_run():
    """Recompute the audit now (the systemd timer calls the module directly)."""
    from audit import run_audit, save_audit
    res = run_audit(cache_dir=IDEA_CACHE_DIR, held_symbols=_held_symbols())
    if not res.get("ok"):
        raise HTTPException(status_code=503, detail=res.get("error", "audit failed"))
    stamp = pd.Timestamp.now().isoformat(timespec="seconds")
    save_audit(res["report"], stamp)
    return {"ok": True, "generated_at": stamp,
            "decayed": [p["pattern"] for p in res["report"]["pattern_decay"]["patterns"]
                        if p["status"] in ("decayed", "negative")]}


@app.post("/api/calibrate/run")
def calibrate_run():
    """Rebuild the base rates on current data and refit the outlook calibration.
    Invalidates the analysis cache so calibrated verdicts show immediately."""
    from calibrate import rebuild_and_calibrate
    res = rebuild_and_calibrate(IDEA_CACHE_DIR)
    with _cache_lock:
        _cache.clear()
    return {"ok": True, "states": res["states"], "bars": res["bars"],
            "horizons": {h: d.get("k") for h, d in res["calibration"]["horizons"].items()}}


# ---------- steward (Phase 1.5 — Claude reads the audit, proposes, human approves) ----------

class OverrideRequest(BaseModel):
    pattern: str
    disabled: bool


@app.get("/api/steward")
def steward_latest():
    from .steward import latest_steward, load_overrides
    return {"assessment": latest_steward(), "overrides": load_overrides()["disabled"]}


@app.post("/api/steward/run")
def steward_run():
    from .steward import run_steward
    res = run_steward()
    if not res.get("ok"):
        raise HTTPException(status_code=503, detail=res.get("error", "steward run failed"))
    return res


@app.post("/api/steward/apply")
def steward_apply(req: OverrideRequest):
    """Human approves (or reverts) a steward proposal. Only additive disables /
    reversals of steward-disabled patterns — base prunes stay untouchable here.
    Invalidates the analysis cache so the change takes effect immediately."""
    from .steward import set_override
    ov = set_override(req.pattern.strip(), req.disabled)
    with _cache_lock:
        _cache.clear()  # verdicts must reflect the new pattern set
    return {"ok": True, "disabled": ov["disabled"]}


@app.get("/api/signals")
def signals(symbols: str = Query(..., description="Comma-separated Yahoo symbols")):
    """Read-only batch fetch of stored snapshots — never computes anything.
    Missing symbols simply aren't in the response."""
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not syms:
        return {"signals": {}}
    conn = _signal_db()
    try:
        placeholders = ",".join("?" * len(syms))
        rows = conn.execute(
            f"SELECT symbol, stored_at, payload FROM signal_snapshots WHERE symbol IN ({placeholders})",
            syms,
        ).fetchall()
    finally:
        conn.close()
    return {"signals": {sym: {**json.loads(payload), "stored_at": stored_at}
                        for sym, stored_at, payload in rows}}


# ---------- Claude holding takes (on-demand, persisted) ----------
# Two tiers: "quick" reads the analysis payload + position only; "deep" also
# enables the server-side web search tool (analyst estimates, announced result
# dates, corporate actions). Both persist to SQLite like signal snapshots.

TAKE_INSTRUCTIONS = (
    "The user holds a position in this stock and wants your read on it. Answer in "
    "under ~300 words of markdown with three short sections: (1) **Verdict** for "
    "their existing position (hold / trim / exit / add, with conviction), (2) **Why** "
    "— the strongest evidence for and against, grounded in the payload numbers, "
    "(3) **What would change my mind** — specific price levels or upcoming dates to "
    "watch. Weigh their entry price only for tax/psychology framing; levels come "
    "from the chart, not their cost basis."
)

DEEP_INSTRUCTIONS = (
    "\n\nAdditionally, use web search before concluding: find (a) the announced or "
    "expected date of the company's next quarterly results (BSE/NSE announcements), "
    "(b) analyst estimates, price targets, or consensus views if any exist for this "
    "company, and (c) recent material news, corporate actions, AGM notices, or "
    "shareholding/pledging developments. Cite each finding with its source and date. "
    "Reconcile the research with the technical picture — say explicitly when they "
    "disagree. If the company is too small for analyst coverage, say so plainly."
)

WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search", "max_uses": 8}

TAKES_DB = os.path.join(ROOT, "data_cache", "claude_takes.db")


def _takes_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(TAKES_DB), exist_ok=True)
    conn = sqlite3.connect(TAKES_DB, timeout=10.0)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS claude_takes ("
        "  symbol TEXT NOT NULL,"
        "  tier TEXT NOT NULL,"
        "  view TEXT NOT NULL DEFAULT 'market_aware',"
        "  stored_at TEXT NOT NULL,"
        "  payload TEXT NOT NULL,"
        "  PRIMARY KEY (symbol, tier, view))"
    )
    # Migrate pre-view rows: the old PK was (symbol, tier), so a stock-only take
    # would have silently overwritten the market-aware one. Existing takes were
    # all market-aware — label them as such rather than dropping them.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(claude_takes)")}
    if "view" not in cols:
        with conn:
            conn.execute("ALTER TABLE claude_takes RENAME TO claude_takes_old")
            conn.execute(
                "CREATE TABLE claude_takes ("
                "  symbol TEXT NOT NULL,"
                "  tier TEXT NOT NULL,"
                "  view TEXT NOT NULL DEFAULT 'market_aware',"
                "  stored_at TEXT NOT NULL,"
                "  payload TEXT NOT NULL,"
                "  PRIMARY KEY (symbol, tier, view))"
            )
            conn.execute(
                "INSERT INTO claude_takes (symbol, tier, view, stored_at, payload) "
                "SELECT symbol, tier, 'market_aware', stored_at, payload FROM claude_takes_old"
            )
            conn.execute("DROP TABLE claude_takes_old")
    return conn


class TakeRequest(BaseModel):
    symbol: str
    quantity: float | None = None
    avg_price: float | None = None
    deep: bool = False
    years: float = 1.0
    # Which analysis view the user is looking at, so the take reasons from the
    # same numbers as their screen (same contract as ChatRequest).
    view: Literal["market_aware", "stock_only"] = "market_aware"


STOCK_ONLY_TAKE_NOTE = (
    "\n\nIMPORTANT — the user is viewing the STOCK-ONLY view: the NIFTY market-regime gate "
    "and the distribution-day ceiling are lifted, and the four market factors (market regime, "
    "sector regime, breadth, distribution days) are excluded from the advice score. Read "
    "`views.stock_only` in the payload, NOT the market-aware top-level fields, and base your "
    "verdict on it. Say plainly that this view ignores validated market filters — it removes "
    "up to 4.0 points of drag against only 2.0 of lift, so it is more permissive on fresh buys "
    "AND softer on exits. Relative strength vs sector, the earnings blackout and the liquidity "
    "floor still apply. If the market-aware view would reach a different verdict, say so and "
    "name the gap."
)


@app.post("/api/claude/take")
def claude_take(req: TakeRequest):
    symbol = req.symbol.strip().upper()
    if not symbol:
        raise HTTPException(status_code=422, detail="Empty symbol")
    try:
        result, _ = cached_analysis(symbol, req.years)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    position = ""
    if req.quantity is not None and req.avg_price is not None:
        position = (f"I currently hold {req.quantity:g} shares of {symbol} at an "
                    f"average buy price of ₹{req.avg_price:.2f}. ")
    prompt = (position + "Give me your read on this position."
              if position else f"Assess {symbol} as a holding right now.")

    system_blocks = [
        {"type": "text", "text": CHAT_SYSTEM, "cache_control": {"type": "ephemeral"}},
        {"type": "text",
         "text": f"Current analysis payload for {symbol}:\n{_chat_context(result, req.view)}"},
        {"type": "text",
         "text": TAKE_INSTRUCTIONS
                 + (DEEP_INSTRUCTIONS if req.deep else "")
                 + (STOCK_ONLY_TAKE_NOTE if req.view == "stock_only" else "")},
    ]

    client = anthropic.Anthropic()
    msgs = [{"role": "user", "content": prompt}]
    try:
        # server-side web search can pause the turn; re-send to resume
        for _ in range(6):
            response = client.messages.create(
                model=CHAT_MODEL,
                max_tokens=8192 if req.deep else 4096,
                thinking={"type": "adaptive"},
                system=system_blocks,
                tools=[WEB_SEARCH_TOOL] if req.deep else [],
                messages=msgs,
            )
            if response.stop_reason != "pause_turn":
                break
            msgs = [msgs[0], {"role": "assistant", "content": response.content}]
    except (anthropic.AuthenticationError, TypeError):
        raise HTTPException(
            status_code=503,
            detail="Claude take needs an Anthropic API key: set ANTHROPIC_API_KEY in "
                   "the server environment and restart the API.",
        )
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limited by the Claude API — try again shortly.")
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Claude API error ({e.status_code}): {e.message}")
    except anthropic.APIConnectionError:
        raise HTTPException(status_code=503, detail="Could not reach the Claude API — check network.")

    text = "".join(b.text for b in response.content if b.type == "text")
    take = {
        "symbol": symbol,
        "tier": "deep" if req.deep else "quick",
        "view": req.view,
        "text": text,
        "model": response.model,
        "position": ({"quantity": req.quantity, "avg_price": req.avg_price}
                     if position else None),
        "as_of": result.get("last_date"),
    }
    stored_at = pd.Timestamp.now().isoformat(timespec="seconds")
    conn = _takes_db()
    try:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO claude_takes (symbol, tier, view, stored_at, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (symbol, take["tier"], req.view, stored_at, json.dumps(take)),
            )
    finally:
        conn.close()
    return {**take, "stored_at": stored_at}


@app.get("/api/claude/takes")
def claude_takes(symbols: str = Query(..., description="Comma-separated Yahoo symbols")):
    """Read-only batch fetch of stored takes — never calls Claude."""
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not syms:
        return {"takes": {}}
    conn = _takes_db()
    try:
        placeholders = ",".join("?" * len(syms))
        rows = conn.execute(
            f"SELECT symbol, tier, view, stored_at, payload FROM claude_takes WHERE symbol IN ({placeholders})",
            syms,
        ).fetchall()
    finally:
        conn.close()
    # symbol -> view -> tier, so the UI shows the take for the view on screen
    # rather than one written against the other view's numbers.
    out: dict[str, dict] = {}
    for sym, tier, view, stored_at, payload in rows:
        out.setdefault(sym, {}).setdefault(view, {})[tier] = {
            **json.loads(payload), "stored_at": stored_at,
        }
    return {"takes": out}


# ---------------------------------------------------------------------------
# Momentum strategy: cross-sectional screener (Phase 9 API integration).
#
# Unlike /api/analyze/{symbol}'s per-symbol "momentum" diagnostics above,
# ranking requires comparing every candidate against the whole universe on
# the SAME date -- that's what core/strategies/momentum_strategy.py's
# cross_sectional_rank/select_candidates do, reused here unmodified. This
# endpoint is a LIVE, today-anchored ranking, not a backtest; the backtested
# track record lives in core/momentum_experiment_results.csv (see
# core/research_momentum_experiment.py) and is not reproduced here.
# ---------------------------------------------------------------------------
from strategies.momentum_strategy import (  # noqa: E402
    MomentumConfig as _MomentumConfig,
    screen_universe_now as _mom_screen_universe_now,
)
from event_study import UNIVERSE as _MOMENTUM_UNIVERSE  # noqa: E402

_MOMENTUM_CACHE_DIR = os.path.join(ROOT, "data_cache", "momentum")
_momentum_screen_cache: dict[tuple, tuple[float, dict]] = {}
_MOMENTUM_SCREEN_TTL_SECONDS = 30 * 60  # a 3-12 month momentum figure barely moves within a session


@app.get("/api/momentum/screen")
def momentum_screen(
    lookback: str = Query("6M", description="One of: " + ", ".join(MOMENTUM_LOOKBACK_BARS)),
    top_n: int = Query(10, ge=1, le=50),
    years: float = Query(2.0, ge=0.5, le=6.0),
    use_relative: bool = Query(True, description="Rank by relative momentum vs NIFTY (else absolute momentum)"),
    use_technical: bool = Query(True, description="Require MA20/MA50 trend confirmation"),
    use_market_regime: bool = Query(True, description="Require NIFTY in an MA20>MA50 uptrend"),
    symbols: str | None = Query(None, description="Comma-separated override of the default universe"),
):
    if lookback not in MOMENTUM_LOOKBACK_BARS:
        raise HTTPException(400, f"lookback must be one of {list(MOMENTUM_LOOKBACK_BARS)}")
    universe = [s.strip().upper() for s in symbols.split(",") if s.strip()] if symbols else _MOMENTUM_UNIVERSE

    cache_key = (lookback, top_n, years, use_relative, use_technical, use_market_regime, tuple(sorted(universe)))
    now = time.time()
    cached = _momentum_screen_cache.get(cache_key)
    if cached and (now - cached[0]) < _MOMENTUM_SCREEN_TTL_SECONDS:
        return {**cached[1], "cached": True}

    cfg = _MomentumConfig(
        lookback_label=lookback, top_n=top_n,
        use_relative_momentum=use_relative,
        use_technical_confirmation=use_technical,
        use_market_regime=use_market_regime,
    )
    screen = _mom_screen_universe_now(universe, years, cfg, cache_dir=_MOMENTUM_CACHE_DIR)
    if screen["n_symbols"] == 0:
        raise HTTPException(502, "Could not load data for any symbol in the requested universe")
    if screen["as_of"] is None:
        raise HTTPException(502, "No valid momentum data for the requested universe/lookback")

    rows = [{
        **row,
        "absolute_momentum": _num(row["absolute_momentum"]),
        "relative_momentum": _num(row["relative_momentum"]),
        "close": _num(row["close"]),
        "entry_est": _num(row["entry_est"]),
        "sl_est": _num(row["sl_est"]),
        "tp_est": _num(row["tp_est"]),
    } for row in screen["rows"]]

    result = {
        "lookback": lookback,
        "lookback_bars": cfg.lookback_bars,
        "as_of": _date(screen["as_of"]),
        "n_symbols": screen["n_symbols"],
        "config": {
            "top_n": top_n, "use_relative_momentum": use_relative,
            "use_technical_confirmation": use_technical, "use_market_regime": use_market_regime,
        },
        "ranked": rows,
        "note": (
            "Live ranking, not a backtest. Not validated as profitable -- see "
            "core/momentum_experiment_results.csv for backtested evidence at "
            "this lookback before treating any row as a recommendation."
        ),
    }
    _momentum_screen_cache[cache_key] = (now, result)
    return {**result, "cached": False}


INTRADAY_INTERVALS = {"5m", "15m", "30m", "60m", "90m"}


def run_intraday(symbol: str, interval: str, years: float = 1.0,
                 start: str | None = None, end: str | None = None) -> dict:
    """Lighter, timeframe-honest pipeline for intraday bars: price + indicators
    + S/R levels + raw pattern sightings, over the requested window (capped to
    what the data source serves for the interval). The daily-only machinery
    (base-rate outlook, calibration, backtest, validated-edge verdicts) is
    deliberately NULL — it was fit on daily bars and does not transfer."""
    from stock_explorer import _INTRADAY_MAX_DAYS
    cap = _INTRADAY_MAX_DAYS.get(interval, 60)
    req_days = (pd.Timestamp(end) - pd.Timestamp(start)).days if start else int(round(365 * years))
    cap_note = (f"{interval} history is limited to ~{cap} days by the data source; "
                f"showing the most recent {cap} days." if req_days > cap else None)

    df = load_data(symbol, years=years, start=start, end=end, interval=interval)
    ind = compute_indicators(df)
    df_ohlcv = extract_symbol_df(df, symbol=symbol, case="title")
    ind_df = build_ind_df_from_vbt(df_ohlcv, ind)
    tf = _ts_intraday

    empty_dets = pd.DataFrame(columns=["date", "pattern", "score", "strength", "regime", "notes", "params"])
    try:
        detections = run_all_patterns(df_ohlcv, ind_df, CFG)
    except ValueError:
        detections = empty_dets
    if detections is None or detections.empty:
        det_enriched = empty_dets.assign(direction=None, alignment=None, qualified=False)
    else:
        det_enriched = enrich_detections(detections, df_ohlcv, ind_df, symbol, ctx=None)
        det_enriched = det_enriched[det_enriched["date"] >= warmup_cutoff(ind_df)]

    detections_json = []
    for r in det_enriched.itertuples(index=False):
        detections_json.append({
            "date": tf(getattr(r, "date")),
            "pattern": getattr(r, "pattern"),
            "score": _num(getattr(r, "score", None)),
            "strength": getattr(r, "strength", None),
            "direction": getattr(r, "direction", None),
            "alignment": getattr(r, "alignment", None),
            "qualified": bool(getattr(r, "qualified", False)),
            "english": getattr(r, "english", None),
            "side": None, "ref_close": None, "sl_est": None, "tp_est": None,
        })

    try:
        look = min(len(df_ohlcv), 400)
        raw_levels = find_levels(df_ohlcv.iloc[-look:], ind_df.iloc[-look:])
    except Exception:
        raw_levels = []
    levels_json = [{"price": lv["price"], "kind": lv["kind"], "touches": lv["touches"],
                    "strength": _num(lv["strength"]), "last_touch": _date(lv["last_touch"]),
                    "source": lv["source"], "distance_pct": _num(lv["distance_pct"])} for lv in raw_levels]

    macd_hist = ind_df["MACD"] - ind_df["MACD_signal"]
    return {
        "symbol": symbol, "interval": interval, "intraday": True,
        "cap_note": cap_note,
        "years": years, "bars": len(df_ohlcv),
        "last_date": pd.Timestamp(df_ohlcv.index[-1]).isoformat(),
        "ohlcv": _ohlcv_rows_t(df_ohlcv, tf),
        "indicators": {
            "ma20": _series_points_t(ind_df["MA20"], tf), "ma50": _series_points_t(ind_df["MA50"], tf),
            "bb_upper": _series_points_t(ind_df["BB_upper"], tf), "bb_lower": _series_points_t(ind_df["BB_lower"], tf),
            "rsi": _series_points_t(ind_df["RSI"], tf), "atr": _series_points_t(ind_df["ATR"], tf),
            "macd": _series_points_t(ind_df["MACD"], tf), "macd_signal": _series_points_t(ind_df["MACD_signal"], tf),
            "macd_hist": _series_points_t(macd_hist, tf),
            "adx": _series_points_t(ind_df["ADX"], tf) if "ADX" in ind_df.columns else [],
        },
        "levels": levels_json,
        "detections": detections_json,
        # daily-only — intentionally absent on intraday
        "events": [], "context": None, "timing": None, "outlook": None, "advice": None,
        "views": None,
        "key_indicators": None, "fundamentals": None, "interpretations": None, "indexes": [],
        "signals": [], "trades": [], "equity": [], "metrics": {}, "by_pattern": [],
        "audit": {"checked": 0, "mismatch_count": 0, "ok": True}, "signal_summary": {},
        "summary_md": "",
    }


@app.get("/api/analyze/{symbol}")
def analyze(
    symbol: str,
    years: float = Query(0.5, gt=0.0, le=10.0),
    start: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    end: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    interval: str = Query("1d"),
):
    symbol = symbol.strip().upper()
    if not symbol:
        raise HTTPException(status_code=422, detail="Empty symbol")
    if interval != "1d" and interval not in INTRADAY_INTERVALS:
        raise HTTPException(status_code=422, detail=f"interval must be 1d or one of {sorted(INTRADAY_INTERVALS)}")
    if start and end and start >= end:
        raise HTTPException(status_code=422, detail="start must be before end")
    if interval != "1d":
        try:
            return {**run_intraday(symbol, interval, years=years, start=start, end=end), "cached": False}
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    try:
        result, was_cached = cached_analysis(symbol, years, start, end)
    except ValueError as e:
        # load_data raises ValueError("No data for <symbol>") on unknown tickers
        raise HTTPException(status_code=404, detail=str(e))
    return {**result, "cached": was_cached}
