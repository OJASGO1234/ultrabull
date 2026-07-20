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
from patterns.config import CFG
from patterns.pipeline import run_all_patterns
from patterns.postprocess import enrich_detections, warmup_cutoff, summarize_signals
from patterns.explain import daily_summary_md
from rules.gates import apply_gates
from rules.signals import compile_signals
from sim.engine import simulate
from sim.metrics import summarize, by_tag, audit_exits
from pipeline import cfg as PIPELINE_BASE_CFG

from .sre import router as sre_router
from .auth import make_token, verify_token, verify_user
from .live import router as live_router, groww_router

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


# ---------- JSON-safe serialization helpers ----------

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

    detections_json = []
    for r in det_enriched.itertuples(index=False):
        plan = _plan(getattr(r, "pattern"), getattr(r, "direction", None),
                     date=getattr(r, "date"))
        detections_json.append({
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

    signals_json = []
    for r in signals.itertuples(index=False):
        bar = int(getattr(r, "bar_idx"))
        plan = _plan(getattr(r, "tag"), None, bar=bar,
                     sl_atr=getattr(r, "sl_atr", None), tp_atr=getattr(r, "tp_atr", None),
                     side=getattr(r, "side", None))
        signals_json.append({
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

    trades_json = []
    if len(trades):
        for r in trades.itertuples(index=False):
            trades_json.append({
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

    by_pattern_json = []
    if len(by_pat):
        for tag, r in by_pat.iterrows():
            by_pattern_json.append({
                "tag": tag,
                "trades": int(r["trades"]),
                "win_rate": _num(r["win_rate"]),
                "total_pnl": _num(r["total_pnl"]),
                "avg_pnl": _num(r["avg_pnl"]),
                "profit_factor": _num_or_inf(r["profit_factor"]),
            })

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

    try:
        key_indicators_json = build_key_indicators(df_ohlcv, ind_df, ctx, raw_levels)
    except Exception:
        key_indicators_json = None

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

    metrics_json = {
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

    return {
        "symbol": symbol,
        "years": years,
        "bars": len(df_ohlcv),
        "last_date": _date(df_ohlcv.index[-1]),
        "ohlcv": _ohlcv_rows(df_ohlcv),
        "indicators": {
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
        },
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


class ChatRequest(BaseModel):
    symbol: str
    years: float = 0.5
    start: str | None = None
    end: str | None = None
    messages: list[ChatMessage]


def _chat_context(result: dict) -> str:
    """Compact analysis context: everything except the heavy per-bar series."""
    slim = {k: v for k, v in result.items() if k not in ("ohlcv", "indicators", "equity")}
    slim["detections"] = [d for d in slim.get("detections", []) if d.get("qualified")][-15:]
    slim["signals"] = slim.get("signals", [])[-15:]
    slim["trades"] = slim.get("trades", [])[-15:]
    slim["last_close"] = result["ohlcv"][-1]["close"] if result.get("ohlcv") else None
    slim.pop("summary_md", None)
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
    msgs = [{"role": m.role, "content": m.content} for m in req.messages[-20:]]
    system_blocks = [
        {"type": "text", "text": CHAT_SYSTEM, "cache_control": {"type": "ephemeral"}},
        {"type": "text",
         "text": f"Current analysis payload for {symbol}:\n{_chat_context(result)}"},
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


def _compact_signal(result: dict) -> dict:
    """Row-sized digest of a full analysis for the holdings table: verdicts,
    plan (stop/target), nearest support/resistance with distance, breakout
    state, and the 1-week base rate."""
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

    factors = [f for f in (advice.get("factors") or []) if abs(f.get("score") or 0) > 0]
    factors.sort(key=lambda f: -abs(f.get("score") or 0))
    h1w = ((result.get("outlook") or {}).get("horizons") or {}).get("1w") or {}

    return {
        "symbol": result["symbol"],
        "last_date": result.get("last_date"),
        "close": _num(close),
        "score": _num(advice.get("score")),
        "fresh": advice.get("fresh"),
        "holding": advice.get("holding"),
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
        "factors": factors[:4],
        "outlook_1w": {"p_up": _num(h1w.get("p_up")), "verdict": h1w.get("verdict"),
                       "n": h1w.get("n"), "fallback": h1w.get("fallback")},
        "key_indicators": result.get("key_indicators"),
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
        "  stored_at TEXT NOT NULL,"
        "  payload TEXT NOT NULL,"
        "  PRIMARY KEY (symbol, tier))"
    )
    return conn


class TakeRequest(BaseModel):
    symbol: str
    quantity: float | None = None
    avg_price: float | None = None
    deep: bool = False
    years: float = 1.0


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
         "text": f"Current analysis payload for {symbol}:\n{_chat_context(result)}"},
        {"type": "text",
         "text": TAKE_INSTRUCTIONS + (DEEP_INSTRUCTIONS if req.deep else "")},
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
                "INSERT OR REPLACE INTO claude_takes (symbol, tier, stored_at, payload) VALUES (?, ?, ?, ?)",
                (symbol, take["tier"], stored_at, json.dumps(take)),
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
            f"SELECT symbol, tier, stored_at, payload FROM claude_takes WHERE symbol IN ({placeholders})",
            syms,
        ).fetchall()
    finally:
        conn.close()
    out: dict[str, dict] = {}
    for sym, tier, stored_at, payload in rows:
        out.setdefault(sym, {})[tier] = {**json.loads(payload), "stored_at": stored_at}
    return {"takes": out}


@app.get("/api/analyze/{symbol}")
def analyze(
    symbol: str,
    years: float = Query(0.5, gt=0.0, le=10.0),
    start: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    end: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
):
    symbol = symbol.strip().upper()
    if not symbol:
        raise HTTPException(status_code=422, detail="Empty symbol")
    if start and end and start >= end:
        raise HTTPException(status_code=422, detail="start must be before end")
    try:
        result, was_cached = cached_analysis(symbol, years, start, end)
    except ValueError as e:
        # load_data raises ValueError("No data for <symbol>") on unknown tickers
        raise HTTPException(status_code=404, detail=str(e))
    return {**result, "cached": was_cached}
