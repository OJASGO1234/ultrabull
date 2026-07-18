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

import copy
import math

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from stock_explorer import load_data, compute_indicators
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

app = FastAPI(title="Ultrabull API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    cfg["disable_patterns"] = ["RSI_60_Down", "RSI_40_Up"]
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


# ---------- pipeline runner ----------

def run_analysis(symbol: str, years: float) -> dict:
    df = load_data(symbol, years)
    ind = compute_indicators(df)

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

    if detections is None or detections.empty:
        det_enriched = empty_dets.assign(direction=None, alignment=None, qualified=False)
        signals = pd.DataFrame(columns=["bar_idx", "side", "tag"])
        trades = pd.DataFrame()
        equity = pd.Series(cfg["risk"]["capital"], index=df_ohlcv.index, dtype=float)
        sig_summary = {}
    else:
        det_enriched = enrich_detections(detections, df_ohlcv, ind_df, symbol)
        cut = warmup_cutoff(ind_df)
        det_enriched = det_enriched[det_enriched["date"] >= cut]
        sig_summary = summarize_signals(det_enriched)

        det_gated = apply_gates(det_enriched, df_ohlcv, ind_df, cfg)
        signals = compile_signals(det_gated, df_ohlcv, ind_df, cfg)
        trades, equity, _fills = simulate(df_ohlcv, signals, ind_df, cfg)

    metrics = summarize(trades, equity)
    by_pat = by_tag(trades)
    audit_bad = audit_exits(trades, cfg)
    summary_md = daily_summary_md(det_enriched, df_ohlcv, ind_df, symbol)

    # ---- serialize ----
    macd_hist = ind_df["MACD"] - ind_df["MACD_signal"]

    detections_json = []
    for r in det_enriched.itertuples(index=False):
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
        })

    signals_json = []
    for r in signals.itertuples(index=False):
        bar = int(getattr(r, "bar_idx"))
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
        },
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


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/analyze/{symbol}")
def analyze(symbol: str, years: float = Query(0.5, gt=0.0, le=10.0)):
    symbol = symbol.strip().upper()
    if not symbol:
        raise HTTPException(status_code=422, detail="Empty symbol")
    try:
        return run_analysis(symbol, years)
    except ValueError as e:
        # load_data raises ValueError("No data for <symbol>") on unknown tickers
        raise HTTPException(status_code=404, detail=str(e))
