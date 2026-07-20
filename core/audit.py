# core/audit.py
"""
Living validation audit — the first tick of the health loop.

Two questions, both measured on held-out time (never in-sample):

  A. CALIBRATION — is the outlook honest? Rebuild the state base-rate table on
     an EARLY window, then on the recent HOLDOUT window compare each bar's
     predicted P(up) against what actually happened. If the app says "57 of 100
     similar days close higher" and the holdout realizes 50, we quantify that.

  B. DECAY — which patterns/gates still have edge? Re-run the event study and
     split each pattern's forward-return excess into full-window vs the recent
     12 months, with t-stats. Flag anything that was an edge but lost it. Also
     checks the flagship market-regime gate (does up-market excess still beat
     down-market excess for longs?).

  C. COVERAGE — which of the user's actual holdings are NOT in the study
     universe, i.e. scored by stats that don't describe them.

Results persist to data_cache/audit.db (latest + history, so trend is visible)
and are served at /api/audit. Rerunnable like the other research scripts:

    python audit.py <cache_dir>          # fast: reuse cached pickles
    python -m core.audit --refresh       # download fresh data (production)

Deterministic throughout — the numbers are facts. The Claude steward layer
(Phase 1.5) reasons OVER this output; it never recomputes it.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import date as _date_cls

import numpy as np
import pandas as pd

from event_study import UNIVERSE, prepare_symbol, compute_event_rows, HORIZONS as EV_HORIZONS
from build_baserates import state_series, HORIZONS as BR_HORIZONS, WARM
from market_context import build_market_context

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_DB = os.path.join(ROOT, "data_cache", "audit.db")

HOLDOUT_DAYS = 365          # recent window held out of the fit
MIN_STATE_N = 100           # below this a fit-state p_up is too thin -> fallback
CALIB_HORIZONS = ["1d", "1w", "1m"]
HEADLINE_H = 5              # event-study horizon (bars) used for pattern headline

# Patterns validated as having raw edge in the original 2026 study (from
# CLAUDE.md). Everything else was context-only or rejected — shown for
# reference so the report can say "this was validated; here's its status now".
VALIDATED_2026 = {"Hammer": "long", "BearishEngulfing": "short"}


# ---------- per-symbol pass (one prepare_symbol per symbol, reused for A+B) ----------

def _symbol_data(sym: str, cache_dir: str | None):
    prep = prepare_symbol(sym, years=4, cache_dir=cache_dir)
    if prep is None:
        return None
    df, ind, dets = prep
    close = df["Close"]
    ctx = build_market_context(sym, close, years=4, cache_dir=cache_dir)
    states = state_series(df, ind, ctx)

    calib = pd.DataFrame({"date": df.index, "state": states.values})
    for name, h in BR_HORIZONS.items():
        calib[name] = (close.shift(-h) / close - 1.0).values
    calib = calib.iloc[WARM:]

    events = compute_event_rows(dets, df, sym)  # may be None
    mkt = None
    try:
        mkt = pd.Series(ctx["mkt_up"].astype(bool).values, index=df.index)
    except Exception:
        pass
    return calib, events, mkt


def _stat(vals: pd.Series) -> dict:
    v = vals.dropna()
    n = len(v)
    if n == 0:
        return {"n": 0, "mean_excess_bps": None, "t_stat": None, "hit_rate": None}
    mu, sd = float(v.mean()), float(v.std(ddof=1)) if n > 1 else 0.0
    return {
        "n": int(n),
        "mean_excess_bps": round(mu * 1e4, 1),
        "t_stat": round(mu / (sd / np.sqrt(n)), 2) if n > 2 and sd > 0 else None,
        "hit_rate": round(float((v > 0).mean()), 3),
    }


# ---------- Part A: calibration ----------

def _calibration(calib: pd.DataFrame) -> dict:
    calib = calib.copy()
    calib["date"] = pd.to_datetime(calib["date"])
    max_date = calib["date"].max()
    cutoff = max_date - pd.Timedelta(days=HOLDOUT_DAYS)
    fit = calib[calib["date"] < cutoff]
    holdout = calib[calib["date"] >= cutoff]

    out = {"cutoff": cutoff.date().isoformat(), "max_date": max_date.date().isoformat(),
           "fit_bars": int(len(fit)), "holdout_bars": int(len(holdout)), "horizons": {}}
    if len(fit) < 500 or len(holdout) < 200:
        out["note"] = "too little data to split — calibration inconclusive"
        return out

    for name in CALIB_HORIZONS:
        # fit table: per-state p_up, with an overall fallback for thin states
        overall_p = float((fit[name].dropna() > 0).mean())
        state_p = {}
        for st, g in fit.groupby("state"):
            up = g[name].dropna()
            if len(up) >= MIN_STATE_N:
                state_p[st] = float((up > 0).mean())

        h = holdout.dropna(subset=[name]).copy()
        if len(h) < 200:
            continue
        h["pred"] = h["state"].map(state_p).fillna(overall_p)
        h["realized"] = (h[name] > 0).astype(float)

        bins = [0, 0.45, 0.50, 0.55, 0.60, 1.01]
        labels = ["<45%", "45-50%", "50-55%", "55-60%", "60%+"]
        h["bucket"] = pd.cut(h["pred"], bins=bins, labels=labels, right=False)
        rel, ece_num = [], 0.0
        for lab in labels:
            b = h[h["bucket"] == lab]
            if len(b):
                pr, re = float(b["pred"].mean()), float(b["realized"].mean())
                rel.append({"bucket": lab, "n": int(len(b)),
                            "predicted": round(pr, 3), "realized": round(re, 3)})
                ece_num += len(b) * abs(pr - re)
        # ECE: n-weighted gap between predicted and realized PER BUCKET (the honest
        # calibration number — per-bar |pred-realized| is ~0.5 by construction on a
        # binary outcome and says nothing). Brier vs the 0.25 coin-flip baseline
        # tells you whether the states carry any skill at all.
        ece = ece_num / len(h)
        brier = float(((h["pred"] - h["realized"]) ** 2).mean())
        out["horizons"][name] = {
            "n": int(len(h)),
            "ece": round(ece, 4),
            "brier": round(brier, 4),
            "brier_baseline": 0.25,  # a pure coin flip
            "overall_predicted": round(float(h["pred"].mean()), 3),
            "overall_realized": round(float(h["realized"].mean()), 3),
            "reliability": rel,
        }
    return out


# ---------- Part B: pattern + gate decay ----------

def _pattern_decay(events: pd.DataFrame) -> dict:
    """Split each pattern's excess into an EARLY window (fit) vs the RECENT 12
    months (holdout) — so decay = 'had edge early, lost it recently', not a
    recent-vs-full comparison that hides the recent slice inside 'full'."""
    ev = events[events["horizon"] == HEADLINE_H].copy()
    ev["date"] = pd.to_datetime(ev["date"])
    max_date = ev["date"].max()
    cutoff = max_date - pd.Timedelta(days=HOLDOUT_DAYS)

    def edge(s):  # a positive, significant excess
        return s["t_stat"] is not None and s["t_stat"] >= 2 and (s["mean_excess_bps"] or 0) > 0

    def neg(s):   # a significant NEGATIVE excess
        return s["t_stat"] is not None and s["t_stat"] <= -2 and (s["mean_excess_bps"] or 0) < 0

    patterns = []
    for pat, g in ev.groupby("pattern"):
        early = _stat(g[g["date"] < cutoff]["excess"])
        recent = _stat(g[g["date"] >= cutoff]["excess"])
        full = _stat(g["excess"])
        validated = pat in VALIDATED_2026
        # status: a traded/validated pattern turning weak or negative is the alert
        weak_recent = recent["t_stat"] is None or recent["t_stat"] < 2 or (recent["mean_excess_bps"] or 0) <= 0
        if recent["n"] < 15:
            status = "insufficient"
        elif edge(early) and weak_recent:
            status = "decayed"                    # had a real edge early, lost it
        elif (edge(early) or validated) and edge(recent):
            status = "holding"                    # still working
        elif validated and weak_recent:
            status = "unconfirmed"                # trusted but no measurable edge in this window
        elif not edge(early) and edge(recent):
            status = "emerging"                   # new positive — treat cautiously
        elif neg(recent):
            status = "negative"                   # significantly negative recently
        else:
            status = "no_edge"
        patterns.append({
            "pattern": pat, "validated_2026": validated,
            "early": early, "recent_12m": recent, "full": full, "status": status,
        })
    order = {"decayed": 0, "negative": 1, "unconfirmed": 2, "emerging": 3, "holding": 4, "no_edge": 5, "insufficient": 6}
    patterns.sort(key=lambda p: (0 if p["validated_2026"] else 1, order.get(p["status"], 9)))
    return {"headline_horizon_bars": HEADLINE_H, "cutoff": cutoff.date().isoformat(),
            "patterns": patterns}


def _regime_gate_decay(events: pd.DataFrame, mkt_by_sym: dict) -> dict:
    """Does the market-regime gate still add value? The validated gate keeps
    LONG detections only when NIFTY MA20>MA50. So its value = mean excess of
    long detections in an up-market minus the mean excess of ALL long
    detections (what filtering buys you). Measured at the study's 10-bar
    horizon, early vs recent. Positive = the gate still helps."""
    from patterns.postprocess import direction_of
    ev = events[events["horizon"] == 10].copy()
    ev["long"] = ev["pattern"].map(lambda p: direction_of(p) == "bullish")
    ev = ev[ev["long"]].copy()
    if ev.empty:
        return {"gate": "market_regime", "status": "insufficient"}
    ev["date"] = pd.to_datetime(ev["date"])

    def _mkt(row):
        s = mkt_by_sym.get(row["symbol"])
        try:
            return bool(s.get(pd.Timestamp(row["date"]))) if s is not None else None
        except Exception:
            return None
    ev["mkt_up"] = ev.apply(_mkt, axis=1)
    ev = ev.dropna(subset=["mkt_up"])
    cutoff = ev["date"].max() - pd.Timedelta(days=HOLDOUT_DAYS)

    def _val(df):
        allx = _stat(df["excess"])
        gated = _stat(df[df["mkt_up"]]["excess"])
        gain = (gated["mean_excess_bps"] - allx["mean_excess_bps"]) \
            if (gated["mean_excess_bps"] is not None and allx["mean_excess_bps"] is not None) else None
        return {"all_longs": allx, "up_market_only": gated,
                "gate_gain_bps": round(gain, 1) if gain is not None else None}

    full = _val(ev)
    recent = _val(ev[ev["date"] >= cutoff])
    status = "holding" if (full["gate_gain_bps"] or 0) > 0 and (recent["gate_gain_bps"] or 0) > 0 else "watch"
    return {"gate": "market_regime", "horizon_bars": 10, "full": full,
            "recent_12m": recent, "status": status,
            "note": "gate value = up-market long excess minus all-long excess; positive means filtering to up-markets still helps"}


# ---------- Part C: holdings coverage ----------

def _coverage(held_symbols: list[str]) -> dict:
    uni = set(UNIVERSE)
    covered = [s for s in held_symbols if s in uni]
    missing = [s for s in held_symbols if s not in uni]
    return {"universe_size": len(uni), "held": len(held_symbols),
            "covered": covered, "not_covered": missing,
            "coverage_pct": round(100 * len(covered) / len(held_symbols), 1) if held_symbols else None}


# ---------- orchestration + storage ----------

def run_audit(cache_dir: str | None = None, held_symbols: list[str] | None = None) -> dict:
    calib_frames, event_frames, mkt_by_sym, used = [], [], {}, []
    for sym in UNIVERSE:
        try:
            res = _symbol_data(sym, cache_dir)
        except Exception:
            res = None
        if res is None:
            continue
        calib, events, mkt = res
        calib_frames.append(calib)
        if events is not None:
            event_frames.append(events)
        if mkt is not None:
            mkt_by_sym[sym] = mkt
        used.append(sym)

    if not calib_frames or not event_frames:
        return {"ok": False, "error": "no data — refresh the price cache"}

    calib = pd.concat(calib_frames, ignore_index=True)
    events = pd.concat(event_frames, ignore_index=True)

    report = {
        "generated_at": None,  # stamped by caller (Date.now unavailable in some contexts)
        "symbols_used": len(used),
        "calibration": _calibration(calib),
        "pattern_decay": _pattern_decay(events),
        "regime_gate": _regime_gate_decay(events, mkt_by_sym),
        "coverage": _coverage(held_symbols or []),
    }
    return {"ok": True, "report": report}


def _db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(AUDIT_DB), exist_ok=True)
    conn = sqlite3.connect(AUDIT_DB, timeout=10.0)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS audits ("
        "  generated_at TEXT PRIMARY KEY,"
        "  report TEXT NOT NULL)"
    )
    return conn


def save_audit(report: dict, generated_at: str) -> None:
    report["generated_at"] = generated_at
    conn = _db()
    try:
        with conn:
            conn.execute("INSERT OR REPLACE INTO audits (generated_at, report) VALUES (?, ?)",
                         (generated_at, json.dumps(report, default=str)))
    finally:
        conn.close()


def latest_audit() -> dict | None:
    conn = _db()
    try:
        row = conn.execute("SELECT report FROM audits ORDER BY generated_at DESC LIMIT 1").fetchone()
    finally:
        conn.close()
    return json.loads(row[0]) if row else None


def audit_history(limit: int = 26) -> list[dict]:
    """Thin per-run trend rows (dates + a couple headline numbers), newest first."""
    conn = _db()
    try:
        rows = conn.execute("SELECT report FROM audits ORDER BY generated_at DESC LIMIT ?", (limit,)).fetchall()
    finally:
        conn.close()
    out = []
    for (r,) in rows:
        rep = json.loads(r)
        decayed = [p["pattern"] for p in rep.get("pattern_decay", {}).get("patterns", [])
                   if p["status"] == "decayed"]
        cal = (rep.get("calibration", {}).get("horizons", {}).get("1w") or {})
        out.append({"generated_at": rep.get("generated_at"),
                    "calibration_error_1w": cal.get("calibration_error"),
                    "decayed_patterns": decayed})
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cache = args[0] if args else os.path.join(ROOT, "data_cache")
    if "--refresh" in sys.argv:
        cache = None  # force fresh download
    print(f"running audit (cache_dir={cache}) ...")
    res = run_audit(cache_dir=cache)
    if not res.get("ok"):
        print("FAILED:", res.get("error"))
        sys.exit(1)
    stamp = pd.Timestamp.now().isoformat(timespec="seconds")
    save_audit(res["report"], stamp)
    rep = res["report"]
    print(f"\nsymbols_used={rep['symbols_used']}  stamped {stamp}")
    for hz, d in rep["calibration"]["horizons"].items():
        print(f"  calibration {hz}: err={d['calibration_error']} "
              f"predicted={d['overall_predicted']} realized={d['overall_realized']} (n={d['n']})")
    print("  decayed patterns:", [p["pattern"] for p in rep["pattern_decay"]["patterns"]
                                   if p["status"] == "decayed"] or "none")
