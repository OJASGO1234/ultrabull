# core/calibrate.py
"""
Recalibrate the outlook — act on the audit's finding that the state base rates
are over-confident above ~55% (a "57% up" bucket realized ~50% out-of-sample).

Two outputs, both written to data_cache/ (the persistent volume, so a redeploy
doesn't wipe them; outlook.py prefers these over the committed defaults):

  1. state_baserates.json  — rebuilt on CURRENT data (fixes staleness).
  2. calibration.json      — a per-horizon shrink factor k, fit on held-out
     time to minimize Brier: calibrated_p = 0.5 + k*(raw_p - 0.5). k<1 pulls
     over-confident states toward a coin flip by exactly the amount the holdout
     says they're inflated. One interpretable number per horizon (temperature
     scaling in probability space) — robust, hard to overfit, honest.

    python calibrate.py <cache_dir>     # rebuild + fit
    python -m calibrate --refresh       # download fresh data first

The next audit measures the CALIBRATED outlook, so the loop closes: recalibrate
-> audit confirms ECE dropped -> done.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from event_study import UNIVERSE
from build_baserates import HORIZONS, WARM
import audit as _audit  # reuse its per-symbol pooling (_symbol_data)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data_cache")
BASERATES_OUT = os.path.join(DATA_DIR, "state_baserates.json")
CALIBRATION_OUT = os.path.join(DATA_DIR, "calibration.json")

HOLDOUT_DAYS = 365
MIN_STATE_N = 100


def _stats(v: pd.Series) -> dict:
    v = v.dropna()
    if not len(v):
        return {"n": 0, "p_up": None, "median": None, "mean": None, "q25": None, "q75": None}
    return {
        "n": int(len(v)),
        "p_up": round(float((v > 0).mean()), 4),
        "median": round(float(v.median()), 5),
        "mean": round(float(v.mean()), 5),
        "q25": round(float(v.quantile(0.25)), 5),
        "q75": round(float(v.quantile(0.75)), 5),
    }


def _pool(cache_dir: str | None) -> pd.DataFrame:
    frames = []
    for sym in UNIVERSE:
        try:
            res = _audit._symbol_data(sym, cache_dir)
        except Exception:
            res = None
        if res is not None:
            frames.append(res[0])  # the calibration frame: date, state, 1d/1w/1m/3m
    if not frames:
        raise RuntimeError("no data — refresh the price cache")
    return pd.concat(frames, ignore_index=True)


def _fit_shrink(pool: pd.DataFrame) -> dict:
    """Per-horizon shrink factor k that best calibrates fit-state predictions on
    the held-out window, by minimizing Brier."""
    pool = pool.copy()
    pool["date"] = pd.to_datetime(pool["date"])
    cutoff = pool["date"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
    fit, hold = pool[pool["date"] < cutoff], pool[pool["date"] >= cutoff]

    out = {}
    for name in HORIZONS:  # 1d/1w/1m/3m
        overall = float((fit[name].dropna() > 0).mean())
        state_p = {st: float((g[name].dropna() > 0).mean())
                   for st, g in fit.groupby("state") if g[name].notna().sum() >= MIN_STATE_N}
        h = hold.dropna(subset=[name]).copy()
        if len(h) < 200:
            out[name] = {"k": 1.0, "n_holdout": int(len(h)), "note": "too few holdout points — no shrink"}
            continue
        pred = h["state"].map(state_p).fillna(overall).to_numpy()
        realized = (h[name] > 0).to_numpy().astype(float)

        def _metrics(k):
            cal = np.clip(0.5 + k * (pred - 0.5), 0.0, 1.0)
            brier = float(np.mean((cal - realized) ** 2))
            return brier, cal

        ks = np.arange(0.0, 1.001, 0.02)
        briers = [(_metrics(k)[0], k) for k in ks]
        best_brier, best_k = min(briers)
        raw_brier, _ = _metrics(1.0)

        def _ece(p):
            df = pd.DataFrame({"p": p, "y": realized})
            df["b"] = pd.cut(df["p"], bins=[0, .45, .5, .55, .6, 1.01], right=False)
            num = 0.0
            for _, g in df.groupby("b", observed=True):
                if len(g):
                    num += len(g) * abs(g["p"].mean() - g["y"].mean())
            return num / len(df)

        out[name] = {
            "k": round(float(best_k), 3),
            "raw_brier": round(raw_brier, 4),
            "calibrated_brier": round(best_brier, 4),
            "raw_ece": round(_ece(np.clip(pred, 0, 1)), 4),
            "calibrated_ece": round(_ece(np.clip(0.5 + best_k * (pred - 0.5), 0, 1)), 4),
            "n_holdout": int(len(h)),
        }
    return out


def rebuild_and_calibrate(cache_dir: str | None) -> dict:
    pool = _pool(cache_dir)

    # 1. rebuild base rates on the FULL current window -> data_cache
    table = {"_meta": {"symbols": int(pool["state"].notna().sum() and len(UNIVERSE)),
                       "bars": int(len(pool)), "horizons": HORIZONS,
                       "note": "rebuilt on current data by calibrate.py"}}
    table["_all"] = {name: _stats(pool[name]) for name in HORIZONS}
    for st, g in pool.groupby("state"):
        table[st] = {name: _stats(g[name]) for name in HORIZONS}
    # Market-marginalized rows for the stock-only view — see build_baserates.py.
    # This is the table outlook.py actually prefers, so the rows must be minted
    # here too, not only in build_baserates.
    #
    # Note the shrink factors below are fit on the market-CONDITIONED states only
    # (_fit_shrink groups by "state", not "state_any") and outlook._calibrate then
    # applies the same k to these pooled rows. Measured, pooled rows sit about as
    # far from 0.5 but carry ~half the state-specific dispersion and double the n,
    # so their ideal k would be closer to 1: reusing the conditioned k OVER-shrinks
    # the stock-only outlook toward "no edge". The error is small (<=0.006 in
    # calibrated p) and points the safe way — it can never make the stock-only view
    # look more bullish than the evidence — so it is accepted deliberately.
    pool = pool.copy()
    pool["state_any"] = pool["state"].str.replace(r"^mkt=(?:up|dn)\|", "mkt=any|", regex=True)
    for st, g in pool.groupby("state_any"):
        table[st] = {name: _stats(g[name]) for name in HORIZONS}
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(BASERATES_OUT, "w") as f:
        json.dump(table, f, indent=1)

    # 2. fit the calibrator -> data_cache
    cal = {"fit_at": pd.Timestamp.now().isoformat(timespec="seconds"),
           "method": "probability-space shrink toward 0.5 (temperature scaling), fit on 12-month holdout",
           "horizons": _fit_shrink(pool)}
    with open(CALIBRATION_OUT, "w") as f:
        json.dump(cal, f, indent=1)

    return {"ok": True, "states": len(table) - 2, "bars": int(len(pool)), "calibration": cal}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cache = args[0] if args else DATA_DIR
    if "--refresh" in sys.argv:
        cache = None
    print(f"recalibrating (cache_dir={cache}) ...")
    res = rebuild_and_calibrate(cache)
    print(f"rebuilt {res['states']} states on {res['bars']} bars")
    for hz, d in res["calibration"]["horizons"].items():
        if "k" in d and d.get("raw_brier"):
            print(f"  {hz}: k={d['k']}  Brier {d['raw_brier']} -> {d['calibrated_brier']}  "
                  f"ECE {d['raw_ece']} -> {d['calibrated_ece']}")
