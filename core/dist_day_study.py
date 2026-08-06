# core/dist_day_study.py
"""
Re-validate the distribution-day ceiling (currently `dist_days_max = 5`).

The ceiling was validated once on a 4-year pool. It is now binding on ~86% of
recent sessions (vs ~51% across the full 4y), so before leaning on it harder —
or letting the UI switch it off — we re-measure two things:

  1. Does forward excess actually degrade as the distribution-day count rises,
     and where is the break? (bucket table + threshold sweep)
  2. Does the answer hold in the LAST 12 MONTHS, or is the 5-in-25 constant
     a stale artefact of the earlier sample? (period split)

Pool = production gate stack (market regime + RS + earnings blackout +
liquidity + the shipped disable list) with the dist-day ceiling itself OFF, so
the ceiling can be applied post-hoc at any threshold. The ceiling only ever
gated longs AND never touches the exempt reversal patterns (Hammer — see
rules/gates.py, where exempt patterns are stripped out of `bull` BEFORE the
ceiling is applied), so the headline tables are bullish-and-non-exempt only.

Statistics health warning, and the reason this file reports three standard
errors instead of one: events overlap heavily (adjacent detection dates share
9 of 10 forward bars) and cluster across 37 correlated large caps (~2.8 events
per date). An iid t-test on ~350 rows is really ~25 independent 10-day windows,
and it overstates significance by roughly a third. Date-clustered and block
bootstrap figures are the ones to quote.

Run:  python dist_day_study.py <cache_dir>   (from the core/ directory)
"""
from __future__ import annotations
import sys

import numpy as np
import pandas as pd

RNG_SEED = 20260725   # fixed so the bootstrap is reproducible run-to-run

from event_study import UNIVERSE, prepare_symbol, compute_event_rows
from evaluate_gates import _load_events_cached
from market_context import build_market_context, distribution_days
from patterns.postprocess import enrich_detections, direction_of
from rules.gates import apply_gates

# Production gate stack (api/main.py::build_cfg) minus the dist-day ceiling.
BASE_CFG = {
    "disable_patterns": [
        "RSI_60_Down", "RSI_40_Up", "Doji", "MACD_Cross_Zeroish_Up",
        "BB_SellFade", "Breakdown20", "Breakdown55", "VCP_Setup",
    ],
    "gates": {
        "bb_touch_eps": 0.0,
        "pullback_ma_tol": 0.01,
        "market": {
            "enabled": True,
            "require_market_regime": True,
            "require_sector_regime": False,
            "rs_col": "rs_20",
            "breakout_rs_min": 0.0,
            "breakdown_rs_max": 0.0,
            "dist_days_max": 0.0,   # OFF — applied post-hoc below
            "breadth_min": 0.0,
        },
        "events": {"enabled": True, "earnings_blackout_before": 3, "earnings_blackout_after": 1},
        "adx": {"breakout_min": 0.0, "meanrev_max": 0.0},
        "liquidity": {"min_turnover_20d": 2e7},
    },
}

# Mirrors gates.py's default `exempt_patterns`: these never see the ceiling in
# production, so including them here would credit/blame the gate for trades it
# does not touch.
EXEMPT_PATTERNS = {"Hammer"}

THRESHOLDS = [3, 4, 5, 6, 7, 8]
RECENT_DAYS = 365
BLOCK_DATES = 10        # block bootstrap: consecutive-date blocks
N_BOOT = 2000


def collect(years: float = 4, cache_dir: str | None = None) -> pd.DataFrame:
    """Event rows for the production pool, tagged with the NIFTY distribution-day
    count on the detection date."""
    dd = distribution_days(years + 0.3)
    if dd is None or not len(dd):
        raise SystemExit("dist_day_study: distribution_days() returned nothing (NIFTY fetch failed)")
    dd.index = pd.to_datetime(dd.index).normalize()

    frames = []
    for sym in UNIVERSE:
        try:
            prep = prepare_symbol(sym, years=years, cache_dir=cache_dir)
            if prep is None:
                continue
            df_ohlcv, ind_df, dets = prep
            ctx = build_market_context(sym, df_ohlcv["Close"], years=years, cache_dir=cache_dir)
            det_e = enrich_detections(dets, df_ohlcv, ind_df, sym, ctx=ctx)
            try:
                sym_events = _load_events_cached(sym, df_ohlcv, cache_dir)
            except Exception:
                sym_events = None
            gated = apply_gates(det_e, df_ohlcv, ind_df, BASE_CFG, ctx=ctx, events=sym_events)
            rows = compute_event_rows(gated, df_ohlcv, sym)
            if rows is not None and len(rows):
                frames.append(rows)
        except Exception as e:
            print(f"  {sym}: FAILED ({e})")

    if not frames:
        raise SystemExit("dist_day_study: no event rows collected")

    ev = pd.concat(frames, ignore_index=True)
    ev["direction"] = ev["pattern"].map(direction_of)
    ev["date"] = pd.to_datetime(ev["date"]).dt.normalize()
    ev["dist_days"] = ev["date"].map(dd)
    # gates.py:246 is PERMISSIVE on a missing count (`~isfinite(d)` keeps the
    # row). Dropping those rows instead would silently move ~1.6% of recent-12m
    # slots out of the kept bucket — so mark them -inf, which lands them in
    # "kept" under every threshold, exactly as production does.
    ev["dist_days"] = ev["dist_days"].fillna(-np.inf)
    ev["exempt"] = ev["pattern"].isin(EXEMPT_PATTERNS)
    # The population the ceiling can actually act on.
    ev["gateable"] = (ev["direction"] == "bullish") & ~ev["exempt"]

    cutoff = ev["date"].max() - pd.Timedelta(days=RECENT_DAYS)
    ev["period"] = np.where(ev["date"] > cutoff, "recent12m", "prior")

    # event_study's `excess` subtracts each symbol's FULL-4y mean forward
    # return, so on a period split any window that underperformed its own 4y
    # average reads negative by construction — the levels are not quotable as
    # "these days lost money". `adj` re-baselines within (period, symbol,
    # horizon), which is the honest level and leaves below-vs-above differences
    # essentially unchanged.
    ev["adj"] = ev["signed_ret"] - ev.groupby(
        ["period", "symbol", "horizon"])["signed_ret"].transform("mean")
    return ev


def dd_distribution(years: float = 4.3) -> pd.DataFrame:
    """The most direct test of "has the constant drifted": what the raw
    distribution-day count actually looks like, period by period. If the
    underlying rate of distribution days is unchanged, a fixed threshold near
    the median binds ~half the time in EVERY period by construction — which is
    a fact about the constant, not a signal about the market."""
    dd = distribution_days(years)
    dd.index = pd.to_datetime(dd.index).normalize()
    out = []
    for lbl, s in (("full 4y", dd), ("last 250", dd.tail(250)),
                   ("last 180", dd.tail(180)), ("last 120", dd.tail(120)),
                   ("last 60", dd.tail(60))):
        out.append({"window": lbl, "n": len(s), "median": s.median(),
                    "mean": round(float(s.mean()), 2),
                    "pct_ge_5": round(100.0 * float((s >= 5).mean()), 1)})
    return pd.DataFrame(out)


def _stats(x: pd.Series) -> tuple[int, float, float]:
    """(n, mean excess in bps, t-stat)."""
    n = len(x)
    if n == 0:
        return 0, np.nan, np.nan
    mu, sd = x.mean(), x.std(ddof=1)
    t = mu / (sd / np.sqrt(n)) if n > 1 and sd and sd > 0 else np.nan
    return n, mu * 1e4, t


def bucket_table(ev: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Gateable excess by distribution-day count — is the degradation monotone,
    and where does it break?

    Treat with suspicion: adjacent buckets swing by >100 bps with |t|>2.7 in
    the full period, which is regime noise rather than dose-response. The
    sweep and separation tables are the read; this one is here to show that no
    clean dose-response exists.
    """
    e = ev[(ev["horizon"] == horizon) & ev["gateable"]]
    out = []
    for period in ("full", "prior", "recent12m"):
        sub = e if period == "full" else e[e["period"] == period]
        for lo, hi, lbl in [(-np.inf, 2, "0-2"), (3, 3, "3"), (4, 4, "4"), (5, 5, "5"),
                            (6, 6, "6"), (7, 99, "7+")]:
            n, bps, t = _stats(sub.loc[sub["dist_days"].between(lo, hi), "excess"])
            out.append({"period": period, "dist_days": lbl, "n": n,
                        "bps": bps, "t": t})
    return pd.DataFrame(out)


def threshold_sweep(ev: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """For each candidate ceiling: what the kept pool earns, what the blocked
    pool earns, and how much of the pool the ceiling throws away.

    Note the shipped threshold is the argmax of the six swept here — so its
    apparent superiority is partly selection. Do not read the peak as
    localising the constant.
    """
    e = ev[(ev["horizon"] == horizon) & ev["gateable"]]
    out = []
    for period in ("full", "prior", "recent12m"):
        sub = e if period == "full" else e[e["period"] == period]
        n_all, bps_all, t_all = _stats(sub["excess"])
        out.append({"period": period, "thr": "none (no ceiling)", "n_kept": n_all,
                    "bps_kept": bps_all, "t_kept": t_all, "n_blocked": 0,
                    "bps_blocked": np.nan, "pct_blocked": 0.0, "lift_bps": 0.0})
        for thr in THRESHOLDS:
            keep = sub["dist_days"] < thr
            n_k, bps_k, t_k = _stats(sub.loc[keep, "excess"])
            n_b, bps_b, _ = _stats(sub.loc[~keep, "excess"])
            out.append({
                "period": period, "thr": f"< {thr}", "n_kept": n_k, "bps_kept": bps_k,
                "t_kept": t_k, "n_blocked": n_b, "bps_blocked": bps_b,
                "pct_blocked": 100.0 * n_b / max(1, n_all),
                "lift_bps": bps_k - bps_all if np.isfinite(bps_k) else np.nan,
            })
    return pd.DataFrame(out)


def _clustered_t(sub: pd.DataFrame, col: str, thr: float) -> float:
    """Difference-in-means t with standard errors clustered by DATE.

    Every stock reacting to the same NIFTY session is one observation, not 37.
    This is the minimum honest correction; the block bootstrap below also
    handles the 9-of-10 overlap between adjacent dates.
    """
    d = sub.assign(_side=(sub["dist_days"] < thr).astype(float))
    if d["_side"].nunique() < 2:
        return np.nan
    # date-level means, weighted by how many events that date carried
    g = d.groupby(["date", "_side"])[col].agg(["mean", "size"]).reset_index()
    below = g[g["_side"] == 1.0]
    above = g[g["_side"] == 0.0]
    if len(below) < 2 or len(above) < 2:
        return np.nan
    mb, ma = below["mean"].mean(), above["mean"].mean()
    se = np.sqrt(below["mean"].var(ddof=1) / len(below) + above["mean"].var(ddof=1) / len(above))
    return (mb - ma) / se if se > 0 else np.nan


def _block_bootstrap_p(sub: pd.DataFrame, col: str, thr: float,
                       block: int = BLOCK_DATES, n_boot: int = N_BOOT) -> tuple[float, float]:
    """Resample CONSECUTIVE-DATE BLOCKS to respect both the overlap in forward
    windows and the fact that market regimes arrive in runs.

    The statistic is a difference of pooled means and resampling is by whole
    dates, so per-date (sum, count) per side is a sufficient statistic: prefix
    sums give each block's totals in O(1) and no frame is ever rebuilt. Exact,
    and fast enough for a 2-core box.

    Returns (t_like, p_one_sided) where p is the share of bootstrap draws in
    which the below-ceiling pool did NOT beat the above-ceiling pool.
    """
    below = sub["dist_days"] < thr
    g = sub.assign(_v=sub[col], _b=below).groupby("date")
    agg = g.apply(lambda d: pd.Series({
        "sb": d.loc[d["_b"], "_v"].sum(), "nb": int(d["_b"].sum()),
        "sa": d.loc[~d["_b"], "_v"].sum(), "na": int((~d["_b"]).sum()),
    }), include_groups=False).sort_index()

    n_dates = len(agg)
    if n_dates < block * 2:
        return np.nan, np.nan

    # prefix sums -> block totals in O(1)
    pre = np.vstack([np.zeros(4), agg[["sb", "nb", "sa", "na"]].to_numpy().cumsum(axis=0)])
    n_starts = n_dates - block + 1
    blocks = pre[block:block + n_starts] - pre[:n_starts]      # (n_starts, 4)

    n_blocks = max(1, n_dates // block)
    rng = np.random.default_rng(RNG_SEED)
    picks = rng.integers(0, n_starts, size=(n_boot, n_blocks))
    tot = blocks[picks].sum(axis=1)                             # (n_boot, 4)

    sb, nb, sa, na = tot[:, 0], tot[:, 1], tot[:, 2], tot[:, 3]
    ok = (nb >= 2) & (na >= 2)
    if ok.sum() < 100:
        return np.nan, np.nan
    diffs = sb[ok] / nb[ok] - sa[ok] / na[ok]
    sd = diffs.std(ddof=1)
    t_like = diffs.mean() / sd if sd > 0 else np.nan
    return t_like, float((diffs <= 0).mean())


def separation_test(ev: pd.DataFrame, horizon: int, thr: float = 5.0,
                    col: str = "adj") -> pd.DataFrame:
    """Is below-ceiling forward return actually different from above-ceiling?

    Reported three ways because the naive one is wrong: `t_iid` assumes 350
    independent observations when there are ~25 independent windows. Quote
    `t_clustered` / `boot_p`, not `t_iid`.
    """
    e = ev[(ev["horizon"] == horizon) & ev["gateable"]]
    out = []
    for period in ("full", "prior", "recent12m"):
        sub = e if period == "full" else e[e["period"] == period]
        a = sub.loc[sub["dist_days"] < thr, col]
        b = sub.loc[sub["dist_days"] >= thr, col]
        na, nb = len(a), len(b)
        if na < 2 or nb < 2:
            out.append({"period": period, "n_below": na, "n_above": nb, "n_dates": 0,
                        "diff_bps": np.nan, "t_iid": np.nan,
                        "t_clustered": np.nan, "t_boot": np.nan, "boot_p": np.nan})
            continue
        diff = a.mean() - b.mean()
        se = np.sqrt(a.var(ddof=1) / na + b.var(ddof=1) / nb)
        t_boot, p_boot = _block_bootstrap_p(sub, col, thr)
        out.append({
            "period": period, "n_below": na, "n_above": nb,
            "n_dates": int(sub["date"].nunique()),
            "diff_bps": diff * 1e4,
            "t_iid": diff / se if se > 0 else np.nan,
            "t_clustered": _clustered_t(sub, col, thr),
            "t_boot": t_boot, "boot_p": p_boot,
        })
    return pd.DataFrame(out)


def _warn_if_pool_diverges() -> None:
    """api/main.py::build_cfg merges steward-approved pattern disables on top of
    the base list. That merge is additive, so once the steward disables anything
    this study's pool silently becomes a superset of production's. Cheap guard,
    since a divergence here invalidates every number the script prints."""
    try:
        from api.steward import load_overrides
        extra = set(load_overrides()["disabled"]) - set(BASE_CFG["disable_patterns"])
    except Exception:
        return
    if extra:
        print(f"  WARNING: steward has disabled {sorted(extra)} in production but this "
              f"study still includes them — pool is a superset of production.")


if __name__ == "__main__":
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"Distribution-day ceiling recheck over {len(UNIVERSE)} symbols, 4y daily bars…")
    _warn_if_pool_diverges()
    ev = collect(cache_dir=cache)
    pd.set_option("display.width", 200)

    span = f"{ev['date'].min().date()} → {ev['date'].max().date()}"
    n_recent = int((ev["period"] == "recent12m").sum())
    n_exempt = int(ev["exempt"].sum())
    print(f"\nPool: {len(ev)} event rows over {span} "
          f"({n_recent} in the last {RECENT_DAYS}d). Tables below are gateable-only: "
          f"bullish and not exempt ({n_exempt} exempt rows excluded — "
          f"{sorted(EXEMPT_PATTERNS)} never sees this ceiling in production).")

    print("\n=== Has the constant drifted? Raw distribution-day counts ===")
    print(dd_distribution().to_string(index=False))
    print("If the median sits at ~5 in every window, a 5-in-25 ceiling binds about\n"
          "half the time BY CONSTRUCTION — that is a property of the constant, not news\n"
          "about the market.")

    for h in (1, 3, 5, 10):
        print(f"\n=== Below-vs-above separation at the shipped ceiling (5) @ {h} bars ===")
        print(separation_test(ev, h).round(2).to_string(index=False))

    for h in (5, 10):
        print(f"\n=== Gateable excess (bps) by distribution-day count @ {h} bars ===")
        print(bucket_table(ev, h).round(2).to_string(index=False))
        print(f"\n=== Ceiling threshold sweep @ {h} bars ===")
        print(threshold_sweep(ev, h).round(2).to_string(index=False))

    ev.to_csv("dist_day_study_events.csv", index=False)
    print("\nwrote dist_day_study_events.csv")
