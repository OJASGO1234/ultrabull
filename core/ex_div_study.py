# core/ex_div_study.py
"""
Is there a tradeable ex-dividend effect on NSE large caps, or only arithmetic?

The hypothesis under test: "if a stock goes up prior to a dividend event, it
should come down after the dividend is distributed" — i.e. a behavioural
run-up into the ex-date and a give-back after it, over and above the
mechanical price adjustment every ex-date carries by construction.

Three things get measured, and keeping them apart is the whole study:

  1. RUN-UP  — excess return over [-10,-5,-3,-1] bars ending the session
     BEFORE the ex-date.
  2. EX-DAY  — the close-to-close return across the ex-date itself, measured
     against the arithmetic expectation, -dividend/prev_close. If measured
     equals expected there is nothing behavioural at the ex-date at all; the
     drop is an accounting entry, not a trade.
  3. GIVE-BACK — excess return over [+1,+3,+5,+10] bars from the ex-date close.

DATA WARNING — this is the axis on which such a study is usually wrong.
`stock_explorer.load_data` (line 128) downloads with `auto_adjust=True`, which
BACK-ADJUSTS the whole history for dividends and so deletes the mechanical
ex-date drop from the series. Every cached pickle in data_cache/ is that
adjusted series. This script therefore downloads its own bars with
`auto_adjust=False` and reads BOTH columns off the same download:

  * `Close`      — raw, split-adjusted but NOT dividend-adjusted. Contains the
                   mechanical drop. This is what a holder's screen shows.
  * `Adj Close`  — dividend-back-adjusted, i.e. total-return. The mechanical
                   drop is removed by construction, so any move left here is
                   BEHAVIOURAL. This is the only column that can answer the
                   owner's question.

Every table below reports both. Reading the unadjusted give-back as an "edge"
would be reading arithmetic as alpha.

Baselining, per the CLAUDE.md gotcha about `event_study.compute_event_rows`:
that function subtracts each symbol's FULL-WINDOW mean forward return, so its
excess LEVELS are not comparable across a period split (any sub-window below
its own 4y average reads negative by construction). `excess` here mirrors it
for parity; `excess_p` re-baselines against the symbol's unconditional mean
k-bar return computed WITHIN the period. Quote `excess_p`. Levels here are the
whole point — this study has no below-vs-above contrast to hide behind — so
the full-window column is shown only to keep the two visibly separate.

Statistics health warning, same shape as dist_day_study.py: ex-dates cluster
hard in calendar time (Indian final dividends bunch into Jun-Aug, interims
into Jan-Feb), so dozens of "independent" events share a handful of weeks and
one market tape. Adjacent horizons also overlap. An iid t on ~450 rows is
worth far less than it looks. Date-clustered and block-bootstrap figures are
the ones to quote.

Run:  python ex_div_study.py <cache_dir>   (from the core/ directory)
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf

RNG_SEED = 20260806   # fixed so the bootstrap is reproducible run-to-run

from event_study import UNIVERSE as CORE_UNIVERSE
from market_context import SYMBOL_SECTOR

# CLAUDE.md: delisted post-demerger, excluded from UNIVERSE elsewhere too.
DELISTED = {"TATAMOTORS.NS", "GANESHCO.NS"}

# Default universe = the same 37 names every other study in this repo runs on
# (event_study.UNIVERSE), minus the delisted pair. `SYMBOL_SECTOR` has since
# grown to ~108 names including midcaps; `--wide` uses that instead, which
# triples the event count but is no longer "NSE large caps". The wide run is a
# robustness check, not the headline.
UNIVERSE = sorted(set(CORE_UNIVERSE) - DELISTED)
WIDE_UNIVERSE = sorted((set(SYMBOL_SECTOR) | set(CORE_UNIVERSE)) - DELISTED)

YEARS = 4.0
PRE_WINDOWS = [10, 5, 3, 1]     # bars ending the day BEFORE the ex-date
POST_WINDOWS = [1, 3, 5, 10]    # bars from the ex-date close
RECENT_DAYS = 365
BLOCK_DATES = 5                 # block bootstrap: consecutive-ex-date blocks
N_BOOT = 2000
YIELD_SPLIT = 1.0               # % of prev close — small vs meaningful dividend


# ---------------------------------------------------------------- data ------

def load_unadjusted(symbol: str, years: float = YEARS,
                    cache_dir: str | None = None) -> pd.DataFrame | None:
    """Daily bars with BOTH raw Close and Adj Close.

    Deliberately does not reuse `event_study._load_cached`: those pickles come
    from `load_data(auto_adjust=True)` and have no mechanical drop left in
    them. Cached under a distinct `_unadj_` prefix so the two never collide.
    """
    path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"{symbol.replace('&','_')}_unadj_{years}y.pkl")
        if os.path.exists(path):
            return pd.read_pickle(path)
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=int(365 * years) + 30)
    df = yf.download(symbol, start=start, end=end, interval="1d",
                     auto_adjust=False, progress=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(symbol, level=1, axis=1)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index).normalize()
    df = df[["Close", "Adj Close"]].dropna()
    if path:
        df.to_pickle(path)
    return df


def load_dividends(symbol: str, cache_dir: str | None = None) -> pd.Series | None:
    """Ex-dividend dates and per-share amounts, tz-stripped and normalized."""
    path = None
    if cache_dir:
        path = os.path.join(cache_dir, f"divs_{symbol.replace('&','_')}.pkl")
        if os.path.exists(path):
            return pd.read_pickle(path)
    try:
        d = yf.Ticker(symbol).dividends
    except Exception:
        return None
    if d is None or not len(d):
        d = pd.Series(dtype=float)
    d.index = pd.to_datetime(d.index)
    if getattr(d.index, "tz", None) is not None:
        d.index = d.index.tz_localize(None)
    d.index = d.index.normalize()
    if path:
        d.to_pickle(path)
    return d


def _mean_k_bar(close: pd.Series, k: int, mask: pd.Series | None = None) -> float:
    """Symbol's unconditional mean k-bar close-to-close return — the baseline
    an event return is 'excess' over. `mask` restricts to a period so the
    baseline is measured inside the same window as the events."""
    r = close.shift(-k) / close - 1.0
    if mask is not None:
        r = r[mask]
    r = r.dropna()
    return float(r.mean()) if len(r) else np.nan


# -------------------------------------------------------------- events ------

def collect(years: float = YEARS, cache_dir: str | None = None,
            universe: list[str] | None = None) -> pd.DataFrame:
    """One row per (ex-date, series, window). `series` is 'unadj' (contains the
    mechanical drop) or 'adj' (total-return, behavioural only)."""
    frames = []
    n_sym_ok = 0
    for sym in (universe or UNIVERSE):
        try:
            px = load_unadjusted(sym, years, cache_dir)
            if px is None or len(px) < 120:
                print(f"  {sym}: no usable price data")
                continue
            div = load_dividends(sym, cache_dir)
            if div is None or not len(div):
                print(f"  {sym}: no dividend history")
                continue
            idx = px.index
            div = div[(div.index >= idx[0]) & (div.index <= idx[-1])]
            div = div[div > 0]
            if not len(div):
                continue
            n_sym_ok += 1

            for series in ("unadj", "adj"):
                close = px["Close"] if series == "unadj" else px["Adj Close"]
                for ex_dt, amt in div.items():
                    # Ex-date should be a trading session; if Yahoo hands us a
                    # holiday, the first session on/after it IS the ex-session.
                    pos = idx.searchsorted(ex_dt, side="left")
                    if pos <= max(PRE_WINDOWS) or pos >= len(idx) - max(POST_WINDOWS):
                        continue
                    i = int(pos)
                    prev_close_raw = float(px["Close"].iloc[i - 1])
                    c = close.to_numpy()
                    row_base = {
                        "symbol": sym, "series": series,
                        "date": idx[i], "div": float(amt),
                        "div_yield_pct": 100.0 * float(amt) / prev_close_raw,
                    }
                    # run-up: from close[i-1-k] to close[i-1] (ends pre-ex)
                    for k in PRE_WINDOWS:
                        frames.append({**row_base, "window": f"pre{k}", "k": k,
                                       "ret": c[i - 1] / c[i - 1 - k] - 1.0})
                    # ex-day: close[i-1] -> close[i]
                    frames.append({**row_base, "window": "ex0", "k": 1,
                                   "ret": c[i] / c[i - 1] - 1.0})
                    # give-back: close[i] -> close[i+k]
                    for k in POST_WINDOWS:
                        frames.append({**row_base, "window": f"post{k}", "k": k,
                                       "ret": c[i + k] / c[i] - 1.0})
        except Exception as e:
            print(f"  {sym}: FAILED ({e})")

    if not frames:
        raise SystemExit("ex_div_study: no ex-dividend events collected (yfinance fetch failed?)")

    ev = pd.DataFrame(frames)
    ev["date"] = pd.to_datetime(ev["date"]).dt.normalize()
    cutoff = ev["date"].max() - pd.Timedelta(days=RECENT_DAYS)
    ev["period"] = np.where(ev["date"] > cutoff, "recent12m", "prior")
    ev["size"] = np.where(ev["div_yield_pct"] >= YIELD_SPLIT,
                          f">={YIELD_SPLIT:g}%", f"<{YIELD_SPLIT:g}%")
    ev["expected"] = np.where(ev["series"] == "unadj",
                              -ev["div_yield_pct"] / 100.0, 0.0)
    ev.attrs["n_sym"] = n_sym_ok

    # Baselines. `excess` mirrors event_study (full-window symbol mean) purely
    # for parity; `excess_p` re-baselines inside the period and is the quotable
    # one. Both subtract a k-bar mean matched to the event's own holding length.
    base_full: dict[tuple, float] = {}
    base_per: dict[tuple, float] = {}
    for sym in ev["symbol"].unique():
        px = load_unadjusted(sym, years, cache_dir)
        cutoff_sym = ev["date"].max() - pd.Timedelta(days=RECENT_DAYS)
        for series in ("unadj", "adj"):
            close = px["Close"] if series == "unadj" else px["Adj Close"]
            recent = pd.Series(close.index > cutoff_sym, index=close.index)
            for k in sorted(set(PRE_WINDOWS) | set(POST_WINDOWS) | {1}):
                base_full[(sym, series, k)] = _mean_k_bar(close, k)
                base_per[(sym, series, k, "recent12m")] = _mean_k_bar(close, k, recent)
                base_per[(sym, series, k, "prior")] = _mean_k_bar(close, k, ~recent)

    keys = list(zip(ev["symbol"], ev["series"], ev["k"]))
    ev["base_full"] = [base_full.get(x, np.nan) for x in keys]
    ev["base_period"] = [base_per.get((s, se, k, p), np.nan) for s, se, k, p
                         in zip(ev["symbol"], ev["series"], ev["k"], ev["period"])]
    ev["excess"] = ev["ret"] - ev["base_full"]
    ev["excess_p"] = ev["ret"] - ev["base_period"]
    # Ex-day only: what is left after the arithmetic is subtracted.
    ev["residual"] = ev["ret"] - ev["expected"]
    return ev


# ---------------------------------------------------------------- stats -----

def _stats(x: pd.Series) -> tuple[int, float, float]:
    """(n, mean in bps, iid t-stat). The iid t is the one NOT to quote."""
    n = len(x)
    if n == 0:
        return 0, np.nan, np.nan
    mu, sd = x.mean(), x.std(ddof=1)
    t = mu / (sd / np.sqrt(n)) if n > 1 and sd and sd > 0 else np.nan
    return n, mu * 1e4, t


def _clustered_t(sub: pd.DataFrame, col: str) -> float:
    """One-sample t on the mean, with standard errors clustered by DATE.

    Every stock going ex on the same session is one observation, not N: Indian
    ex-dates bunch into a few weeks a year and those weeks share one tape.
    This is the minimum honest correction; the block bootstrap below also
    handles the runs of adjacent ex-dates inside a dividend season.
    """
    g = sub.groupby("date")[col].mean()
    n = len(g)
    if n < 3:
        return np.nan
    sd = g.std(ddof=1)
    return float(g.mean() / (sd / np.sqrt(n))) if sd and sd > 0 else np.nan


def _block_bootstrap_p(sub: pd.DataFrame, col: str, block: int = BLOCK_DATES,
                       n_boot: int = N_BOOT) -> tuple[float, float]:
    """Resample CONSECUTIVE-DATE BLOCKS of ex-dates to respect the fact that
    ex-dividend events arrive in seasonal runs under a shared market regime.

    The statistic is a pooled mean and resampling is by whole dates, so per-date
    (sum, count) is a sufficient statistic: prefix sums give each block's totals
    in O(1) and no frame is ever rebuilt.

    Returns (t_like, p_one_sided) where p is the share of bootstrap draws whose
    mean falls on the opposite side of zero from the observed mean — i.e. the
    one-sided p for "the effect really has the sign we measured".
    """
    agg = sub.groupby("date")[col].agg(["sum", "size"]).sort_index()
    n_dates = len(agg)
    if n_dates < block * 2:
        return np.nan, np.nan

    pre = np.vstack([np.zeros(2), agg[["sum", "size"]].to_numpy().cumsum(axis=0)])
    n_starts = n_dates - block + 1
    blocks = pre[block:block + n_starts] - pre[:n_starts]      # (n_starts, 2)

    n_blocks = max(1, n_dates // block)
    rng = np.random.default_rng(RNG_SEED)
    picks = rng.integers(0, n_starts, size=(n_boot, n_blocks))
    tot = blocks[picks].sum(axis=1)                            # (n_boot, 2)
    means = tot[:, 0] / np.maximum(tot[:, 1], 1)

    sd = means.std(ddof=1)
    t_like = float(means.mean() / sd) if sd > 0 else np.nan
    obs = float(sub[col].mean())
    p = float((means >= 0).mean()) if obs < 0 else float((means <= 0).mean())
    return t_like, p


def _row(label_fields: dict, sub: pd.DataFrame, col: str) -> dict:
    n, bps, t_iid = _stats(sub[col])
    t_boot, p_boot = _block_bootstrap_p(sub, col)
    return {**label_fields, "n": n, "n_dates": int(sub["date"].nunique()),
            "bps": bps, "t_iid": t_iid, "t_clustered": _clustered_t(sub, col),
            "t_boot": t_boot, "boot_p": p_boot}


# --------------------------------------------------------------- tables -----

def profile(ev: pd.DataFrame, col: str = "excess_p", period: str = "full",
            size: str | None = None) -> pd.DataFrame:
    """The event profile: run-up, ex-day, give-back, side by side for the
    unadjusted (mechanical drop present) and adjusted (behavioural only)
    series. The hypothesis predicts positive `pre*` and negative `post*` in
    BOTH columns; arithmetic alone predicts them only in `unadj`, and only at
    `ex0`."""
    e = ev if period == "full" else ev[ev["period"] == period]
    if size:
        e = e[e["size"] == size]
    order = [f"pre{k}" for k in PRE_WINDOWS] + ["ex0"] + [f"post{k}" for k in POST_WINDOWS]
    out = []
    for series in ("unadj", "adj"):
        for w in order:
            sub = e[(e["series"] == series) & (e["window"] == w)]
            if len(sub) < 3:
                continue
            out.append(_row({"series": series, "window": w}, sub, col))
    return pd.DataFrame(out)


def mechanical_check(ev: pd.DataFrame) -> pd.DataFrame:
    """The decisive table. On the ex-date, unadjusted return vs its arithmetic
    expectation -div/prev_close. `residual_bps` is what is left over: if it is
    indistinguishable from zero, the ex-date move IS the dividend adjustment
    and there is no behavioural ex-day effect to trade."""
    e = ev[(ev["series"] == "unadj") & (ev["window"] == "ex0")]
    out = []
    for period in ("full", "prior", "recent12m"):
        sub = e if period == "full" else e[e["period"] == period]
        for size in ("all", f"<{YIELD_SPLIT:g}%", f">={YIELD_SPLIT:g}%"):
            s = sub if size == "all" else sub[sub["size"] == size]
            if len(s) < 3:
                continue
            r = _row({"period": period, "size": size}, s, "residual")
            r["measured_bps"] = 1e4 * float(s["ret"].mean())
            r["expected_bps"] = 1e4 * float(s["expected"].mean())
            r["residual_bps"] = r.pop("bps")
            out.append(r)
    cols = ["period", "size", "n", "n_dates", "measured_bps", "expected_bps",
            "residual_bps", "t_iid", "t_clustered", "t_boot", "boot_p"]
    return pd.DataFrame(out)[cols]


def size_split(ev: pd.DataFrame, col: str = "excess_p") -> pd.DataFrame:
    """A 0.3% dividend cannot produce a visible give-back, so pooling all
    dividend sizes washes out any real effect. Split at 1% of price."""
    out = []
    for size in (f"<{YIELD_SPLIT:g}%", f">={YIELD_SPLIT:g}%"):
        for series in ("unadj", "adj"):
            for w in ("pre10", "pre5", "ex0", "post5", "post10"):
                sub = ev[(ev["size"] == size) & (ev["series"] == series)
                         & (ev["window"] == w)]
                if len(sub) < 3:
                    continue
                out.append(_row({"size": size, "series": series, "window": w},
                                sub, col))
    return pd.DataFrame(out)


def period_split(ev: pd.DataFrame, col: str = "excess_p") -> pd.DataFrame:
    """Edges in this repo decay — the last 12 months are reported separately
    every time. Baseline is re-measured inside each period (`excess_p`), so
    these levels are comparable; the `excess` column would not be."""
    out = []
    for period in ("prior", "recent12m"):
        for series in ("unadj", "adj"):
            for w in ("pre10", "pre5", "ex0", "post5", "post10"):
                sub = ev[(ev["period"] == period) & (ev["series"] == series)
                         & (ev["window"] == w)]
                if len(sub) < 3:
                    continue
                out.append(_row({"period": period, "series": series, "window": w},
                                sub, col))
    return pd.DataFrame(out)


def runup_conditional(ev: pd.DataFrame, col: str = "excess_p") -> pd.DataFrame:
    """The owner's hypothesis stated literally: IF a stock ran up into the
    dividend, THEN it gives it back afterwards. Conditions the post-ex return
    on the sign of the pre-ex 10-bar excess run-up. Adjusted series only —
    the unadjusted one would just re-report the dividend."""
    pre = ev[(ev["series"] == "adj") & (ev["window"] == "pre10")][
        ["symbol", "date", col]].rename(columns={col: "runup"})
    out = []
    for w in ("post1", "post3", "post5", "post10"):
        post = ev[(ev["series"] == "adj") & (ev["window"] == w)]
        m = post.merge(pre, on=["symbol", "date"], how="inner")
        for lbl, mask in (("ran up", m["runup"] > 0), ("did not", m["runup"] <= 0)):
            sub = m[mask]
            if len(sub) < 3:
                continue
            out.append(_row({"window": w, "pre10": lbl}, sub, col))
    return pd.DataFrame(out)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    wide = "--wide" in sys.argv
    cache = args[0] if args else None
    uni = WIDE_UNIVERSE if wide else UNIVERSE
    print(f"Ex-dividend study over {len(uni)} NSE symbols "
          f"({'wide SYMBOL_SECTOR map incl. midcaps' if wide else 'core large-cap universe'}), "
          f"{YEARS:g}y daily bars…")
    print("Prices fetched with auto_adjust=False: raw Close keeps the mechanical\n"
          "ex-date drop, Adj Close removes it. Cached pickles from load_data() are\n"
          "adjusted and are deliberately NOT reused here.")
    ev = collect(cache_dir=cache, universe=uni)
    pd.set_option("display.width", 220)

    n_ev = int((ev["series"] == "adj").sum() and
               ev[ev["series"] == "adj"].groupby(["symbol", "date"]).ngroups)
    span = f"{ev['date'].min().date()} → {ev['date'].max().date()}"
    n_recent = ev[(ev["series"] == "adj") & (ev["period"] == "recent12m")] \
        .groupby(["symbol", "date"]).ngroups
    print(f"\n{n_ev} ex-dividend events across {ev['symbol'].nunique()} symbols, {span} "
          f"({n_recent} in the last {RECENT_DAYS}d). Median dividend yield "
          f"{ev['div_yield_pct'].median():.2f}% of price.")

    print("\n=== Is the ex-date move just arithmetic? (unadjusted vs -div/prev_close) ===")
    print(mechanical_check(ev).round(2).to_string(index=False))
    print("residual_bps ≈ 0 with |t_clustered| < 2 means the ex-date drop IS the\n"
          "dividend adjustment — arithmetic, not a behavioural effect.")

    print("\n=== Event profile, full period (excess_p, bps) ===")
    print(profile(ev).round(2).to_string(index=False))
    print("Hypothesis predicts pre* > 0 and post* < 0 in the ADJ rows too.\n"
          "If only the unadj rows move, the 'give-back' is the dividend.")

    print(f"\n=== By dividend size (split at {YIELD_SPLIT:g}% of price) ===")
    print(size_split(ev).round(2).to_string(index=False))

    print("\n=== Period split (prior vs last 12m, baseline re-measured in-period) ===")
    print(period_split(ev).round(2).to_string(index=False))

    print("\n=== Hypothesis stated literally: does a run-up get given back? (adj only) ===")
    print(runup_conditional(ev).round(2).to_string(index=False))

    ev.to_csv("ex_div_study_events.csv", index=False)
    print("\nwrote ex_div_study_events.csv")
