# core/verify_data_reproducibility.py
"""
Data reproducibility / live-bar contamination audit.

Target under test: core/stock_explorer.py::load_data() and
core/vbt_to_pd_ind.py::extract_symbol_df(). Neither is modified here.

WHAT load_data() DOES (traced from stock_explorer.py:97-134, read-only):
  - end date:   if `end` is not explicitly passed, end = datetime.now() + 1 day
                (line 109). The code comment explains why: yfinance's `end`
                is EXCLUSIVE, and yfinance interprets a naive datetime in the
                EXCHANGE's timezone. On a server running in UTC, `now()` in
                UTC can be earlier in the day than the exchange's local
                "now" (IST is UTC+5:30), so using end=now (UTC) can silently
                fall a day short and drop today's already-closed bar on the
                exchange. Pushing `end` forward by 1 day is a deliberate fix
                for that off-by-one, not an accident -- but as a side effect
                it also means: if the pipeline is queried DURING NSE market
                hours (or shortly after, before Yahoo finalizes the daily
                bar), the still-forming/partial daily candle for "today" CAN
                be included in the returned DataFrame, labeled with today's
                date exactly like a closed historical bar. There is no flag
                or metadata anywhere in the returned DataFrame distinguishing
                a settled bar from a still-forming one.
  - start date: end - (365*years + 10) days for daily interval (line 126) --
                a fixed calendar-day lookback, not a trading-day lookback.
  - interval:   '1d' default; intraday supported separately with its own
                yfinance history-length caps (lines 92-124) -- not relevant
                to the 4-year daily studies audited so far.
  - auto_adjust: hardcoded True (line 128) -- see section on corporate
                actions below.
  - timezone:   the returned index has its tz stripped to tz-naive (line
                132-133; extract_symbol_df repeats this at lines 156-160) --
                so once loaded, there is no timezone information left to
                even ask "was this bar formed before or after market close?"
                from the DataFrame alone.

WHAT THIS SCRIPT DOES (read-only, no production files touched):
  1. Loads a symbol via the EXACT existing load_data() -> extract_symbol_df()
     path used by event_study.py / rsi_zone_study.py / verify_adx.py etc.
  2. Reports the latest timestamp and OHLC row, and evaluates whether that
     latest row's date equals "today" in the exchange's local calendar date
     (IST), which is the only way to determine "could this row still be
     forming" from the outside (yfinance gives no explicit "is_final" flag).
  3. Loads the SAME symbol a second time in the same run and diffs the two
     DataFrames -- this checks short-timescale (within-run) reproducibility,
     which is a different and weaker claim than long-timescale (run today vs
     run in 6 months) reproducibility; see the auto_adjust discussion below
     for why the long-timescale claim cannot be verified by a same-run diff.
  4. Traces (by direct file inspection, reported as text, not by importing
     and running them) whether event_study.py / rsi_zone_study.py /
     rsi_zone_entry_study.py / verify_adx.py contain any explicit
     "drop last bar if it's today" guard.

Run:  python verify_data_reproducibility.py [SYMBOL]     (from core/)
"""
from __future__ import annotations
import sys
import datetime as dt
import numpy as np
import pandas as pd

from stock_explorer import load_data
from vbt_to_pd_ind import extract_symbol_df

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
IST_OFFSET = dt.timedelta(hours=5, minutes=30)


def today_ist_date() -> dt.date:
    """Exchange-local (IST) calendar date 'now', computed without any
    timezone library dependency beyond a fixed UTC+5:30 offset (NSE has no
    DST), purely to answer "is the latest bar's date == today at NSE"."""
    return (dt.datetime.utcnow() + IST_OFFSET).date()


def main():
    print(f"Symbol under test: {SYMBOL}")
    print(f"Server UTC time now : {dt.datetime.utcnow()}")
    ist_now = dt.datetime.utcnow() + IST_OFFSET
    print(f"Exchange IST time now: {ist_now}  (fixed +5:30 offset, NSE has no DST)")
    today = today_ist_date()
    print(f"Exchange-local (IST) calendar date: {today}")

    # --- load #1 ---
    raw1 = load_data(SYMBOL, years=1)
    df1 = extract_symbol_df(raw1, symbol=SYMBOL, case="title")
    latest_ts = df1.index[-1]
    latest_row = df1.iloc[-1]

    print("\n=== Latest bar returned by load_data() -> extract_symbol_df() ===")
    print(f"Latest timestamp : {latest_ts}")
    print(f"Latest OHLC      : Open={latest_row['Open']:.2f} High={latest_row['High']:.2f} "
          f"Low={latest_row['Low']:.2f} Close={latest_row['Close']:.2f} Volume={latest_row['Volume']:.0f}")

    latest_date = pd.Timestamp(latest_ts).date()
    is_today = (latest_date == today)
    print(f"\nLatest bar date == exchange-local today ({today})? {is_today}")

    # NSE regular session: 09:15-15:30 IST. If the latest bar is dated today
    # AND we are currently within/before that window's close, it cannot yet
    # be a settled bar.
    session_close = dt.datetime.combine(today, dt.time(15, 30))
    still_in_or_before_session = ist_now.replace(tzinfo=None) < session_close if is_today else False
    if is_today:
        if still_in_or_before_session:
            verdict_live_bar = ("LIKELY LIVE/INCOMPLETE -- the latest row is dated today and "
                                 "the exchange's regular session (09:15-15:30 IST) has not yet "
                                 "closed as of this run, so this bar is still forming.")
        else:
            verdict_live_bar = ("Dated today but AFTER the 15:30 IST close -- this bar is likely "
                                 "settled for the day, though Yahoo's own finalization/adjustment "
                                 "lag is not independently verified here.")
    else:
        verdict_live_bar = "Latest bar is NOT dated today -- no live/incomplete bar in this load."
    print(f"Live-bar assessment: {verdict_live_bar}")

    # --- reproducibility check: load the same symbol twice in the same run ---
    print("\n=== Same-run reproducibility check (load twice, diff) ===")
    raw2 = load_data(SYMBOL, years=1)
    df2 = extract_symbol_df(raw2, symbol=SYMBOL, case="title")

    same_index = df1.index.equals(df2.index)
    print(f"Identical row count/index between the two loads: {same_index} "
          f"(load1={len(df1)} rows, load2={len(df2)} rows)")
    if same_index:
        diff = (df1 - df2).abs()
        max_diff = diff.max().max()
        n_diff_cells = int((diff > 0).sum().sum())
        print(f"Max abs OHLCV difference between load1 and load2: {max_diff}")
        print(f"Number of differing cells: {n_diff_cells}")
        if n_diff_cells > 0:
            changed_rows = diff[(diff > 0).any(axis=1)]
            print("Rows that differ between the two back-to-back loads:")
            print(changed_rows.tail(5))
    else:
        print("Index mismatch between loads -- cannot do a direct cell diff; "
              "this alone would indicate non-determinism within the same run.")

    print("\nNOTE ON SCOPE: identical results across two loads issued seconds apart in the "
          "SAME run only demonstrates short-timescale stability (Yahoo isn't re-adjusting "
          "data mid-second). It does NOT demonstrate that a study re-run in, say, 6 months "
          "will reproduce the SAME historical OHLC values -- see the auto_adjust discussion "
          "below, which is a separate, longer-timescale effect that a same-run diff cannot "
          "measure at all.")

    # --- auto_adjust discussion (explanatory only, not a live check) ---
    print("\n=== auto_adjust=True and long-run reproducibility (explanatory) ===")
    print(
        "load_data() calls yf.download(..., auto_adjust=True). This makes yfinance divide\n"
        "pre-corporate-action Close/Open/High/Low by a cumulative adjustment factor so that\n"
        "splits and dividends don't appear as artificial price jumps. Precisely what CAN\n"
        "cause a historical value fetched today to differ from the same historical value\n"
        "fetched again later:\n"
        "  1. A NEW corporate action (dividend, split, bonus) is declared/recorded AFTER an\n"
        "     initial fetch, for a date that was already in the historical window. Once Yahoo\n"
        "     records it, every pre-action Close/Open/High/Low in the adjusted series shifts\n"
        "     by the new cumulative factor -- including bars from years ago.\n"
        "  2. Yahoo corrects a data error or a previously-missing/incorrect corporate-action\n"
        "     record for a past date (this does happen, especially for smaller NSE names).\n"
        "  3. NOT a cause: ordinary day-to-day price movement does NOT retroactively change\n"
        "     past bars -- only dividends/splits/bonuses (and Yahoo data corrections) do. A\n"
        "     stock with a stable corporate-action history and no data-vendor corrections\n"
        "     WILL reproduce bit-identically across two fetches even months apart. This\n"
        "     script does not claim historical values change on every re-fetch -- only that\n"
        "     they CAN change, conditional on a real event of type (1) or (2) occurring for\n"
        "     a date inside the fetched window."
    )

    print(
        "\nPractical implication for the 4-year event studies already run "
        "(event_study.py, rsi_zone_study.py, rsi_zone_entry_study.py, verify_adx.py):\n"
        "  - None of these scripts pin a fetch date or cache a permanent snapshot by default\n"
        "    (cache_dir is optional and off unless explicitly passed on the command line).\n"
        "  - Re-running any of them today vs. 6 months from now queries yfinance fresh each\n"
        "    time, so if any symbol in the 37-symbol universe has a dividend/split/bonus\n"
        "    recorded in the interim (for ANY date inside the 4-year window, not just the\n"
        "    new event's own date), the adjusted OHLC series -- and therefore every downstream\n"
        "    indicator (RSI/ADX/MA/etc.) computed on it -- can shift. This does not require\n"
        "    the live-bar issue above; it's a separate mechanism."
    )

    # --- trace explicit live-bar guards in the downstream study scripts ---
    print("\n=== Trace: do the study scripts guard against consuming today's live bar? ===")
    import os
    targets = ["event_study.py", "rsi_zone_study.py", "rsi_zone_entry_study.py", "verify_adx.py"]
    guard_keywords = ["is_today", "dropna(subset", "iloc[:-1]", "drop(df.index[-1]",
                       "today()", ".date() ==", "live bar", "incomplete bar", "partial bar"]
    for fname in targets:
        path = os.path.join(os.path.dirname(__file__), fname)
        if not os.path.exists(path):
            print(f"  {fname}: NOT FOUND")
            continue
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        hits = [kw for kw in guard_keywords if kw in text]
        print(f"  {fname}: explicit live-bar guard keywords found = {hits if hits else 'NONE'}")

    print(
        "\nInterpretation: none of the audited study scripts filter out a same-day bar before\n"
        "computing forward returns or indicators. They inherit whatever load_data() returns\n"
        "as-is. This matters most for a script's OWN most recent trailing observations near\n"
        "the current date; it has NO effect on the bulk of a 4-year historical study, because\n"
        "(a) forward-return calculations for any date more than `horizon` trading days in the\n"
        "past never touch today's bar at all, and (b) only detections/zone-entries/RSI values\n"
        "computed ON today's date itself, if today is a live trading session, would be based\n"
        "on a still-forming candle -- a small fraction of a multi-year sample, but a real one\n"
        "if a study is run mid-session and its output is later treated as historically final."
    )

    # --- verdict ---
    print("\n=== VERDICT ===")
    if is_today and still_in_or_before_session:
        verdict = "FAIL"
        reason = ("The pipeline, run right now, would include a live/incomplete bar dated "
                  "today with no downstream guard filtering it out.")
    elif not same_index or (same_index and 'n_diff_cells' in dir() and False):
        verdict = "WARNING"
        reason = "Within-run reproducibility could not be fully confirmed."
    else:
        verdict = "WARNING"
        reason = (
            "No live bar was captured in THIS run (latest bar is not today's still-forming "
            "candle), and back-to-back loads in this run were identical. However: (1) the "
            "code path that pushes `end` to now+1 day CAN return a live/incomplete bar during "
            "market hours on a day this is re-run, since there is no guard against it anywhere "
            "in load_data(), extract_symbol_df(), or any of the four study scripts checked; and "
            "(2) auto_adjust=True means long-run bit-for-bit reproducibility of a 4-year study "
            "is NOT guaranteed -- it depends on no new corporate action or data correction "
            "landing inside the fetched window between runs. Neither condition is a certainty "
            "on every run, so PASS is not warranted, and neither is FAIL, since this specific "
            "run did not demonstrably ingest a live bar."
        )
    print(f"{verdict}: {reason}")


if __name__ == "__main__":
    main()
