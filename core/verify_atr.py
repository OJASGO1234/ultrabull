import numpy as np
import pandas as pd
import yfinance as yf

from stock_explorer import compute_indicators
from vbt_to_pd_ind import build_ind_df_from_vbt, extract_symbol_df


SYMBOL = "CAMS.NS"
PERIOD = 14
YEARS = 1.0
TOLERANCE = 1e-6


def independent_atr(df, period=14):
    """
    Independent reproduction of VECTORBT'S ATR CONVENTION -- NOT textbook
    Wilder ATR.

    UltraBull's production code calls `vbt.ATR.run(high, low, close,
    window=14)` (core/stock_explorer.py). Inspection of the installed
    vectorbt 1.0.0 source (vectorbt/indicators/basic.py: `ATR = ...
    window=14, ewm=True, adjust=False`, and vectorbt/generic/nb.py:
    `ewm_mean_1d_nb`, documented there as the numba equivalent of
    `pd.Series(a).ewm(span=span, adjust=adjust).mean()`) shows that
    vectorbt's ATR does NOT use Wilder's classic smoothing
    (ATR_t = (ATR_prev*(n-1) + TR_t)/n, i.e. alpha = 1/n). Instead it uses
    a pandas `span`-style EMA:

        alpha = 2 / (period + 1)          # = 2/15 for period=14
        ATR_0 = TR_0                       # seeded from the FIRST raw TR value
        ATR_t = (1 - alpha) * ATR_{t-1} + alpha * TR_t     # adjust=False

    Note on vectorbt's internal `old_wt` bookkeeping (vectorbt/generic/nb.py
    `ewm_mean_1d_nb`): for adjust=True it accumulates `old_wt += new_wt`
    across steps (a running-weight normalization), but for adjust=False --
    which is what UltraBull's call site uses -- it explicitly RESETS
    `old_wt = 1.0` after every step, so the blend
    `(old_wt*prev + alpha*cur) / (old_wt + alpha)` collapses on every step
    to `((1-alpha)*prev + alpha*cur) / 1.0`, i.e. exactly the textbook EMA
    recursion above. There is no lingering transient beyond bar 0 to
    converge from; it is the plain EMA formula from the second bar onward.

    IMPORTANT -- warm-up masking: vectorbt calls `ewm_mean_nb(tr, window,
    minp=window, adjust=False)`. Internally the EMA recursion still starts
    accumulating from the very first TR value (bar 0), but the function
    reports NaN for every bar before `nobs >= minp`, i.e. it MASKS bars
    0..period-2 as NaN and only exposes a numeric value from bar
    `period-1` onward -- even though that first reported value already
    reflects EMA smoothing carried from bar 0, not a simple average of the
    first `period` TR values (that "simple-average seed" is the Wilder
    convention, not this one). This function reproduces that exact masking
    below.

    True Range (unchanged, standard, same in both conventions):
        TR = max(
            High - Low,
            abs(High - Previous Close),
            abs(Low - Previous Close)
        )

    This function intentionally does NOT call vbt.ATR.run() -- it is a
    from-scratch reimplementation of vectorbt's documented/inspected
    algorithm, so the comparison below remains an independent check of
    "does an outside implementation of the SAME stated convention agree",
    not a tautological self-comparison.
    """

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = pd.Series(np.nan, index=df.index, dtype=float)

    if len(tr) == 0:
        return atr

    alpha = 2.0 / (period + 1)

    # Seed from the first raw TR value (vectorbt: weighted_avg = a[0]).
    raw_ema = np.empty(len(tr), dtype=float)
    weighted_avg = tr.iloc[0]
    raw_ema[0] = weighted_avg

    # EMA recursion with adjust=False, matching vectorbt's ewm_mean_1d_nb
    # exactly: old_wt is reset to 1.0 every step under adjust=False, so the
    # blend collapses to the plain (1-alpha)*prev + alpha*cur form.
    for i in range(1, len(tr)):
        cur = tr.iloc[i]
        old_wt = 1.0 - alpha
        if weighted_avg != cur:
            weighted_avg = (old_wt * weighted_avg + alpha * cur) / (old_wt + alpha)
        raw_ema[i] = weighted_avg

    # minp=window masking: only expose a value from bar `period-1` onward,
    # same as vectorbt's ewm_mean_nb(..., minp=window).
    if len(tr) >= period:
        atr.iloc[period - 1:] = raw_ema[period - 1:]

    return atr


def main():

    print("=" * 60)
    print("ULTRABULL ATR(14) INDEPENDENT VERIFICATION")
    print("=" * 60)

    print(f"Symbol       : {SYMBOL}")
    print(f"Period       : {PERIOD}")
    print(f"Years        : {YEARS}")

    # ---------------------------------------------------------
    # Download the same kind of raw market data
    # ---------------------------------------------------------

    df = yf.download(
        SYMBOL,
        period=f"{int(YEARS * 365)}d",
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise RuntimeError("No data downloaded.")

    print(f"Bars loaded  : {len(df)}")

    # ---------------------------------------------------------
    # Normalize the raw data
    # ---------------------------------------------------------

    df_ohlcv = extract_symbol_df(
        df,
        symbol=SYMBOL,
        case="title",
    )

    # ---------------------------------------------------------
    # Get UltraBull's ATR
    # ---------------------------------------------------------

    ind = compute_indicators(
        extract_symbol_df(
            df,
            symbol=SYMBOL,
            case="lower",
        )
    )

    ind_df = build_ind_df_from_vbt(
        df_ohlcv,
        ind,
        symbol=SYMBOL,
    )

    ultrabull_atr = ind_df["ATR"].astype(float)

    # ---------------------------------------------------------
    # Calculate ATR independently
    # ---------------------------------------------------------

    independent = independent_atr(
        df_ohlcv,
        period=PERIOD,
    )

    # ---------------------------------------------------------
    # Compare the two calculations
    # ---------------------------------------------------------

    comparison = pd.DataFrame(
        {
            "UltraBull_ATR": ultrabull_atr,
            "Independent_ATR": independent,
        }
    ).dropna()

    comparison["Difference"] = (
        comparison["UltraBull_ATR"]
        - comparison["Independent_ATR"]
    )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    max_difference = comparison["Difference"].abs().max()

    mean_difference = comparison["Difference"].abs().mean()

    mismatches = (
        comparison["Difference"].abs() > TOLERANCE
    ).sum()

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("VERIFICATION RESULTS")
    print("-" * 60)

    print(f"Bars compared        : {len(comparison)}")
    print(f"Maximum difference   : {max_difference:.10f}")
    print(f"Mean difference      : {mean_difference:.10f}")
    print(f"Mismatches           : {mismatches}")
    print(f"Tolerance            : {TOLERANCE}")

    # ---------------------------------------------------------
    # Latest values
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("LATEST 10 VALUES")
    print("-" * 60)

    print(
        comparison.tail(10).round(6).to_string()
    )

    # ---------------------------------------------------------
    # Verdict
    # ---------------------------------------------------------

    print("\n" + "-" * 60)

    if mismatches == 0:

        print("RESULT: PASS")
        print(
            "UltraBull ATR matches the independent calculation."
        )

    else:

        print("RESULT: INVESTIGATE")
        print(
            "Differences were detected between UltraBull "
            "and the independent ATR calculation."
        )

        print("\nLargest differences:")

        print(
            comparison
            .assign(
                AbsDifference=lambda x:
                    x["Difference"].abs()
            )
            .sort_values(
                "AbsDifference",
                ascending=False,
            )
            .head(10)
            .round(6)
            .to_string()
        )

    print("-" * 60)


if __name__ == "__main__":
    main()