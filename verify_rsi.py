import numpy as np
import pandas as pd
import vectorbt as vbt

from core.stock_explorer import load_data
from core.vbt_to_pd_ind import extract_symbol_df


SYMBOL = "CAMS.NS"
YEARS = 1.0
RSI_PERIOD = 14




def independent_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Independent implementation of the RSI algorithm used
    by VectorBT's default RSI.

    VectorBT RSI defaults to ewm=False, which means
    simple rolling averages (SMA) are used for gains/losses.

    This function intentionally does NOT use VectorBT.
    """

    close = close.astype(float)

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(
        window=period,
        min_periods=period,
    ).mean()

    avg_loss = loss.rolling(
        window=period,
        min_periods=period,
    ).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


def main():

    print("=" * 60)
    print("ULTRABULL RSI(14) INDEPENDENT VERIFICATION")
    print("=" * 60)

    print(f"Symbol       : {SYMBOL}")
    print(f"Period       : {RSI_PERIOD}")
    print(f"Years        : {YEARS}")

    # ---------------------------------------------------------
    # 1. Get the SAME raw data used by UltraBull
    # ---------------------------------------------------------

    df = load_data(
        SYMBOL,
        years=YEARS,
    )

    # Yahoo Finance may return MultiIndex columns.
    # Normalize them exactly like UltraBull does.
    df_ohlcv = extract_symbol_df(
        df,
        symbol=SYMBOL,
        case="title",
    )

    close = df_ohlcv["Close"].astype(float)

    print(f"Bars loaded  : {len(close)}")

    # ---------------------------------------------------------
    # 2. UltraBull's RSI
    # ---------------------------------------------------------

    vbt_rsi = vbt.RSI.run(
        close,
        window=RSI_PERIOD,
    ).rsi

    # ---------------------------------------------------------
    # 3. OUR independent RSI
    # ---------------------------------------------------------

    independent = independent_rsi(
        close,
        period=RSI_PERIOD,
    )

    # ---------------------------------------------------------
    # 4. Put both results together
    # ---------------------------------------------------------

    comparison = pd.DataFrame({
        "UltraBull_RSI": vbt_rsi,
        "Independent_RSI": independent,
    })

    comparison["Difference"] = (
        comparison["UltraBull_RSI"]
        - comparison["Independent_RSI"]
    ).abs()

    # Compare only rows where BOTH calculations exist.
    comparison = comparison.dropna()

    # ---------------------------------------------------------
    # 5. Verification statistics
    # ---------------------------------------------------------

    max_difference = comparison["Difference"].max()
    mean_difference = comparison["Difference"].mean()

    tolerance = 1e-6

    mismatches = (
        comparison["Difference"] > tolerance
    ).sum()

    # ---------------------------------------------------------
    # 6. Print results
    # ---------------------------------------------------------

    print()
    print("-" * 60)
    print("VERIFICATION RESULTS")
    print("-" * 60)

    print(f"Bars compared        : {len(comparison)}")
    print(f"Maximum difference   : {max_difference:.10f}")
    print(f"Mean difference      : {mean_difference:.10f}")
    print(f"Mismatches           : {mismatches}")
    print(f"Tolerance            : {tolerance}")

    # ---------------------------------------------------------
    # 7. Show latest values
    # ---------------------------------------------------------

    print()
    print("-" * 60)
    print("LATEST 10 VALUES")
    print("-" * 60)

    print(
        comparison.tail(10).to_string()
    )

    # ---------------------------------------------------------
    # 8. Final result
    # ---------------------------------------------------------

    print()
    print("-" * 60)

    if mismatches == 0:

        print("RESULT: PASS")
        print(
            "UltraBull RSI matches the independent calculation."
        )

    else:

        print("RESULT: INVESTIGATE")
        print(
            "Differences were detected between "
            "VectorBT and the independent RSI calculation."
        )

        print()
        print("Largest differences:")

        print(
            comparison
            .sort_values(
                "Difference",
                ascending=False,
            )
            .head(10)
            .to_string()
        )

    print("-" * 60)


if __name__ == "__main__":
    main()