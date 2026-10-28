import numpy as np
import pandas as pd
import yfinance as yf

from stock_explorer import compute_indicators
from vbt_to_pd_ind import build_ind_df_from_vbt, extract_symbol_df


SYMBOL = "CAMS.NS"
YEARS = 1.0


def independent_sma(close, window):
    return close.rolling(window=window).mean()


def main():
    print("=" * 60)
    print("ULTRABULL MA INDEPENDENT VERIFICATION")
    print("=" * 60)
    print(f"Symbol : {SYMBOL}")
    print(f"Period : 20 / 50")
    print(f"Years  : {YEARS}")

    # Same raw source used by UltraBull
    df = yf.download(
        SYMBOL,
        period=f"{int(YEARS * 365)}d",
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise RuntimeError("No data downloaded.")

    # Normalize raw OHLCV
    df_ohlcv = extract_symbol_df(df, symbol=SYMBOL, case="title")

    # UltraBull indicators
    ind = compute_indicators(
        extract_symbol_df(df, symbol=SYMBOL, case="lower")
    )

    ind_df = build_ind_df_from_vbt(
        df_ohlcv,
        ind,
        symbol=SYMBOL,
    )

    close = df_ohlcv["Close"].astype(float)

    # Independent calculations
    ma20_ind = independent_sma(close, 20)
    ma50_ind = independent_sma(close, 50)

    # Compare
    comparison = pd.DataFrame({
        "UltraBull_MA20": ind_df["MA20"],
        "Independent_MA20": ma20_ind,
        "MA20_Difference": ind_df["MA20"] - ma20_ind,

        "UltraBull_MA50": ind_df["MA50"],
        "Independent_MA50": ma50_ind,
        "MA50_Difference": ind_df["MA50"] - ma50_ind,
    }).dropna()

    max_diff_20 = comparison["MA20_Difference"].abs().max()
    max_diff_50 = comparison["MA50_Difference"].abs().max()

    mean_diff_20 = comparison["MA20_Difference"].abs().mean()
    mean_diff_50 = comparison["MA50_Difference"].abs().mean()

    mismatch_20 = (
        comparison["MA20_Difference"].abs() > 1e-6
    ).sum()

    mismatch_50 = (
        comparison["MA50_Difference"].abs() > 1e-6
    ).sum()

    print("\n" + "-" * 60)
    print("VERIFICATION RESULTS")
    print("-" * 60)

    print(f"Bars compared : {len(comparison)}")

    print("\nMA20")
    print(f"Maximum difference : {max_diff_20:.10f}")
    print(f"Mean difference    : {mean_diff_20:.10f}")
    print(f"Mismatches         : {mismatch_20}")

    print("\nMA50")
    print(f"Maximum difference : {max_diff_50:.10f}")
    print(f"Mean difference    : {mean_diff_50:.10f}")
    print(f"Mismatches         : {mismatch_50}")

    print("\n" + "-" * 60)
    print("LATEST 10 VALUES")
    print("-" * 60)

    print(comparison.tail(10).round(6).to_string())

    if mismatch_20 == 0 and mismatch_50 == 0:
        print("\n" + "=" * 60)
        print("RESULT: PASS")
        print("UltraBull MA20 and MA50 match the independent calculation.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("RESULT: INVESTIGATE")
        print("Differences were detected.")
        print("=" * 60)


if __name__ == "__main__":
    main()