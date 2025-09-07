from stock_explorer import analyze_symbol, load_events, plot_candles_with_events
from vbt_to_pd_ind import build_ind_df_from_vbt, extract_symbol_df
from patterns.config import CFG
from patterns.pipeline import run_all_patterns

if __name__ == "__main__":
    symbol = "CAMS.NS"
    df, ind, trend = analyze_symbol(symbol, years=1)
    print(df.head())
    events = load_events(symbol, df)
    print(events)
    #plot_candles_with_events(df, ind, symbol, events)
    # print(ind)
    # print(ind.keys())
    print(df.columns)
    print(df.head())
    df_ohlcv = extract_symbol_df(df, symbol=symbol, case="title")
    ind_df = build_ind_df_from_vbt(df_ohlcv, ind)
    print(ind_df)
    print(ind_df.columns)
    print("======df_ohlcv========")
    print(df_ohlcv)
    print(df_ohlcv.columns)
    detections = run_all_patterns(df_ohlcv, ind_df, CFG)
    print(detections.tail(20))