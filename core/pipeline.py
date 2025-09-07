from stock_explorer import analyze_symbol, load_events, plot_candles_with_events
from vbt_to_pd_ind import build_ind_df_from_vbt, extract_symbol_df
from patterns.config import CFG
from patterns.pipeline import run_all_patterns
from patterns.postprocess import enrich_detections
from patterns.viz import overlay_patterns, make_signal_addplots
from patterns.postprocess import summarize_signals
from patterns.explain import daily_summary_md, recent_summary_md


import mplfinance as mpf

if __name__ == "__main__":
    symbol = "SHALBY.NS"
    df, ind, trend = analyze_symbol(symbol, years=1)
    print(df.head())
    events = load_events(symbol, df)
    print(events)
    # plot_candles_with_events(df, ind, symbol, events, show=True)
    # print(ind)
    # print(ind.keys())
    # print(df.columns)
    # print(df.head())
    df_ohlcv = extract_symbol_df(df, symbol=symbol, case="title")
    ind_df = build_ind_df_from_vbt(df_ohlcv, ind)
    # print(ind_df)
    # print(ind_df.columns)
    print("======df_ohlcv========")
    print(df_ohlcv)
    print(df_ohlcv.columns)
    detections = run_all_patterns(df_ohlcv, ind_df, CFG)
    print(detections.tail(10))

    detections_enriched = enrich_detections(
        dets=detections,
        df_ohlcv=df_ohlcv,   # lowercase OHLCV is fine here
        ind_df=ind_df,
        symbol=symbol
    )

    detections_enriched.to_csv("patterns_with_explanations.csv", index=False)
    watchlist = detections_enriched[detections_enriched["qualified"]].sort_values(["date","score"], ascending=[False, False])
    # print("======watchlist========")
    # print(watchlist)
    # print(watchlist.columns)
    ax, idx = plot_candles_with_events(df_ohlcv, ind, symbol, events)
    sig_ap = make_signal_addplots(idx, detections_enriched, panel=2,
                              only_qualified=True, min_score=60)
    # overlay_patterns(
    #     ax, detections_enriched, idx,
    #     only_qualified=True,   # show only watchlist-worthy
    #     min_score=60,          # bump to 70 if still busy
    #     families={"breakout","pullback"},  # or {"momentum","candles"} etc.
    #     top_per_day=1,         # at most one marker per day
    #     y_pos="top",           # or "bottom"
    #     draw_lines=True        # set False to show labels only (no dashed lines)
    # )
    overlay_patterns(ax, detections_enriched, idx, only_qualified=True, min_score=70, top_per_day=1)
    mpf.show()

    s = summarize_signals(detections_enriched)
    print({k:v for k,v in s.items() if k!="top_examples"})
    print(s["top_examples"])


    # print(daily_summary_md(detections_enriched, df_ohlcv, ind_df, symbol))
    print(recent_summary_md(detections_enriched, df_ohlcv, ind_df, symbol, days=5))
