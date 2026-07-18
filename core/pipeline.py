from stock_explorer import analyze_symbol, load_events, plot_candles_with_events
from vbt_to_pd_ind import build_ind_df_from_vbt, extract_symbol_df
from patterns.config import CFG
from patterns.pipeline import run_all_patterns
from patterns.postprocess import enrich_detections, warmup_cutoff
from patterns.viz import overlay_patterns, make_signal_addplots
from patterns.postprocess import summarize_signals
from patterns.explain import daily_summary_md, recent_summary_md
from rules.signals import compile_signals
from sim.engine import simulate
from sim.metrics import summarize, by_tag, audit_exits
from sim.viz import overlay_trades, overlay_fills, overlay_signals_rail
from sim.tune import try_presets
from rules.gates import apply_gates
from market_context import build_market_context
from sim.report import export_backtest_excel, summarize_to_md

import mplfinance as mpf

cfg = {
    "execution": {
        "delay_bars": 1,
        "enter_at": "Close",
        "max_open_positions": 1,
        "cooldown_bars":  int(CFG.get("cooldowns", {}).get("bars", 0)),  # from patterns.config
        "max_hold_bars": 0,
    },
    "risk": {  # used by the simulator
        "capital": 1_000_000.0,
        "max_risk_per_trade": 0.005,
        "position_sizing": "atr",   # or "fixed_cash"
        "fixed_cash_per_trade": 50_000.0,
    },
    "filters": {
        "min_atr": 0.0,
        "warmup_cutoff_bars": 60,
    },
    "patterns": {
        "Breakout20": {"side": "long",  "sl_atr": 1.5, "tp_atr": 3.0, "trail_atr": 1.0},
        "Breakout55": {"side": "long",  "sl_atr": 2.0, "tp_atr": 4.0, "trail_atr": 1.5},
        "Breakdown20": {"side": "short","sl_atr": 1.5, "tp_atr": 3.0, "trail_atr": 1.0},
        "Breakdown55": {"side": "short","sl_atr": 2.0, "tp_atr": 4.0, "trail_atr": 1.5},
        "PullbackToMA20": {"side": "long","sl_atr": 1.2, "tp_atr": 2.4, "trail_atr": 1.0},
        "BB_BuyDip": {"side": "long",  "sl_atr": 1.2, "tp_atr": 2.0, "trail_atr": 0.8},
        "BB_SellFade": {"side": "short","sl_atr": 1.5, "tp_atr": 3.0, "trail_atr": 1.0},
        "RSI_40_Up": {"side": "long",  "sl_atr": 1.5, "tp_atr": 3.0},
        "RSI_60_Down": {"side": "short","sl_atr": 1.5, "tp_atr": 3.0},
        "Hammer": {"side": "long", "sl_atr": 1.5, "tp_atr": 3.0},
        "BullishEngulfing": {"side": "long","sl_atr": 1.5, "tp_atr": 3.0},
        "BearishEngulfing": {"side": "short","sl_atr": 1.5, "tp_atr": 3.0},
        "Doji": {"side": "long","sl_atr": 1.2, "tp_atr": 2.0},
        "MACD_Cross_Zeroish_Down": {"side": "short", "sl_atr": 1.5, "tp_atr": 3.0},
    },
    "defaults": {"sl_atr": 1.5, "tp_atr": 3.0, "trail_atr": 0.0},
}


if __name__ == "__main__":
    symbol = "SWSOLAR.NS"
    symbol = "GANESHCO.NS"
    symbol = "HINDCOPPER.NS"
    df, ind, trend = analyze_symbol(symbol, years=0.5)
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

    # market/sector context first: enrichment folds relative strength into the score
    ctx = build_market_context(symbol, df_ohlcv["Close"], years=1.0)

    detections_enriched = enrich_detections(
        dets=detections,
        df_ohlcv=df_ohlcv,   # lowercase OHLCV is fine here
        ind_df=ind_df,
        symbol=symbol,
        ctx=ctx,
    )

    cut = warmup_cutoff(ind_df)
    detections_enriched = detections_enriched[detections_enriched["date"] >= cut]

    detections_enriched.to_csv("patterns_with_explanations.csv", index=False)
    watchlist = detections_enriched[detections_enriched["qualified"]].sort_values(["date","score"], ascending=[False, False])
    # print("======watchlist========")
    # print(watchlist)
    # print(watchlist.columns)
    # ax, idx = plot_candles_with_events(df_ohlcv, ind, symbol, events)
    # sig_ap = make_signal_addplots(idx, detections_enriched, panel=2,
    #                           only_qualified=True, min_score=60)
    # overlay_patterns(ax, detections_enriched, idx, only_qualified=True, min_score=70, top_per_day=1)
    # mpf.show()

    s = summarize_signals(detections_enriched)
    print({k:v for k,v in s.items() if k!="top_examples"})
    print(s["top_examples"])

    md = daily_summary_md(detections_enriched, df_ohlcv, ind_df, symbol)
    print(md)

    with open(f"{symbol}_daily_summary.md","w") as f: f.write(md)
    # print(recent_summary_md(detections_enriched, df_ohlcv, ind_df, symbol, days=5))

    md5 = recent_summary_md(detections_enriched, df_ohlcv, ind_df, symbol, days=5)
    with open(f"{symbol}_recent_5.md","w") as f: f.write(md5)


    cfg.setdefault("costs", {})
    cfg["costs"].update({
        "per_share": 0.0,     # e.g., 0.01 if you want per-share brokerage
        "per_trade": 0.0,     # fixed fee per fill
        "bps_fee": 0.0,       # commission in bps of notional (try 1–5 to start)
        "slippage_bps": 2.0,  # typical 1–5 bps; set 0 to disable
    })

    # Keep the time stop that helped
    cfg["execution"]["max_hold_bars"] = 10

    # Mean-reversion: take profits faster; a touch stricter on trailing
    cfg["patterns"]["BB_BuyDip"].update({"tp_atr": 1.6, "trail_atr": 1.0})
    cfg["patterns"]["BB_SellFade"].update({"tp_atr": 1.6, "trail_atr": 1.2})

    # gate knobs (optional)
    cfg.setdefault("gates", {})
    cfg["gates"]["bb_touch_eps"] = 0.0      # try 0.002 if you want slight slack
    cfg["gates"]["pullback_ma_tol"] = 0.01  # 1% within MA20

    cfg["execution"].update({"enter_at": "Open", "delay_bars": 1})

    # optionally sideline clearly weak patterns for now
    # cfg["disable_patterns"] = ["RSI_60_Down"]
    # Doji is indecision, not a tradeable direction.
    # RSI crosses, MACD_Cross_Zeroish_Up, BB_SellFade and both Breakdowns showed
    # zero/negative edge in the 35-symbol 4y event study, before AND after
    # gating (see core/event_study.py + core/evaluate_gates.py).
    # BearishEngulfing is the one short-side pattern that earns its keep.
    cfg["disable_patterns"] = [
        "RSI_60_Down", "RSI_40_Up", "Doji", "MACD_Cross_Zeroish_Up",
        "BB_SellFade", "Breakdown20", "Breakdown55", "VCP_Setup",
    ]

    cfg.setdefault("equity", {})
    cfg["equity"]["mtm"] = True
    cfg["equity"]["mtm_haircut_bps"] = 2.0   # optional: small adverse haircut; set 0 to disable



    # market/sector regime + relative-strength context (NIFTY + sector index)
    cfg["gates"]["market"] = {
        "enabled": True,
        "require_market_regime": True,
        "require_sector_regime": False,
        "rs_col": "rs_20",
        "breakout_rs_min": 0.0,
        "breakdown_rs_max": 0.0,
        "dist_days_max": 5.0,   # validated: pool excess @10 nearly doubles
        "breadth_min": 0.0,     # tested, redundant with NIFTY regime — display only
    }
    # no fresh entries just before/after earnings announcements
    cfg["gates"]["events"] = {
        "enabled": True,
        "earnings_blackout_before": 3,
        "earnings_blackout_after": 1,
    }

    # ADX gates evaluated and rejected (they select late, extended entries) —
    # available as knobs but off by default. Liquidity floor kept: tradability.
    cfg["gates"]["adx"] = {"breakout_min": 0.0, "meanrev_max": 0.0}
    cfg["gates"]["liquidity"] = {"min_turnover_20d": 2e7}  # ₹2 cr median daily turnover

    det_gated = apply_gates(detections_enriched, df_ohlcv, ind_df, cfg, ctx=ctx, events=events)
    signals = compile_signals(det_gated, df_ohlcv, ind_df, cfg)
    print(signals)


    trades, equity, fills = simulate(df_ohlcv, signals, ind_df, cfg)
    print(trades.head(15))
    print(equity.tail())
    print(fills.tail())

    report = summarize(trades, equity)
    print(report)

    print("\nPer-pattern:")
    by_pat = by_tag(trades)
    print(by_pat.round(2).head(15))

    print("\nAudit (SL/TP mismatches, if any):")
    print(audit_exits(trades, cfg))

    # out_path = "ultrabull_run.xlsx"
    # export_backtest_excel(out_path, trades, equity, by_pattern=by_pat, signals=signals, cfg=cfg)

    md = summarize_to_md(report, by_pat, title="Ultrabull Backtest — demo", notes="gate+slip on; RSI_60_Down pending disable")
    print(md)

    # Your existing candlestick plot call
    fig, axlist = mpf.plot(
        df_ohlcv,            # ['Open','High','Low','Close','Volume'] index matches df_ohlcv rows
        type='candle',
        volume=True,
        returnfig=True,
        style='yahoo',              # or your custom style
    )
    ax_price = axlist[0]
    overlay_trades(ax_price, trades)
    overlay_signals_rail(ax_price, signals)
    mpf.show()

    table = try_presets(detections_enriched, df_ohlcv, ind_df, cfg)
    # print(table)