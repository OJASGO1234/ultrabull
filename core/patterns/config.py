# patterns/config.py
CFG = {
    "emit_min": 40,
    "cooldowns": {"bars": 3},
    "candles": {
        "engulfing_ratio": 1.05,
        "hammer_shadow_ratio": 2.0,
        "doji_body_pct": 0.10
    },
    "breakout": {"lookbacks": [20, 55], "vol_k": 1.5},
    "pullback": {"ma": 20, "atr_tol_k": 0.5},
    "momentum": {
        "rsi_levels": [40, 60],
        "macd_zero_eps_atr_pct": 0.25
    }
}
