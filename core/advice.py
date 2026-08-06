# core/advice.py
"""
Level-aware buy/sell advice for the CURRENT bar.

Combines, with fully transparent scoring:
  - position vs support/resistance (entry quality, chase risk, breakout)
  - regime gates (market, sector, relative strength, stock trend)
  - momentum (RSI band incl. the tested-negative >70 zone, MACD state)
  - volume (20d multiple, signed by the day's direction)
  - recent qualified pattern detections
  - earnings blackout (veto on fresh entries)

Outputs two verdicts — fresh money vs existing holding — plus a concrete plan
whose stop sits BELOW the nearest support (a level break proves the thesis
wrong) and whose target is the next resistance. Every factor is returned with
its contribution so the UI can show WHY.

Factors carry an `axis`:
  "setup"       — evidence about THIS stock (trend, RSI, MACD, volume, levels,
                  reward:risk, patterns, relative strength vs its sector)
  "environment" — evidence about the TAPE (NIFTY regime, sector regime,
                  breadth, distribution days)

and the two are scored separately, because blending them into one number lets
a hostile tape bury a good setup with no way to see that that is what happened.
`use_market_context=False` produces the stock-only view: environment factors are
still computed and returned (so the UI can show what is being ignored) but are
excluded from the score and cannot veto a verdict. That view is deliberately
NOT the default — see the distribution-day evidence in dist_day_study.py.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Factors describing the tape rather than the stock. Relative strength is
# deliberately NOT here: it measures this stock against its sector, which is a
# stock-selection statement, and it survived validation on its own.
ENV_FACTORS = {"Market regime", "Sector regime", "Distribution days", "Market breadth"}
# Vetoes carry a sentinel score and belong to neither axis — they are not
# evidence about the stock, they are a reason not to act at all.
VETO_FACTORS = {"Earnings blackout"}


def _factor(name: str, score: float, detail: str) -> dict:
    return {
        "name": name,
        "score": round(score, 2),
        "detail": detail,
        "axis": ("veto" if name in VETO_FACTORS
                 else "environment" if name in ENV_FACTORS else "setup"),
    }


def build_advice(
    df_ohlcv: pd.DataFrame,
    ind_df: pd.DataFrame,
    ctx: pd.DataFrame | None,
    levels: list[dict] | None,
    timing: dict | None,
    detections: pd.DataFrame | None,
    use_market_context: bool = True,
) -> dict:
    close = float(df_ohlcv["Close"].iloc[-1])
    atr = float(ind_df["ATR"].iloc[-1]) if np.isfinite(ind_df["ATR"].iloc[-1]) else close * 0.02
    rsi = float(ind_df["RSI"].iloc[-1]) if np.isfinite(ind_df["RSI"].iloc[-1]) else 50.0
    macd_up = bool(ind_df["MACD"].iloc[-1] > ind_df["MACD_signal"].iloc[-1]) \
        if "MACD" in ind_df.columns else None
    trend_up = bool(ind_df["MA20"].iloc[-1] > ind_df["MA50"].iloc[-1])

    vol = df_ohlcv["Volume"]
    volx = float(vol.iloc[-1] / max(1e-9, vol.rolling(20).mean().iloc[-1]))
    up_day = bool(df_ohlcv["Close"].iloc[-1] >= df_ohlcv["Close"].iloc[-2]) if len(df_ohlcv) > 1 else True

    mkt_up = sect_up = None
    rs20 = breadth = dist_days = None
    if ctx is not None and len(ctx):
        last = ctx.iloc[-1]
        mkt_up = bool(last.get("mkt_up"))
        sect_up = bool(last.get("sect_up"))
        rs20 = float(last["rs_20"]) if np.isfinite(last.get("rs_20", np.nan)) else None
        breadth = float(last["breadth"]) if np.isfinite(last.get("breadth", np.nan)) else None
        dist_days = float(last["dist_days"]) if np.isfinite(last.get("dist_days", np.nan)) else None

    supports = sorted([lv for lv in (levels or []) if lv["kind"] == "support" and lv["price"] <= close],
                      key=lambda lv: -lv["price"])
    resistances = sorted([lv for lv in (levels or []) if lv["kind"] == "resistance" and lv["price"] >= close],
                         key=lambda lv: lv["price"])
    sup = supports[0] if supports else None
    res = resistances[0] if resistances else None
    res2 = resistances[1] if len(resistances) > 1 else None

    blackout = bool((timing or {}).get("in_earnings_blackout"))

    factors: list[dict] = []

    # --- regime ---
    if mkt_up is not None:
        factors.append(_factor("Market regime", 1.0 if mkt_up else -1.5,
                               "NIFTY uptrend supports longs" if mkt_up
                               else "NIFTY downtrend — the engine gates all fresh longs off here"))
    if sect_up is not None:
        factors.append(_factor("Sector regime", 0.5 if sect_up else -0.5,
                               "sector index trending up" if sect_up else "sector index trending down"))
    if rs20 is not None:
        factors.append(_factor("Relative strength", 1.0 if rs20 >= 0 else -1.0,
                               f"{'leading' if rs20 >= 0 else 'lagging'} its sector by {abs(rs20) * 100:.1f}% over 20d"))
    factors.append(_factor("Stock trend", 1.0 if trend_up else -1.0,
                           "MA20 above MA50" if trend_up else "MA20 below MA50"))

    # --- market health (validated: dist-day ceiling doubled pool edge @10 bars) ---
    if dist_days is not None:
        if dist_days >= 5:
            factors.append(_factor("Distribution days", -1.5,
                                   f"NIFTY fell on higher volume than the day before {int(dist_days)} times in the last "
                                   f"25 sessions — the footprint of large funds selling (they can't exit quietly). At 5+, "
                                   f"our 4y test showed new buying does badly, so the system pauses fresh buys until it clears "
                                   f"(Hammer reversals are exempt — they fire best in exactly this kind of tape)"))
        elif dist_days >= 4:
            factors.append(_factor("Distribution days", -0.5,
                                   f"{int(dist_days)} of the last 25 sessions saw the index fall on rising volume "
                                   f"(a sign of big-fund selling) — one more and the system pauses new buys"))
        else:
            factors.append(_factor("Distribution days", 0.0,
                                   f"only {int(dist_days)} index-falls-on-rising-volume in 25 sessions — no sign of heavy fund selling"))
    if breadth is not None:
        factors.append(_factor(
            "Market breadth",
            0.5 if breadth >= 0.55 else (-0.5 if breadth <= 0.35 else 0.0),
            f"{breadth:.0%} of the stocks we track are above their own 50-day average — "
            + ("the rally has broad participation, not just a few heavyweights" if breadth >= 0.55
               else "the index is being held up by a handful of big names, which is fragile" if breadth <= 0.35
               else "neither broad nor narrow")))

    # --- momentum ---
    if rsi > 70:
        factors.append(_factor("RSI", -1.0,
                               f"RSI {rsi:.0f} — overbought; fresh longs here tested NEGATIVE (−16bps/5d) in our study"))
    elif 40 <= rsi <= 65:
        factors.append(_factor("RSI", 1.0, f"RSI {rsi:.0f} — momentum without overheat (the one RSI zone with tested edge)"))
    elif rsi < 35:
        factors.append(_factor("RSI", -0.5, f"RSI {rsi:.0f} — washed out; wait for a reversal bar before catching it"))
    else:
        factors.append(_factor("RSI", 0.0, f"RSI {rsi:.0f} — neutral zone"))
    if macd_up is not None:
        factors.append(_factor("MACD", 0.5 if macd_up else -0.5,
                               "MACD above signal line" if macd_up else "MACD below signal line"))

    # --- volume ---
    if volx >= 1.5:
        factors.append(_factor("Volume", 0.5 if up_day else -0.5,
                               f"{volx:.1f}× 20d volume on an {'up' if up_day else 'down'} day"))
    else:
        factors.append(_factor("Volume", 0.0, f"{volx:.1f}× 20d average — unremarkable"))

    # --- position vs levels ---
    breakout = res is None or close > res["price"]
    if breakout and res is None and supports:
        factors.append(_factor("Levels", 0.5, "above every mapped resistance (blue-sky region near 52w high)"))
    elif res is not None:
        d_sup = (close - sup["price"]) / atr if sup else None
        d_res = (res["price"] - close) / atr
        if sup and d_sup is not None and d_sup <= 1.0:
            factors.append(_factor("Levels", 1.0,
                                   f"sitting on support ₹{sup['price']:.0f} ({sup['touches']} touches) — defined-risk entry zone"))
        elif d_res <= 0.75:
            factors.append(_factor("Levels", -1.0,
                                   f"pressed against resistance ₹{res['price']:.0f} ({res['touches']} touches) — chasing here buys someone's exit; wait for a close above it or a dip"))
        else:
            factors.append(_factor("Levels", 0.0,
                                   f"mid-range between ₹{sup['price']:.0f} and ₹{res['price']:.0f}" if sup
                                   else f"below resistance ₹{res['price']:.0f}, no mapped support beneath"))

    # reward:risk from the levels themselves
    rr = None
    if sup and res:
        risk = close - (sup["price"] - 0.5 * atr)
        reward = res["price"] - close
        if risk > 0:
            rr = reward / risk
            factors.append(_factor("Reward:risk", 0.5 if rr >= 1.5 else (-0.5 if rr < 1.0 else 0.0),
                                   f"{rr:.1f}:1 to next resistance vs stop below support"))

    # --- recent patterns ---
    if detections is not None and len(detections):
        d = detections[detections.get("qualified", False)]
        if len(d):
            recent = d[pd.to_datetime(d["date"]) >= df_ohlcv.index[-4]] if len(df_ohlcv) > 4 else d
            bulls = int((recent["direction"] == "bullish").sum())
            bears = int((recent["direction"] == "bearish").sum())
            if bulls > bears:
                pats = ", ".join(recent.loc[recent["direction"] == "bullish", "pattern"].unique()[:3])
                factors.append(_factor("Recent patterns", 1.0, f"fresh bullish setups: {pats}"))
            elif bears > bulls:
                pats = ", ".join(recent.loc[recent["direction"] == "bearish", "pattern"].unique()[:3])
                factors.append(_factor("Recent patterns", -1.0, f"fresh bearish setups: {pats}"))

    # --- earnings blackout (veto) ---
    if blackout:
        factors.append(_factor("Earnings blackout", -99,
                               "results imminent — gap risk dwarfs any setup edge; no fresh entries"))

    # --- two axes: this stock vs this tape ---
    # The earnings-blackout veto carries a sentinel score (-99); it is a veto,
    # not a weight, so it is excluded from both sums.
    weighted = [f for f in factors if f["score"] > -90]
    setup_score = float(sum(f["score"] for f in weighted if f["axis"] == "setup"))
    env_score = float(sum(f["score"] for f in weighted if f["axis"] == "environment"))
    if not use_market_context:
        for f in factors:
            if f["axis"] == "environment":
                f["ignored"] = True
    # Thresholds below are the SAME in both modes, and that makes stock-only
    # materially MORE PERMISSIVE — state it plainly rather than dressing it up.
    # The environment axis spans -4.0 … +2.0, so dropping it removes twice as
    # much drag as lift. Measured over every reachable (setup, env) pair: when
    # dist_days >= 5 the fresh verdict differs in 55% of cases and is more
    # bullish in 49%; market-aware cannot emit BUY or BUY_ON_DIP AT ALL in that
    # regime, because dist_pause intercepts first. The toggle is therefore close
    # to binary in the current tape, not a matter of degree.
    #
    # It is left permissive on purpose: the whole point of the view is to see
    # the stock with the tape removed, and re-tightening the bar would just
    # reintroduce the market through the back door. The honesty burden sits on
    # the UI copy instead (api/main.py::stock_only_cfg -> views.stock_only.warning).
    score = setup_score + env_score if use_market_context else setup_score

    # --- verdicts ---
    # Entry-price quality vetoes: a high score means the BACKDROP is right, but
    # buying overbought into resistance (or against a hit distribution-day
    # ceiling) tested negative — the fresh-money verdict must not contradict
    # the very factors that flagged it (or the holding-side trim call).
    pressed = res is not None and (res["price"] - close) / atr <= 0.75
    overbought = rsi > 70
    # Re-checked Jul 2026 (dist_day_study.py) and left EXACTLY as shipped: over
    # the last 12 months below-vs-above separation is directionally right at
    # 5-10 bars (+74/+167 bps, one-sided block-bootstrap p 0.07/0.06 after
    # correcting for overlapping, date-clustered events) but is absent in the
    # prior 3 years (t~0.1) and REVERSES at 1 bar. That is too weak to move the
    # constant in EITHER direction, so it stays a veto by inertia, not by fresh
    # proof — and only in the market-aware view.
    # It binds hard right now (85.8% of the last 120 sessions vs 50.9% across
    # 4 years) for two reasons that must not be conflated: the threshold sits
    # at the 4-year median (5.0), AND the count has genuinely shifted right
    # lately (median 6.0, mean 5.83 over the last 120 sessions vs 4.77 over 4y).
    # So it is partly an artifact of where the constant sits and partly a real
    # regime change — neither alone explains it.
    dist_pause = use_market_context and dist_days is not None and dist_days >= 5

    if blackout:
        fresh = {"verdict": "WAIT", "reason": "earnings blackout — re-assess after results"}
    elif dist_pause and score >= 1.5:
        fresh = {"verdict": "WAIT",
                 "reason": "the stock looks fine, but the index hit the distribution-day ceiling — "
                           "the engine pauses ALL fresh buys until that selling pressure clears"}
    elif score >= 3.5 and (overbought or pressed):
        why = []
        if overbought:
            why.append(f"RSI {rsi:.0f} is overbought (fresh longs here tested negative)")
        if pressed:
            why.append(f"price is pressed against resistance ₹{res['price']:.0f}")
        fresh = {"verdict": "BUY_ON_DIP",
                 "reason": "strong backdrop, but " + " and ".join(why)
                           + (f" — better entry near support ₹{sup['price']:.0f}" if sup
                              else " — wait for a pullback")}
    elif score >= 3.5:
        fresh = {"verdict": "BUY", "reason": "regime, momentum and level position all align"}
    elif score >= 1.5:
        fresh = ({"verdict": "BUY", "reason": "constructive backdrop and price is already at support"}
                 if sup and (close - sup["price"]) / atr <= 1.0
                 else {"verdict": "BUY_ON_DIP",
                       "reason": f"constructive, but entry is better near support ₹{sup['price']:.0f}" if sup
                       else "constructive, but wait for a defined-risk entry"})
    elif score > -1.5:
        fresh = {"verdict": "WAIT", "reason": "signals are mixed — no defined edge at this price"}
    else:
        fresh = {"verdict": "AVOID", "reason": "regime and momentum are against fresh longs"}

    if score <= -3.0:
        holding = {"verdict": "EXIT_OR_REDUCE", "reason": "multiple factors negative — protect capital"}
    elif score <= -1.0:
        holding = {"verdict": "TIGHTEN_STOP",
                   "reason": f"deteriorating — raise stop to just below support ₹{sup['price']:.0f}" if sup
                   else "deteriorating — tighten stop to ~2 ATR below price"}
    elif overbought and pressed:
        holding = {"verdict": "CONSIDER_TRIM", "reason": "overbought into resistance — a partial profit-take is reasonable"}
    else:
        holding = {"verdict": "HOLD", "reason": "no exit trigger; let the position work with the stop in place"}

    # --- what is holding this back, and what would clear it ---
    # Only meaningful in the market-aware view, and ONLY when the tape is
    # actually holding the verdict back: a blocker listed under a BUY tile
    # (reachable whenever the market drags without changing the verdict) reads
    # as a contradiction. Gate on the verdict, not on setup_score alone.
    #
    # Breadth is deliberately NOT a blocker: it is a -0.5 score factor and
    # nothing more (breadth_min is 0.0 in every config, and the >=45% gate was
    # tested and rejected — see CLAUDE.md). Giving it a `clears_when` would
    # advertise a gate that does not exist.
    blockers: list[dict] = []
    held_back = fresh["verdict"] in ("WAIT", "AVOID", "BUY_ON_DIP")
    if use_market_context and held_back and setup_score >= 1.5:
        if dist_pause:
            blockers.append({
                "name": "Distribution days",
                "detail": f"{int(dist_days)} index-falls-on-rising-volume in the last 25 sessions "
                          f"(ceiling is 5) — fresh buys are paused market-wide, except Hammer reversals",
                "clears_when": "the count falls below 5",
            })
        if mkt_up is False and env_score < 0:
            blockers.append({
                "name": "Market regime",
                "detail": "NIFTY's 20-day average is below its 50-day — the pattern engine gates fresh "
                          "long signals off here (Hammer excepted), and this card weights it heavily "
                          "against fresh buys",
                "clears_when": "NIFTY's 20-day average crosses back above its 50-day",
            })

    # --- plan (levels first, ATR fallback) ---
    stop = round(sup["price"] - 0.5 * atr, 2) if sup else round(close - 2.0 * atr, 2)
    plan = {
        "entry_zone_low": round(sup["price"], 2) if sup else round(close - 0.5 * atr, 2),
        "entry_zone_high": round(min(close, sup["price"] + 0.75 * atr), 2) if sup else round(close, 2),
        "stop": stop,
        "stop_basis": "0.5×ATR below nearest support — a close below that level breaks the setup"
                      if sup else "2×ATR below price (no mapped support beneath)",
        "target1": round(res["price"], 2) if res else round(close + 2.0 * atr, 2),
        "target2": round(res2["price"], 2) if res2 else None,
        "reward_risk": round(rr, 2) if rr else None,
    }

    # The target is where the KNOWN math ends, not a ceiling — spell out the
    # conditional path so a target sitting at resistance reads sequentially:
    # clear the level, the level becomes support, the objective moves up.
    if res is not None:
        jump = round(res2["price"], 2) if res2 else round(res["price"] + 2.0 * atr, 2)
        stop_after = round(res["price"] - 0.5 * atr, 2)
        plan["if_breakout"] = {
            "trigger": round(res["price"], 2),
            "target": jump,
            "target_basis": "next mapped resistance" if res2
                            else "2×ATR extension (no higher level on the map)",
            "stop_after": stop_after,
            "note": (f"a close above ₹{res['price']:.0f} flips that level into support — raise the stop to "
                     f"₹{stop_after:.0f} and the objective moves to ₹{jump:.0f} "
                     + ("(the next mapped resistance)" if res2
                        else "(a 2×ATR extension; no higher level on the map)")
                     + "; re-analyse after that close"),
        }
    else:
        plan["if_breakout"] = None

    return {
        "as_of": str(df_ohlcv.index[-1].date()),
        "close": round(close, 2),
        "score": round(score, 2),
        "setup_score": round(setup_score, 2),
        "env_score": round(env_score, 2),
        "market_context_applied": bool(use_market_context),
        "blockers": blockers,
        "fresh": fresh,
        "holding": holding,
        "factors": factors,
        "plan": plan,
        "note": ("Rule-based synthesis of the layers above — factor weights are fixed and shown, "
                 "not fitted. It frames risk; it does not predict. Confirm with your fundamental view."
                 + ("" if use_market_context else
                    " STOCK-ONLY VIEW: market-wide factors (NIFTY regime, sector regime, breadth, "
                    "distribution days) are shown below but excluded from the score and cannot veto "
                    "a verdict. Those filters are validated — ignoring them is a deliberate override, "
                    "not a second opinion of equal standing. Note this cuts BOTH ways: it removes up "
                    "to 4.0 points of drag against only 2.0 of lift, so it makes fresh-buy verdicts "
                    "more permissive AND softens the holding-side exit call — a market-aware "
                    "'exit / reduce' can read as 'tighten stop' or 'hold' here.")),
    }
