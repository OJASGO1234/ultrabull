from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np

@dataclass
class Trade:
    entry_bar: int
    exit_bar: int
    side: str
    entry_price: float
    exit_price: float
    size_shares: float
    sl_price: float
    tp_price: float
    tag: str
    pnl: float
    reason: str
    sl_price_exit: float
    tp_price_exit: float
    trail_price_exit: float
    fees: float  # NEW

def _get(cfg: Dict[str, Any], path: str, default=None):
    cur = cfg
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur

def _entry_price(bars_open, bars_close, bar, enter_at: str) -> float:
    if enter_at == "open" and bar < len(bars_open):
        return float(bars_open[bar])
    return float(bars_close[min(bar, len(bars_close) - 1)])

def _size_shares(entry_px: float, atr: float, sl_atr_k: float, cfg: Dict[str, Any], equity: float) -> float:
    method = _get(cfg, "risk.position_sizing", "atr")
    if method == "fixed_cash":
        cash = float(_get(cfg, "risk.fixed_cash_per_trade", 50_000))
        return max(0.0, cash) / max(1e-9, entry_px)
    risk_cap = equity * float(_get(cfg, "risk.max_risk_per_trade", 0.005))
    stop_dist = max(1e-8, atr * max(1e-9, sl_atr_k))
    shares = risk_cap / stop_dist
    # cap notional at available equity: risk-based sizing on low-ATR names
    # can otherwise imply leverage the account doesn't have
    shares = min(shares, equity / max(1e-9, entry_px))
    return max(0.0, shares)

# ==== NEW: simple costs/slippage helpers ====
def _cost_cfg(cfg: Dict[str, Any]) -> Dict[str, float]:
    c = _get(cfg, "costs", {}) or {}
    return {
        "per_share": float(c.get("per_share", 0.0)),     # e.g., 0.01 currency / share
        "per_trade": float(c.get("per_trade", 0.0)),     # fixed per fill
        "bps_fee":   float(c.get("bps_fee", 0.0)),       # commission in bps of notional
        "slip_bps":  float(c.get("slippage_bps", 0.0)),  # slippage in bps (adverse)
    }

def _apply_slippage(px: float, side: str, is_entry: bool, slip_bps: float) -> float:
    """Adverse slippage: long pays higher on entry, sells lower on exit; short vice versa."""
    m = slip_bps / 10_000.0
    if side == "long":
        return px * (1 + m) if is_entry else px * (1 - m)
    else:  # short
        return px * (1 - m) if is_entry else px * (1 + m)

def _trade_fee(px: float, shares: float, costs: Dict[str, float]) -> float:
    notional = abs(px * shares)
    return costs["per_trade"] + costs["per_share"] * abs(shares) + (costs["bps_fee"] / 10_000.0) * notional
# ===========================================

def simulate(
    df_ohlcv: pd.DataFrame,
    signals: pd.DataFrame,
    ind_df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if len(df_ohlcv) == 0:
        return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame()

    close = df_ohlcv["Close"].to_numpy()
    open_ = df_ohlcv["Open"].to_numpy()
    high = df_ohlcv["High"].to_numpy()
    low = df_ohlcv["Low"].to_numpy()
    atr = ind_df["ATR"].to_numpy() if "ATR" in ind_df.columns else np.zeros_like(close)

    warmup = int(_get(cfg, "filters.warmup_cutoff_bars", 0) or 0)
    delay = int(_get(cfg, "execution.delay_bars", 1) or 0)
    enter_at = str(_get(cfg, "execution.enter_at", "Close")).lower()
    cooldown = int(_get(cfg, "execution.cooldown_bars", 0) or 0)
    max_hold = int(_get(cfg, "execution.max_hold_bars", 0) or 0)
    max_open = int(_get(cfg, "execution.max_open_positions", 1) or 1)

    costs = _cost_cfg(cfg)

    start_cap = float(_get(cfg, "risk.capital", 1_000_000))
    equity = start_cap

    #  MTM config
    mtm_on = bool(_get(cfg, "equity.mtm", False))
    mtm_haircut = float(_get(cfg, "equity.mtm_haircut_bps", 0.0)) / 10_000.0  # adverse haircut on mark

    equity_curve = np.full(len(close), np.nan, dtype=float)

    if signals is None or len(signals) == 0:
        cols = ["entry_bar","exit_bar","side","entry_price","exit_price",
                "size_shares","sl_price","tp_price","tag","pnl","reason",
                "sl_price_exit","tp_price_exit","trail_price_exit","fees"]
        return (
            pd.DataFrame(columns=cols),
            pd.Series(equity_curve, index=df_ohlcv.index).ffill().fillna(start_cap),
            pd.DataFrame(columns=["bar","price","type","tag"]),
        )

    sig = signals.sort_values("bar_idx").reset_index(drop=True)

    open_positions: List[dict] = []
    trades: List[dict] = []
    fills: List[dict] = []
    last_filled_bar = -10_000

    for t in range(len(close)):
        equity_curve[t] = equity

        # 1) exits
        still_open = []
        for pos in open_positions:
            side = pos["side"]
            trail_k = pos["trail_atr"]

            if side == "long":
                if trail_k > 0 and t > pos["entry_bar"]:
                    # trail off the PRIOR bar's close: the current close isn't
                    # knowable intrabar when the stop would actually trigger
                    pos["trail_price"] = max(pos.get("trail_price", -np.inf), close[t-1] - float(atr[t-1]) * trail_k)
                sl_eff = max(pos["sl_price"], pos.get("trail_price", -np.inf))
                tp_eff = pos["tp_price"]
                hit_sl = low[t] <= sl_eff
                hit_tp = high[t] >= tp_eff
                exit_px = None
                reason = None
                if hit_sl and hit_tp:
                    exit_px, reason = sl_eff, "SL"   # conservative
                elif hit_sl:
                    exit_px, reason = sl_eff, "SL"
                elif hit_tp:
                    exit_px, reason = tp_eff, "TP"
                elif max_hold and (t - pos["entry_bar"] >= max_hold):
                    exit_px, reason = close[t], "TIME"

                # gap handling: if the bar opened beyond the level, the real
                # fill is the open, not the level
                if reason == "SL" and float(open_[t]) < exit_px:
                    exit_px, reason = float(open_[t]), "SL_GAP"
                elif reason == "TP" and float(open_[t]) > exit_px:
                    exit_px, reason = float(open_[t]), "TP_GAP"

                if exit_px is not None:
                    # apply slippage + fees on exit
                    exit_px_adj = _apply_slippage(exit_px, side, is_entry=False, slip_bps=costs["slip_bps"])
                    fee_exit = _trade_fee(exit_px_adj, pos["shares"], costs)

                    pnl = (exit_px_adj - pos["entry_price"]) * pos["shares"] - (pos.get("fee_entry", 0.0) + fee_exit)
                    equity += pnl
                    trades.append({
                        "entry_bar": pos["entry_bar"], "exit_bar": t, "side": side,
                        "entry_price": pos["entry_price"], "exit_price": exit_px_adj,
                        "size_shares": pos["shares"], "sl_price": pos["sl_price"], "tp_price": pos["tp_price"],
                        "tag": pos["tag"], "pnl": pnl, "reason": reason,
                        "sl_price_exit": sl_eff if reason == "SL" else np.nan,
                        "tp_price_exit": tp_eff if reason == "TP" else np.nan,
                        "trail_price_exit": pos.get("trail_price", np.nan),
                        "fees": pos.get("fee_entry", 0.0) + fee_exit,
                    })
                    fills.append({"bar": t, "price": exit_px_adj, "type": f"EXIT_{reason}", "tag": pos["tag"]})
                else:
                    still_open.append(pos)

            else:  # short
                if trail_k > 0 and t > pos["entry_bar"]:
                    pos["trail_price"] = min(pos.get("trail_price", np.inf), close[t-1] + float(atr[t-1]) * trail_k)
                sl_eff = min(pos["sl_price"], pos.get("trail_price", np.inf))
                tp_eff = pos["tp_price"]
                hit_sl = high[t] >= sl_eff
                hit_tp = low[t] <= tp_eff
                exit_px = None
                reason = None
                if hit_sl and hit_tp:
                    exit_px, reason = sl_eff, "SL"
                elif hit_sl:
                    exit_px, reason = sl_eff, "SL"
                elif hit_tp:
                    exit_px, reason = tp_eff, "TP"
                elif max_hold and (t - pos["entry_bar"] >= max_hold):
                    exit_px, reason = close[t], "TIME"

                if reason == "SL" and float(open_[t]) > exit_px:
                    exit_px, reason = float(open_[t]), "SL_GAP"
                elif reason == "TP" and float(open_[t]) < exit_px:
                    exit_px, reason = float(open_[t]), "TP_GAP"

                if exit_px is not None:
                    exit_px_adj = _apply_slippage(exit_px, side, is_entry=False, slip_bps=costs["slip_bps"])
                    fee_exit = _trade_fee(exit_px_adj, pos["shares"], costs)

                    pnl = (pos["entry_price"] - exit_px_adj) * pos["shares"] - (pos.get("fee_entry", 0.0) + fee_exit)
                    equity += pnl
                    trades.append({
                        "entry_bar": pos["entry_bar"], "exit_bar": t, "side": side,
                        "entry_price": pos["entry_price"], "exit_price": exit_px_adj,
                        "size_shares": pos["shares"], "sl_price": pos["sl_price"], "tp_price": pos["tp_price"],
                        "tag": pos["tag"], "pnl": pnl, "reason": reason,
                        "sl_price_exit": sl_eff if reason == "SL" else np.nan,
                        "tp_price_exit": tp_eff if reason == "TP" else np.nan,
                        "trail_price_exit": pos.get("trail_price", np.nan),
                        "fees": pos.get("fee_entry", 0.0) + fee_exit,
                    })
                    fills.append({"bar": t, "price": exit_px_adj, "type": f"EXIT_{reason}", "tag": pos["tag"]})
                else:
                    still_open.append(pos)
        open_positions = still_open

        # 2) entries
        if len(open_positions) < max_open:
            todays = sig[sig["bar_idx"] == t]
            for _, s in todays.iterrows():
                if len(open_positions) >= max_open:
                    break
                if t < warmup:
                    continue
                if t - last_filled_bar < cooldown:
                    continue

                # NEW: per-signal overrides with fallback to cfg
                s_delay = int(s.get("delay_bars", _get(cfg, "execution.delay_bars", 1)))
                s_enter = str(s.get("enter_at", _get(cfg, "execution.enter_at", "open"))).lower()

                entry_bar = t + s_delay
                if entry_bar >= len(close):
                    continue

                raw_px = _entry_price(open_, close, entry_bar, s_enter)
                side = s["side"]
                atr_e = float(atr[entry_bar]) if len(atr) else 0.0

                # position sizing
                shares = _size_shares(raw_px, atr_e, float(s["sl_atr"]), cfg, equity)

                # compute SL/TP on raw price first
                if side == "long":
                    sl = raw_px - s["sl_atr"] * atr_e
                    tp = raw_px + s["tp_atr"] * atr_e
                else:
                    sl = raw_px + s["sl_atr"] * atr_e
                    tp = raw_px - s["tp_atr"] * atr_e

                # apply slippage and entry fee; store adjusted entry price
                entry_px = _apply_slippage(raw_px, side, is_entry=True, slip_bps=costs["slip_bps"])
                fee_entry = _trade_fee(entry_px, shares, costs)

                pos = {
                    "entry_bar": entry_bar,
                    "side": side,
                    "entry_price": entry_px,
                    "shares": shares,
                    "sl_price": sl,
                    "tp_price": tp,
                    "trail_atr": float(s.get("trail_atr", 0.0) or 0.0),
                    "tag": s.get("tag", ""),
                    "fee_entry": fee_entry,  # carry and charge on exit
                }
                open_positions.append(pos)
                last_filled_bar = entry_bar
                fills.append({"bar": entry_bar, "price": entry_px, "type": "ENTRY", "tag": pos["tag"]})

        eq_closed = equity
        if mtm_on and len(open_positions):
            # mark open positions to CLOSE[t], with slight adverse haircut
            unreal = 0.0
            px = float(close[t])
            for pos in open_positions:
                if pos["side"] == "long":
                    mark = px * (1.0 - mtm_haircut)
                    unreal += (mark - pos["entry_price"]) * pos["shares"]
                else:
                    mark = px * (1.0 + mtm_haircut)
                    unreal += (pos["entry_price"] - mark) * pos["shares"]
            equity_curve[t] = eq_closed + unreal
        else:
            equity_curve[t] = eq_closed

    trades_df = pd.DataFrame(trades)
    eq = pd.Series(equity_curve, index=df_ohlcv.index).ffill().fillna(start_cap)
    fills_df = pd.DataFrame(fills)
    return trades_df, eq, fills_df
