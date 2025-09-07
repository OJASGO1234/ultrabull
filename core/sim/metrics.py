from __future__ import annotations
import pandas as pd
import numpy as np

def _max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min()) if len(dd) else 0.0

def _maybe_sharpe(equity: pd.Series) -> float:
    rets = equity.pct_change().dropna()
    if len(rets) < 2:
        return float("nan")
    return float((rets.mean() / (rets.std() + 1e-12)) * np.sqrt(252))

def summarize(trades: pd.DataFrame, equity: pd.Series) -> dict:
    out = {}
    out["trades"] = int(len(trades))
    if len(trades):
        wins = trades["pnl"] > 0
        out["win_rate"] = float(wins.mean())
        out["avg_win"] = float(trades.loc[wins, "pnl"].mean()) if wins.any() else 0.0
        out["avg_loss"] = float(trades.loc[~wins, "pnl"].mean()) if (~wins).any() else 0.0
        gp = float(trades.loc[wins, "pnl"].sum()) if wins.any() else 0.0
        gl = -float(trades.loc[~wins, "pnl"].sum()) if (~wins).any() else 0.0
        out["profit_factor"] = (gp / gl) if gl > 0 else float("inf")
        out["expectancy"] = float(trades["pnl"].mean())
        out["median_hold_bars"] = float((trades["exit_bar"] - trades["entry_bar"]).median())
    if len(equity):
        out["return_total"] = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        out["max_drawdown"] = _max_drawdown(equity)
        out["sharpe_daily"] = _maybe_sharpe(equity)
    return out

def by_tag(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or len(trades) == 0:
        cols = ["trades","win_rate","total_pnl","avg_pnl","profit_factor"]
        return pd.DataFrame(columns=cols)
    g = trades.groupby("tag", dropna=False)
    wins = trades.assign(win=trades["pnl"] > 0).groupby("tag", dropna=False)
    gp = trades.assign(gp=trades["pnl"].clip(lower=0.0)).groupby("tag", dropna=False)["gp"].sum()
    gl = trades.assign(gl=(-trades["pnl"]).clip(lower=0.0)).groupby("tag", dropna=False)["gl"].sum()  # <-- no leading minus
    out = pd.DataFrame({
        "trades": g.size(),
        "win_rate": wins["win"].mean(),
        "total_pnl": g["pnl"].sum(),
        "avg_pnl": g["pnl"].mean(),
    })
    out["profit_factor"] = np.where(gl > 0, gp / gl, np.inf)
    return out.sort_values(["total_pnl","win_rate"], ascending=[False, False])

def audit_exits(trades: pd.DataFrame, cfg: dict | None = None, eps_abs: float = 1e-3, eps_rel: float = 5e-4) -> pd.DataFrame:
    """
    Verify that exit_price matches the expected SL/TP level *including slippage* (if configured).
    - If cfg is None or no slippage is set, falls back to exact-level checks.
    - eps_abs / eps_rel provide absolute & relative tolerances.
    """
    if trades is None or len(trades) == 0:
        return trades

    slip_bps = 0.0
    if cfg:
        c = (cfg.get("costs") or {})
        slip_bps = float(c.get("slippage_bps", 0.0) or 0.0)
    m = slip_bps / 10_000.0  # slippage multiplier

    t = trades.copy()
    mism = []

    for r in t.itertuples(index=False):
        if r.reason not in ("SL", "TP"):
            mism.append(False); continue

        # Prefer exit-specific fields if present
        slx = getattr(r, "sl_price_exit", np.nan)
        tpx = getattr(r, "tp_price_exit", np.nan)

        if r.reason == "SL":
            ref = slx if np.isfinite(slx) else r.sl_price
            if not np.isfinite(ref):
                mism.append(True); continue
            # adverse slippage
            ref_slip = ref * (1 - m) if r.side == "long" else ref * (1 + m)

        else:  # TP
            ref = tpx if np.isfinite(tpx) else r.tp_price
            if not np.isfinite(ref):
                mism.append(True); continue
            # adverse slippage (opposite sign of SL)
            ref_slip = ref * (1 + m) if r.side == "long" else ref * (1 - m)

        tol = max(eps_abs, abs(ref_slip) * eps_rel)
        mism.append(abs(r.exit_price - ref_slip) > tol)

    t["audit_mismatch"] = mism
    return t[t["audit_mismatch"]]

