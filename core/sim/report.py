from __future__ import annotations
from typing import Dict, Any, Optional
import json
import pandas as pd
import numpy as np

# ---------- small helpers ----------

def daily_pnl(equity: pd.Series) -> pd.DataFrame:
    """
    Return a tidy daily PnL table from an equity series indexed like df_ohlcv.
    Columns: ['date','equity','pnl','ret']
    """
    if equity is None or len(equity) == 0:
        return pd.DataFrame(columns=["date", "equity", "pnl", "ret"])
    eq = equity.ffill()
    pnl = eq.diff().fillna(0.0)
    ret = eq.pct_change().fillna(0.0)
    out = pd.DataFrame({"date": eq.index, "equity": eq.values, "pnl": pnl.values, "ret": ret.values})
    return out

def _as_frame(x, cols=None):
    if x is None:
        return pd.DataFrame(columns=cols or [])
    if isinstance(x, pd.Series):
        return x.to_frame()
    if isinstance(x, pd.DataFrame):
        return x
    return pd.DataFrame(x)

# ---------- exports ----------

def export_backtest_excel(
    path: str,
    trades: pd.DataFrame,
    equity: pd.Series,
    by_pattern: Optional[pd.DataFrame] = None,
    signals: Optional[pd.DataFrame] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Write a single Excel workbook with all key artifacts.
    Sheets:
      - Trades, Equity, DailyPnL, ByPattern, Signals (optional), Config (json)
    """
    trades_df = _as_frame(trades)
    equity_s = equity.copy() if equity is not None else pd.Series(dtype=float)
    byp = _as_frame(by_pattern)
    sig = _as_frame(signals)
    pnl_df = daily_pnl(equity_s)

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        # Trades
        if len(trades_df):
            t = trades_df.copy()
            # Stable column order if present
            cols = [c for c in [
                "entry_bar","exit_bar","side","entry_price","exit_price",
                "size_shares","sl_price","tp_price","sl_price_exit","tp_price_exit",
                "trail_price_exit","fees","tag","pnl","reason"
            ] if c in t.columns] + [c for c in t.columns if c not in {
                "entry_bar","exit_bar","side","entry_price","exit_price",
                "size_shares","sl_price","tp_price","sl_price_exit","tp_price_exit",
                "trail_price_exit","fees","tag","pnl","reason"
            }]
            t = t[cols]
            t.to_excel(xw, sheet_name="Trades", index=False)
        else:
            pd.DataFrame(columns=["no_trades"]).to_excel(xw, sheet_name="Trades", index=False)

        # Equity & Daily PnL
        if len(equity_s):
            pd.DataFrame({"date": equity_s.index, "equity": equity_s.values}).to_excel(xw, sheet_name="Equity", index=False)
            pnl_df.to_excel(xw, sheet_name="DailyPnL", index=False)
        else:
            pd.DataFrame(columns=["date","equity"]).to_excel(xw, sheet_name="Equity", index=False)
            pd.DataFrame(columns=["date","equity","pnl","ret"]).to_excel(xw, sheet_name="DailyPnL", index=False)

        # By-pattern
        (byp.reset_index().rename(columns={"index":"tag"}) if len(byp) else pd.DataFrame(columns=["tag"])) \
            .to_excel(xw, sheet_name="ByPattern", index=False)

        # Signals (optional; sometimes big)
        if len(sig):
            keep_cols = [c for c in ["bar_idx","side","sl_atr","tp_atr","trail_atr","tag","size_cash"] if c in sig.columns]
            meta_cols = [c for c in sig.columns if c not in keep_cols]
            s = sig[keep_cols + meta_cols]
            s.to_excel(xw, sheet_name="Signals", index=False)

        # Config as pretty JSON
        cfg_txt = json.dumps(cfg or {}, indent=2, default=str)
        pd.DataFrame({"config_json":[cfg_txt]}).to_excel(xw, sheet_name="Config", index=False)

    return path

# ---------- summaries ----------

def summarize_to_md(
    metrics: Dict[str, Any],
    by_pattern: pd.DataFrame,
    title: str = "Ultrabull Backtest",
    top_n: int = 6,
    notes: Optional[str] = None,
) -> str:
    """
    Build a compact Markdown summary for quick sharing.
    """
    m = metrics or {}
    lines = []
    lines.append(f"# {title}")
    if notes:
        lines.append(f"> {notes}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Trades: **{int(m.get('trades', 0))}**")
    lines.append(f"- Win rate: **{m.get('win_rate', float('nan')):.2%}**")
    pf = m.get("profit_factor", float("nan"))
    lines.append(f"- Profit factor: **{pf:.2f}**")
    lines.append(f"- Expectancy (₹/trade): **{m.get('expectancy', 0.0):,.0f}**")
    lines.append(f"- Total return: **{m.get('return_total', 0.0):.2%}**")
    lines.append(f"- Max drawdown: **{m.get('max_drawdown', 0.0):.2%}**")
    sharpe = m.get("sharpe_daily", float("nan"))
    if np.isfinite(sharpe):
        lines.append(f"- Sharpe (daily): **{sharpe:.2f}**")
    lines.append("")
    # Top/bottom patterns
    if by_pattern is not None and len(by_pattern):
        bp = by_pattern.reset_index().rename(columns={"index":"tag"})
        bp = bp.sort_values("total_pnl", ascending=False)
        top = bp.head(top_n)
        worst = bp.tail(min(top_n, len(bp)))
        lines.append("## Best Patterns")
        for r in top.itertuples(index=False):
            lines.append(f"- **{r.tag}** — pnl: {r.total_pnl:,.0f} | win: {r.win_rate:.0%} | pf: {r.profit_factor:.2f}")
        lines.append("")
        lines.append("## Worst Patterns")
        for r in worst.itertuples(index=False):
            lines.append(f"- **{r.tag}** — pnl: {r.total_pnl:,.0f} | win: {r.win_rate:.0%} | pf: {r.profit_factor:.2f}")
        lines.append("")
    return "\n".join(lines)
