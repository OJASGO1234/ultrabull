# core/fundamentals.py
"""
Fundamental snapshot + trend flags from Yahoo Finance statements.

Deliberately DISPLAY-ONLY: fundamentals are the user's research domain and a
universe-selection input, not a timing signal — nothing here feeds the gates.
The card exists so red flags (falling revenue, rising debt, margin compression)
are visible next to the technicals, and good ones (deleveraging, consistent
growth) too.

Coverage notes (NSE via yfinance): ~5 fiscal years of annual income statement +
balance sheet, ~5 quarters, current market cap / P/E / D-E / institutional
holding. Promoter holding is unreliable (often missing) and PLEDGING is not
available at all — both need an NSE shareholding-pattern or Screener-type feed;
fields exist in the payload so that source can drop in later.
"""
from __future__ import annotations
import math

import pandas as pd
import yfinance as yf

CR = 1e7  # ₹ crore


def _row(df: pd.DataFrame | None, name: str) -> pd.Series | None:
    if df is None or df.empty or name not in df.index:
        return None
    s = df.loc[name].dropna()
    return s if len(s) else None


def _year_points(s: pd.Series | None) -> list[dict]:
    """Oldest-first [{fy, value_cr}] from a newest-first yfinance row."""
    if s is None:
        return []
    out = [{"fy": str(ts.year), "value_cr": round(float(v) / CR, 1)} for ts, v in s.items()]
    return sorted(out, key=lambda x: x["fy"])


def _num(x):
    try:
        f = float(x)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def fetch_fundamentals(symbol: str) -> dict:
    t = yf.Ticker(symbol)
    out = {
        "available": False,
        "snapshot": {}, "annual": {}, "quarterly": {}, "flags": [],
        "promoter": {
            "holding_pct": None,          # yfinance 'heldPercentInsiders' when present
            "institutions_pct": None,
            "pledge_pct": None,           # NOT available via yfinance — needs NSE/Screener feed
            "history": [],                # same — quarterly shareholding pattern needed
            "source_note": "Promoter holding from Yahoo when available; pledging and "
                           "multi-year holding history need an NSE shareholding-pattern feed.",
        },
    }

    try:
        info = t.info or {}
    except Exception:
        info = {}
    mcap = _num(info.get("marketCap"))
    out["snapshot"] = {
        "market_cap_cr": round(mcap / CR, 0) if mcap else None,
        "trailing_pe": _num(info.get("trailingPE")),
        "price_to_book": _num(info.get("priceToBook")),
        "debt_to_equity": _num(info.get("debtToEquity")),  # percent, e.g. 52.9
        "dividend_yield_pct": (_num(info.get("dividendYield")) or 0) or None,
    }
    ins = _num(info.get("heldPercentInsiders"))
    inst = _num(info.get("heldPercentInstitutions"))
    out["promoter"]["holding_pct"] = round(ins * 100, 1) if ins is not None else None
    out["promoter"]["institutions_pct"] = round(inst * 100, 1) if inst is not None else None

    try:
        inc_a, bs_a = t.income_stmt, t.balance_sheet
        inc_q = t.quarterly_income_stmt
    except Exception:
        inc_a = bs_a = inc_q = None

    rev = _row(inc_a, "Total Revenue")
    ni = _row(inc_a, "Net Income")
    debt = _row(bs_a, "Total Debt")
    eq = _row(bs_a, "Stockholders Equity")

    out["annual"] = {
        "revenue": _year_points(rev),
        "net_income": _year_points(ni),
        "total_debt": _year_points(debt),
    }
    # net margin per FY where both exist
    if rev is not None and ni is not None:
        margins = []
        for ts in rev.index:
            if ts in ni.index and float(rev[ts]) > 0:
                margins.append({"fy": str(ts.year),
                                "value_pct": round(float(ni[ts]) / float(rev[ts]) * 100, 1)})
        out["annual"]["net_margin"] = sorted(margins, key=lambda x: x["fy"])

    # latest-quarter YoY when the same quarter last year is present
    q_rev = _row(inc_q, "Total Revenue")
    q_ni = _row(inc_q, "Net Income")
    def _qoq_yoy(s):
        if s is None or len(s) < 2:
            return None
        latest_ts = s.index[0]
        yoy = None
        for ts in s.index[1:]:
            if abs((latest_ts - ts).days - 365) <= 20:
                prev = float(s[ts])
                if prev != 0:
                    yoy = round((float(s[latest_ts]) / prev - 1) * 100, 1)
                break
        return {"quarter": str(latest_ts.date()), "value_cr": round(float(s[latest_ts]) / CR, 1), "yoy_pct": yoy}
    out["quarterly"] = {"revenue": _qoq_yoy(q_rev), "net_income": _qoq_yoy(q_ni)}

    # ---- flags ----
    flags: list[dict] = []

    def vals(points):
        return [p["value_cr"] for p in points]

    r = vals(out["annual"].get("revenue", []))
    if len(r) >= 3 and r[-1] < r[-2] < r[-3]:
        flags.append({"level": "bad", "text": "Revenue declining two consecutive years"})
    elif len(r) >= 2 and r[-1] < r[-2]:
        flags.append({"level": "warn", "text": f"Revenue fell {abs(r[-1]/r[-2]-1)*100:.0f}% in the latest FY"})
    elif len(r) >= 3 and r[-1] > r[-2] > r[-3]:
        flags.append({"level": "good", "text": "Revenue growing two consecutive years"})

    n = vals(out["annual"].get("net_income", []))
    if n and n[-1] < 0:
        flags.append({"level": "bad", "text": "Loss-making in the latest FY"})
    elif len(n) >= 3 and n[-1] < n[-2] < n[-3]:
        flags.append({"level": "bad", "text": "Profit declining two consecutive years"})
    elif len(n) >= 2 and n[-1] < n[-2]:
        flags.append({"level": "warn", "text": f"Profit fell {abs(n[-1]/n[-2]-1)*100:.0f}% in the latest FY"})
    elif len(n) >= 3 and n[-1] > n[-2] > n[-3]:
        flags.append({"level": "good", "text": "Profit growing two consecutive years"})

    m = [p["value_pct"] for p in out["annual"].get("net_margin", [])]
    if len(m) >= 2 and m[-1] < m[-2] - 3:
        flags.append({"level": "warn", "text": f"Net margin compressed {m[-2]:.1f}% → {m[-1]:.1f}%"})

    d = vals(out["annual"].get("total_debt", []))
    if len(d) >= 2 and d[-2] > 0:
        chg = d[-1] / d[-2] - 1
        if d[-1] < 1 or (mcap and d[-1] * CR < 0.02 * mcap):
            flags.append({"level": "good", "text": "Effectively debt-free"})
        elif chg <= -0.15:
            flags.append({"level": "good", "text": f"Debt reduced {abs(chg)*100:.0f}% in the latest FY"})
        elif chg >= 0.25:
            flags.append({"level": "warn", "text": f"Debt rose {chg*100:.0f}% in the latest FY"})
    de = out["snapshot"].get("debt_to_equity")
    if de is not None and de > 200:
        flags.append({"level": "bad", "text": f"High leverage: debt/equity ≈ {de/100:.1f}x"})

    qn = out["quarterly"].get("net_income")
    if qn and qn.get("yoy_pct") is not None:
        if qn["yoy_pct"] <= -25:
            flags.append({"level": "warn", "text": f"Latest quarter profit down {abs(qn['yoy_pct']):.0f}% YoY"})
        elif qn["yoy_pct"] >= 25:
            flags.append({"level": "good", "text": f"Latest quarter profit up {qn['yoy_pct']:.0f}% YoY"})

    if out["promoter"]["holding_pct"] is not None and out["promoter"]["holding_pct"] < 26:
        flags.append({"level": "warn",
                      "text": f"Promoter/insider holding only {out['promoter']['holding_pct']:.0f}% (low skin in the game)"})

    out["flags"] = flags
    out["available"] = bool(r or n or out["snapshot"].get("market_cap_cr"))
    return out
