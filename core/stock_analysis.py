# core/stock_analysis.py
"""
Thin adapter only. This module does NOT compute anything new — it imports
no indicator/pattern/gate/signal/backtest logic and calls none of it. It
only takes the already-computed JSON-serializable pieces api/main.py's
run_analysis() builds (fundamentals, indicators, detections, market
context, signals, advice, outlook, levels, metrics) and gathers them into
one typed, named structure: StockAnalysis.

Purpose: give the forthcoming AI recommendation layer a single, stable
input object instead of reading ~10 loose local variables out of
run_analysis(). Nothing here changes what any existing module computes;
indicators.py/patterns/gates/signals.py/advice.py/outlook.py/
fundamentals.py are all untouched and unimported except for type hints in
comments.

build_stock_analysis() is called at the very END of run_analysis(), after
every existing field is already computed exactly as before. Its output is
added as ONE NEW key ("stock_analysis") in the dict run_analysis() returns
— every existing top-level key (advice, outlook, views, detections,
signals, ...) stays exactly as it is today, so this is purely additive
and backward compatible: no existing API consumer's response shape
changes.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class StockAnalysis:
    """Unified, read-only view of one symbol's analysis run.

    Every field here is exactly the already-serialized JSON value
    run_analysis() produces for that piece today (same dicts/lists, not a
    re-derivation) — this dataclass only gives those pieces one shared,
    named container and a couple of convenience accessors.
    """
    symbol: str
    as_of: Optional[str]                    # last bar date, e.g. ind_df's last_date
    fundamentals: Optional[dict]            # fetch_fundamentals() output, as-is (display-only today)
    indicators: dict                        # the {"ma20":..., "rsi":..., "adx":..., ...} series dict
    key_indicators: Optional[dict]          # build_key_indicators() output (latest-value snapshot)
    patterns: list                          # detections_json: one row per detected pattern
    market_context: Optional[dict]          # context_json: regime/breadth/distribution-days/RS snapshot
    signals: list                           # signals_json: gated, compiled trade signals
    advice: Optional[dict]                  # build_advice() output (includes "plan": stop/target/side)
    outlook: Optional[dict]                 # build_outlook() output (p_up per horizon, base rates)
    levels: list                            # support/resistance levels (find_levels() output)
    metrics: Optional[dict]                 # backtest summary metrics for this symbol's own history
    extras: dict = field(default_factory=dict)  # anything not yet promoted to a named field

    @property
    def plan(self) -> Optional[dict]:
        """Shortcut to advice["plan"] (side/sl_est/tp_est/ref_close), if present."""
        if isinstance(self.advice, dict):
            return self.advice.get("plan")
        return None

    @property
    def latest_pattern(self) -> Optional[dict]:
        """Most recent detected pattern row, if any (patterns is already date-ordered
        the same way detections_json is built in run_analysis())."""
        return self.patterns[-1] if self.patterns else None

    def to_dict(self) -> dict:
        return asdict(self)


def build_stock_analysis(
    *,
    symbol: str,
    as_of: str | None,
    fundamentals_json: dict | None,
    indicators_json: dict,
    key_indicators_json: dict | None,
    detections_json: list,
    context_json: dict | None,
    signals_json: list,
    advice_json: dict | None,
    outlook_json: dict | None,
    levels_json: list,
    metrics_json: dict | None,
    extras: dict | None = None,
) -> StockAnalysis:
    """Assemble a StockAnalysis from already-computed run_analysis() pieces.

    Every argument is expected to already be the exact JSON-serializable
    value run_analysis() built for that field — this function performs no
    computation of its own, only gathers them into the dataclass above.
    """
    return StockAnalysis(
        symbol=symbol,
        as_of=as_of,
        fundamentals=fundamentals_json,
        indicators=indicators_json,
        key_indicators=key_indicators_json,
        patterns=detections_json or [],
        market_context=context_json,
        signals=signals_json or [],
        advice=advice_json,
        outlook=outlook_json,
        levels=levels_json or [],
        metrics=metrics_json,
        extras=extras or {},
    )
