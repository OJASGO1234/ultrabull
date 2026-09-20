# core/strategies/momentum_strategy.py
"""
Daily absolute + relative momentum strategy (cross-sectional, multi-period).

This module implements the strategy the mentor asked to "try out": for a
universe of stocks, rank names by multi-month price momentum relative to a
benchmark, confirm with existing trend indicators, and trade the strongest
names through the SAME simulator and risk engine the existing pattern-based
strategy already uses. It is a NEW, independently-configurable strategy that
coexists with core/patterns/ + core/rules/ + core/advice.py -- it does not
modify, gate, or replace any of them.

Pipeline (mirrors the brief exactly):

    Daily OHLCV (stock_explorer.load_data, cached to disk for research)
        -> Momentum (core.momentum.absolute_momentum, per lookback)
        -> Relative strength (core.momentum.relative_momentum vs NIFTY)
        -> Cross-sectional ranking (rank across the universe, PER DATE)
        -> Technical confirmation (core.momentum.technical_confirmation,
           reusing MA20/MA50/ADX from vbt_to_pd_ind.py -- no new indicator)
        -> Market-regime filter (core.market_context.build_market_context,
           reused unchanged)
        -> Signal (core.rules.signals.Signal, the SAME dataclass every other
           strategy in this repo uses)
        -> Existing risk management + simulator (core.sim.engine.simulate --
           ATR stop/target/trail/position sizing/slippage/fees are all
           computed THERE, unmodified)
        -> Trade results / metrics (core.sim.metrics-compatible summaries)

WHY A NEW MODULE INSTEAD OF A NEW core/patterns/*.py FILE
-----------------------------------------------------------
Every existing pattern function in core/patterns/ has the signature
`(df, ind, **params) -> DataFrame` and is evaluated one symbol at a time
(see core/patterns/pipeline.py::run_all_patterns). Cross-sectional ranking is
structurally different: a single stock's "momentum" number is meaningless
for this strategy until it is compared against the WHOLE universe on the
SAME date. That comparison has to happen in an orchestration layer that
holds all symbols' data at once -- something core/patterns/pipeline.py's
per-symbol signature cannot express. This module is that orchestration
layer; per-symbol detections are still produced in the standard shape a
human reading core/patterns/*.py would recognize, and are compiled into the
exact same Signal dataclass so they run through the unmodified simulator.

LOOK-AHEAD DISCIPLINE
----------------------
1. absolute_momentum/relative_momentum (core/momentum.py) are shift-based:
   the value at bar t only ever reads close[t] and close[t-lookback_bars].
2. technical_confirmation at bar t reads MA20/MA50/ADX at t, which are
   rolling windows ending at t -- standard for every indicator in this
   codebase, not something introduced here.
3. Cross-sectional rank at date t (`cross_sectional_rank`) is computed with
   pandas `.rank(axis=1)` on the row for date t ONLY -- ranking never
   consults another date's row, so a later date's momentum can never affect
   an earlier date's rank.
4. market regime (mkt_up) at bar t uses the index's own MA20/MA50 as of t
   (core.market_context, unmodified).
5. A signal generated from information available at bar t's CLOSE is only
   ever filled at bar t+1's OPEN (delay_bars=1, enter_at="open"), identical
   to how the existing next-open patterns avoid trading on their own setup
   bar (core/rules/signals.py's `enter_overrides` defaults).
core/strategies/test_momentum_strategy.py asserts (1), (3) and (5) directly
with synthetic data where the correct answer is known by construction.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from stock_explorer import load_data, compute_indicators
from vbt_to_pd_ind import extract_symbol_df, build_ind_df_from_vbt
from market_context import build_market_context, MARKET_INDEX
from rules.signals import Signal
from sim.engine import simulate

from momentum import (
    LOOKBACK_BARS,
    absolute_momentum,
    relative_momentum,
    technical_confirmation,
)

# Reuse the exact same NIFTY ticker market_context.py benchmarks against, so
# "the market" means the same thing across the whole codebase.
BENCHMARK = MARKET_INDEX  # "^NSEI"


# ---------------------------------------------------------------------------
# Configuration -- every knob called for in the research brief. None of these
# defaults are claimed to be optimal; they are a reasonable, documented
# starting point for a first, explicitly-unoptimized research pass (see
# core/research_momentum_experiment.py, which sweeps the lookback and leaves
# everything else fixed so results are comparable).
# ---------------------------------------------------------------------------
@dataclass
class MomentumConfig:
    lookback_label: str = "6M"            # one of LOOKBACK_BARS: "3M","6M","9M","12M"

    use_absolute_momentum: bool = True
    min_absolute_momentum: float = 0.0    # eligibility floor on momentum_{lookback}

    use_relative_momentum: bool = True
    min_relative_momentum: float = 0.0    # eligibility floor on relative_momentum_{lookback}

    # Cross-sectional selection
    selection_method: str = "top_n"       # "top_n" | "top_percentile"
    top_n: int = 5
    top_percentile: float = 0.1           # used only when selection_method == "top_percentile"

    # Technical trend confirmation (reuses existing MA20/MA50/ADX; no new indicator)
    use_technical_confirmation: bool = True
    require_close_above_ma20: bool = True
    require_close_above_ma50: bool = True
    require_ma20_above_ma50: bool = True
    use_adx: bool = False
    adx_min: float = 20.0

    # Market/environment filter (reuses core.market_context, unmodified)
    use_market_regime: bool = True

    # Avoid re-firing every single day a name sits in the top N; only signal
    # again for the same symbol after this many bars since its last signal.
    rebalance_bars: int = 21

    # Risk management handed to the EXISTING simulator as ATR multiples --
    # the simulator computes the actual stop/target/trailing prices and
    # position size (core/sim/engine.py), this strategy never does.
    sl_atr: float = 2.0
    tp_atr: float = 4.0
    trail_atr: float = 1.5
    max_hold_bars: int = 0                # 0 = no time-based exit

    @property
    def lookback_bars(self) -> int:
        return LOOKBACK_BARS[self.lookback_label]

    @property
    def tag(self) -> str:
        return f"Momentum_{self.lookback_label}"


# ---------------------------------------------------------------------------
# Data loading (with disk caching -- mirrors core/event_study.py::_load_cached
# so repeated research runs over the same universe don't re-hit yfinance).
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str, symbol: str, years: float) -> str:
    safe = symbol.replace("&", "_").replace("^", "IDX").replace(".", "_")
    return os.path.join(cache_dir, f"{safe}_{years}y.pkl")


def _load_cached_raw(symbol: str, years: float, cache_dir: Optional[str]) -> pd.DataFrame:
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = _cache_path(cache_dir, symbol, years)
        if os.path.exists(path):
            return pd.read_pickle(path)
    df = load_data(symbol, years)
    if cache_dir:
        df.to_pickle(_cache_path(cache_dir, symbol, years))
    return df


def load_symbol_frames(symbol: str, years: float, cache_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(df_ohlcv [title-case OHLCV], ind_df [MA20/MA50/ADX/ATR/...]) for one
    symbol, built with the same helpers every other strategy in this repo
    uses (stock_explorer.load_data -> vbt_to_pd_ind.extract_symbol_df ->
    stock_explorer.compute_indicators -> vbt_to_pd_ind.build_ind_df_from_vbt)."""
    raw = _load_cached_raw(symbol, years, cache_dir)
    df_ohlcv = extract_symbol_df(raw, symbol=symbol, case="title")
    ind = compute_indicators(extract_symbol_df(raw, symbol=symbol, case="lower"))
    ind_df = build_ind_df_from_vbt(df_ohlcv, ind)
    return df_ohlcv, ind_df


def load_benchmark_close(years: float, cache_dir: Optional[str] = None) -> pd.Series:
    raw = _load_cached_raw(BENCHMARK, years, cache_dir)
    return extract_symbol_df(raw, symbol=BENCHMARK, case="title")["Close"]


def build_universe_frames(
    symbols: List[str], years: float, cache_dir: Optional[str] = None
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """symbol -> (df_ohlcv, ind_df); symbols that fail to load are skipped
    (and reported), never silently dropped without a message."""
    frames: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    for sym in symbols:
        try:
            frames[sym] = load_symbol_frames(sym, years, cache_dir)
        except Exception as e:
            print(f"momentum_strategy: skipping {sym} ({e})")
    return frames


# ---------------------------------------------------------------------------
# Cross-sectional momentum, ranking and selection
# ---------------------------------------------------------------------------
def compute_momentum_panels(
    frames: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    benchmark_close: pd.Series,
    lookback_bars: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-symbol absolute/relative momentum, assembled into two
    date x symbol panels. Symbols trade on slightly different calendars
    (holidays/halts), so the panel is naturally outer-joined on date by
    pandas; a symbol with no data for a given date is NaN there, never
    forward-filled across symbols (only relative_momentum's OWN benchmark
    alignment forward-fills, and only backward-looking, per core/momentum.py)."""
    abs_panel, rel_panel = {}, {}
    for sym, (df_ohlcv, _ind) in frames.items():
        close = df_ohlcv["Close"]
        abs_panel[sym] = absolute_momentum(close, lookback_bars)
        rel_panel[sym] = relative_momentum(close, benchmark_close, lookback_bars)
    return pd.DataFrame(abs_panel), pd.DataFrame(rel_panel)


def eligibility_mask(abs_df: pd.DataFrame, rel_df: pd.DataFrame, cfg: MomentumConfig) -> pd.DataFrame:
    """Boolean date x symbol mask of names allowed into the ranking at all.
    A symbol with insufficient history (NaN momentum) is NEVER eligible,
    regardless of threshold settings."""
    mask = pd.DataFrame(True, index=abs_df.index, columns=abs_df.columns)
    if cfg.use_absolute_momentum:
        mask &= abs_df >= cfg.min_absolute_momentum
    if cfg.use_relative_momentum:
        mask &= rel_df >= cfg.min_relative_momentum
    mask &= abs_df.notna() & rel_df.notna()
    return mask


def cross_sectional_rank(rank_basis: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    """Per-date rank across symbols (1 = strongest momentum). pandas
    `.rank(axis=1)` operates row-by-row, so date t's rank can only ever be a
    function of date t's own cross-section -- this is the no-lookahead
    guarantee for the ranking step specifically."""
    basis = rank_basis.where(eligible)
    return basis.rank(axis=1, ascending=False, method="first")


def select_candidates(rank_df: pd.DataFrame, eligible: pd.DataFrame, cfg: MomentumConfig) -> pd.DataFrame:
    """Boolean date x symbol mask of names selected on each date, per
    `cfg.selection_method`. Neither top_n nor top_percentile is asserted to
    be the right choice -- both are supported so the research script can
    compare them; nothing here defaults to "best".
    """
    if cfg.selection_method == "top_percentile":
        n_eligible = eligible.sum(axis=1)
        cutoff = (n_eligible * cfg.top_percentile).clip(lower=1)
    elif cfg.selection_method == "top_n":
        cutoff = pd.Series(cfg.top_n, index=rank_df.index)
    else:
        raise ValueError(f"Unknown selection_method: {cfg.selection_method!r}")
    selected = rank_df.le(cutoff, axis=0)
    return (selected & eligible).fillna(False)


# ---------------------------------------------------------------------------
# Per-symbol detections -> Signal compilation
# ---------------------------------------------------------------------------
def generate_symbol_detections(
    symbol: str,
    df_ohlcv: pd.DataFrame,
    ind_df: pd.DataFrame,
    selected_col: pd.Series,
    abs_mom_col: pd.Series,
    rel_mom_col: pd.Series,
    rank_col: pd.Series,
    ctx: Optional[pd.DataFrame],
    cfg: MomentumConfig,
) -> pd.DataFrame:
    """One row per day this symbol newly qualifies as a momentum candidate:
    selected in the cross-sectional screen AND (optionally) technically
    confirmed AND (optionally) the market regime is favorable. Columns
    mirror what a core/patterns/*.py detector would emit (date, pattern,
    score/rank, notes) plus the momentum-specific fields the brief asked to
    keep visible (absolute/relative momentum, rank, technical/regime flags).
    """
    idx = df_ohlcv.index
    # reindex with an explicit fill_value (rather than reindex().fillna())
    # so a boolean Series stays boolean instead of round-tripping through
    # object dtype for the missing slots.
    selected = selected_col.reindex(idx, fill_value=False).astype(bool)
    abs_mom = abs_mom_col.reindex(idx)
    rel_mom = rel_mom_col.reindex(idx)
    rank = rank_col.reindex(idx)

    tech = technical_confirmation(
        df_ohlcv["Close"], ind_df,
        require_close_above_ma20=cfg.require_close_above_ma20,
        require_close_above_ma50=cfg.require_close_above_ma50,
        require_ma20_above_ma50=cfg.require_ma20_above_ma50,
        adx_min=(cfg.adx_min if cfg.use_adx else 0.0),
    )
    confirmed = tech["confirmed"].astype(bool) if cfg.use_technical_confirmation else pd.Series(True, index=idx)

    if cfg.use_market_regime and ctx is not None and "mkt_up" in ctx.columns:
        regime_ok = ctx["mkt_up"].reindex(idx, fill_value=True).astype(bool)
    else:
        regime_ok = pd.Series(True, index=idx)

    candidate = selected & confirmed & regime_ok

    # Only fire on the day a symbol *becomes* a fresh candidate, and no more
    # often than every `rebalance_bars` bars -- otherwise a stock sitting in
    # the top N for months would re-signal on every single bar it stays
    # there, drowning the simulator's single-open-position slot in
    # redundant same-symbol signals.
    is_new = candidate & ~candidate.shift(1, fill_value=False)
    fire = np.zeros(len(idx), dtype=bool)
    last_fire = -10 ** 9
    for b in np.flatnonzero(is_new.to_numpy()):
        if b - last_fire >= cfg.rebalance_bars:
            fire[b] = True
            last_fire = b

    rows = []
    for b in np.flatnonzero(fire):
        rows.append({
            "bar_idx": int(b),
            "date": idx[b],
            "symbol": symbol,
            "pattern": cfg.tag,
            "lookback": cfg.lookback_label,
            "absolute_momentum": float(abs_mom.iloc[b]) if pd.notna(abs_mom.iloc[b]) else np.nan,
            "relative_momentum": float(rel_mom.iloc[b]) if pd.notna(rel_mom.iloc[b]) else np.nan,
            "rank": float(rank.iloc[b]) if pd.notna(rank.iloc[b]) else np.nan,
            "technical_confirmed": bool(confirmed.iloc[b]),
            "market_regime_ok": bool(regime_ok.iloc[b]),
        })
    return pd.DataFrame(rows, columns=[
        "bar_idx", "date", "symbol", "pattern", "lookback",
        "absolute_momentum", "relative_momentum", "rank",
        "technical_confirmed", "market_regime_ok",
    ])


def compile_momentum_signals(detections: pd.DataFrame, cfg: MomentumConfig) -> pd.DataFrame:
    """detections -> Signal rows, reusing core.rules.signals.Signal
    unmodified. Entries are placed at the NEXT bar's open (delay_bars=1,
    enter_at="open"): the decision made from bar t's close is never filled
    on bar t itself. sl_atr/tp_atr/trail_atr are ATR *multiples*; the actual
    stop/target/trailing prices and position size are computed inside
    core.sim.engine.simulate(), identical to every other strategy."""
    cols = ["bar_idx", "side", "sl_atr", "tp_atr", "trail_atr", "tag",
            "size_cash", "enter_at", "delay_bars", "priority", "meta"]
    if detections is None or detections.empty:
        return pd.DataFrame(columns=cols)

    out = []
    for r in detections.itertuples(index=False):
        out.append(Signal(
            bar_idx=int(r.bar_idx),
            side="long",
            sl_atr=cfg.sl_atr,
            tp_atr=cfg.tp_atr,
            trail_atr=cfg.trail_atr,
            tag=cfg.tag,
            enter_at="open",
            delay_bars=1,
            priority=5,
            meta={
                "lookback": r.lookback,
                "absolute_momentum": r.absolute_momentum,
                "relative_momentum": r.relative_momentum,
                "rank": r.rank,
                "technical_confirmed": r.technical_confirmed,
                "market_regime_ok": r.market_regime_ok,
            },
        ).to_dict())
    return pd.DataFrame(out, columns=cols)


# ---------------------------------------------------------------------------
# Simulation config + per-symbol backtest
# ---------------------------------------------------------------------------
def default_sim_cfg(cfg: MomentumConfig) -> dict:
    """Risk/execution/cost settings for core.sim.engine.simulate(), mirroring
    the SAME production numbers the existing pattern strategy uses
    (core/pipeline.py's `cfg`, api/main.py::build_cfg()) so momentum results
    are cost-comparable rather than benchmarked in an easier environment."""
    return {
        "execution": {
            "delay_bars": 1,
            "enter_at": "open",
            "max_open_positions": 1,
            "cooldown_bars": 0,
            "max_hold_bars": cfg.max_hold_bars,
        },
        "risk": {
            "capital": 1_000_000.0,
            "max_risk_per_trade": 0.005,
            "position_sizing": "atr",
            "fixed_cash_per_trade": 50_000.0,
        },
        "filters": {
            "min_atr": 0.0,
            # small safety margin past the lookback so we never evaluate an
            # entry before there's enough history for this lookback's
            # momentum AND the vbt indicators' own warmup (MA50 etc).
            "warmup_cutoff_bars": max(cfg.lookback_bars + 5, 60),
        },
        "costs": {"per_share": 0.0, "per_trade": 0.0, "bps_fee": 0.0, "slippage_bps": 2.0},
        "equity": {"mtm": False, "mtm_haircut_bps": 0.0},
    }


def run_symbol_backtest(
    symbol: str,
    df_ohlcv: pd.DataFrame,
    ind_df: pd.DataFrame,
    detections: pd.DataFrame,
    cfg: MomentumConfig,
    sim_cfg: Optional[dict] = None,
) -> dict:
    """Compile this symbol's momentum detections into signals and run them
    through the UNMODIFIED core.sim.engine.simulate()."""
    signals = compile_momentum_signals(detections, cfg)
    full_cfg = sim_cfg or default_sim_cfg(cfg)
    trades, equity, fills = simulate(df_ohlcv, signals, ind_df, full_cfg)
    metrics = summarize_trades_only(trades)
    metrics.update(_equity_metrics(equity))
    return {"symbol": symbol, "signals": signals, "trades": trades, "equity": equity, "fills": fills, "metrics": metrics}


def _equity_metrics(equity: pd.Series) -> dict:
    if equity is None or len(equity) == 0:
        return {}
    roll_max = equity.cummax()
    dd = float((equity / roll_max - 1.0).min())
    rets = equity.pct_change().dropna()
    sharpe = float((rets.mean() / (rets.std() + 1e-12)) * np.sqrt(252)) if len(rets) >= 2 else float("nan")
    return {
        "return_total": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "max_drawdown": dd,
        "sharpe_daily": sharpe,
    }


def summarize_trades_only(trades: pd.DataFrame) -> dict:
    """Trade-based metrics only (no equity curve required) -- mirrors
    core/sim/metrics.py::summarize's trade fields exactly, so numbers are
    directly comparable to the existing strategy's reporting, extended with
    total_pnl/total_fees for pooling trades across many symbols (a single
    combined equity curve doesn't exist for a multi-symbol universe scan,
    since core.sim.engine.simulate() is single-symbol -- see module
    docstring)."""
    if trades is None or len(trades) == 0:
        return {"trades": 0}
    wins = trades["pnl"] > 0
    gp = float(trades.loc[wins, "pnl"].sum()) if wins.any() else 0.0
    gl = -float(trades.loc[~wins, "pnl"].sum()) if (~wins).any() else 0.0
    return {
        "trades": int(len(trades)),
        "win_rate": float(wins.mean()),
        "avg_win": float(trades.loc[wins, "pnl"].mean()) if wins.any() else 0.0,
        "avg_loss": float(trades.loc[~wins, "pnl"].mean()) if (~wins).any() else 0.0,
        "profit_factor": (gp / gl) if gl > 0 else float("inf"),
        "expectancy": float(trades["pnl"].mean()),
        "total_pnl": float(trades["pnl"].sum()),
        "median_hold_bars": float((trades["exit_bar"] - trades["entry_bar"]).median()),
        "total_fees": float(trades["fees"].sum()) if "fees" in trades.columns else 0.0,
    }


# ---------------------------------------------------------------------------
# Full cross-sectional experiment across a universe
# ---------------------------------------------------------------------------
def run_momentum_experiment(
    symbols: List[str],
    years: float,
    cfg: MomentumConfig,
    cache_dir: Optional[str] = None,
    train_frac: float = 0.7,
) -> dict:
    """Run the full pipeline (load -> momentum -> relative strength -> rank
    -> select -> technical/regime confirm -> Signal -> simulate) across
    `symbols`, and report train/valid metrics using the SAME convention as
    the existing core/audit_*.py research scripts: a positional, per-symbol
    train_frac split on bar count (see
    core/audit_walkforward_market_condition_diagnosis.py), never a threshold
    re-fit on the validation half -- nothing in this strategy is fit on data
    at all (momentum/rank/technical/regime are all fixed formulas), so the
    split here is purely for honest train-vs-out-of-sample REPORTING, not
    parameter selection.
    """
    frames = build_universe_frames(symbols, years, cache_dir)
    if not frames:
        raise RuntimeError("No symbols could be loaded for the momentum experiment")

    benchmark_close = load_benchmark_close(years, cache_dir)

    abs_df, rel_df = compute_momentum_panels(frames, benchmark_close, cfg.lookback_bars)
    rank_basis = rel_df if cfg.use_relative_momentum else abs_df
    eligible = eligibility_mask(abs_df, rel_df, cfg)
    rank_df = cross_sectional_rank(rank_basis, eligible)
    selected = select_candidates(rank_df, eligible, cfg)

    per_symbol: Dict[str, dict] = {}
    all_trades = []
    for sym, (df_ohlcv, ind_df) in frames.items():
        if sym not in selected.columns:
            continue
        try:
            ctx = build_market_context(sym, df_ohlcv["Close"], years=years, cache_dir=cache_dir)
        except Exception:
            ctx = None
        dets = generate_symbol_detections(
            sym, df_ohlcv, ind_df,
            selected_col=selected[sym],
            abs_mom_col=abs_df[sym],
            rel_mom_col=rel_df[sym],
            rank_col=rank_df[sym],
            ctx=ctx,
            cfg=cfg,
        )
        result = run_symbol_backtest(sym, df_ohlcv, ind_df, dets, cfg)
        per_symbol[sym] = result

        trades = result["trades"]
        if trades is not None and len(trades):
            trades = trades.copy()
            split_bar = int(len(df_ohlcv) * train_frac)
            trades["period"] = np.where(trades["entry_bar"] < split_bar, "train", "valid")
            trades["symbol"] = sym
            all_trades.append(trades)

    trades_all = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    return {
        "lookback": cfg.lookback_label,
        "config": asdict(cfg),
        "n_symbols": len(frames),
        "n_symbols_with_trades": len(all_trades),
        "overall": summarize_trades_only(trades_all),
        "train": summarize_trades_only(trades_all[trades_all["period"] == "train"]) if len(trades_all) else {"trades": 0},
        "valid": summarize_trades_only(trades_all[trades_all["period"] == "valid"]) if len(trades_all) else {"trades": 0},
        "by_symbol": {s: r["metrics"] for s, r in per_symbol.items()},
        "trades_all": trades_all,
        "per_symbol": per_symbol,
    }


# ---------------------------------------------------------------------------
# Live, today-anchored cross-sectional screen (as opposed to the historical
# backtest above). One implementation shared by api/main.py's
# GET /api/momentum/screen and the point-in-time paper-trading bridge
# (api/momentum_paper.py), so a symbol's live rank/signal can never be
# computed two different ways by two different callers.
# ---------------------------------------------------------------------------
def screen_universe_now(symbols: List[str], years: float, cfg: MomentumConfig, cache_dir: Optional[str] = None) -> dict:
    """Rank `symbols` by today's cross-sectional momentum and report, for
    each, whether it currently qualifies as a momentum candidate (selected
    AND technically confirmed) with an ATR-based entry/stop/target estimate
    -- the same condition core/strategies/momentum_strategy.py would open a
    Signal on on that date.

    Anchored on the LAST DATE WITH ANY REAL DATA rather than the panel's
    final index entry: yfinance can return an in-progress session as a row
    of NaN OHLC (see stock_explorer.load_data), which would otherwise NaN
    out every symbol at once and make a perfectly valid yesterday's close
    look like "no data".

    Returns {"as_of": pd.Timestamp | None, "n_symbols": int, "rows": [...]}
    with plain Python types (None where NaN), sorted by rank ascending.
    """
    frames = build_universe_frames(symbols, years, cache_dir)
    if not frames:
        return {"as_of": None, "n_symbols": 0, "rows": []}

    bench_close = load_benchmark_close(years, cache_dir)
    abs_df, rel_df = compute_momentum_panels(frames, bench_close, cfg.lookback_bars)
    rank_basis = rel_df if cfg.use_relative_momentum else abs_df
    eligible = eligibility_mask(abs_df, rel_df, cfg)
    rank_df = cross_sectional_rank(rank_basis, eligible)
    selected = select_candidates(rank_df, eligible, cfg)

    has_data = abs_df.notna().any(axis=1) | rel_df.notna().any(axis=1)
    valid_dates = abs_df.index[has_data]
    if len(valid_dates) == 0:
        return {"as_of": None, "n_symbols": len(frames), "rows": []}
    last_date = valid_dates[-1]

    def _f(x) -> Optional[float]:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) else None

    rows = []
    for sym, (df_ohlcv, ind_df) in frames.items():
        if sym not in rank_df.columns or last_date not in rank_df.index or last_date not in df_ohlcv.index:
            continue
        r = rank_df.loc[last_date, sym]
        if pd.isna(r):
            continue
        close_last = float(df_ohlcv["Close"].loc[last_date])
        atr_last = float(ind_df["ATR"].loc[last_date]) if "ATR" in ind_df.columns else float("nan")
        is_selected = bool(selected.loc[last_date, sym]) if sym in selected.columns else False
        # Mirrors generate_symbol_detections' handling of the SAME flag: when
        # use_technical_confirmation is off, confirmation is bypassed
        # (always true) rather than silently still being required.
        if cfg.use_technical_confirmation:
            tech = technical_confirmation(
                df_ohlcv["Close"], ind_df,
                require_close_above_ma20=cfg.require_close_above_ma20,
                require_close_above_ma50=cfg.require_close_above_ma50,
                require_ma20_above_ma50=cfg.require_ma20_above_ma50,
                adx_min=(cfg.adx_min if cfg.use_adx else 0.0),
            )
            is_confirmed = bool(tech["confirmed"].loc[last_date])
        else:
            is_confirmed = True
        row = {
            "symbol": sym,
            "rank": int(r),
            "absolute_momentum": _f(abs_df.loc[last_date, sym]) if sym in abs_df.columns else None,
            "relative_momentum": _f(rel_df.loc[last_date, sym]) if sym in rel_df.columns else None,
            "technical_confirmed": is_confirmed,
            "selected": is_selected,
            "close": close_last,
            "signal": None, "entry_est": None, "sl_est": None, "tp_est": None,
            "sl_atr": None, "tp_atr": None,
        }
        # Only a fully-qualifying candidate (selected AND technically
        # confirmed) gets an entry/stop/target estimate, using the SAME
        # ATR-multiple convention the simulator trades with (cfg.sl_atr/
        # tp_atr) -- not a new estimate methodology.
        if is_selected and is_confirmed and math.isfinite(atr_last):
            row.update({
                "signal": "long",
                "entry_est": close_last,
                "sl_est": close_last - cfg.sl_atr * atr_last,
                "tp_est": close_last + cfg.tp_atr * atr_last,
                "sl_atr": cfg.sl_atr, "tp_atr": cfg.tp_atr,
            })
        rows.append(row)
    rows.sort(key=lambda r: r["rank"])
    return {"as_of": last_date, "n_symbols": len(frames), "rows": rows}
