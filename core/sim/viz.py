from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def overlay_trades(ax, trades_df, *, show_lines: bool = True, show_legend: bool = True):
    """
    Overlay trade markers on an mplfinance Axes (using integer bar indices).
    Expects columns: entry_bar, exit_bar, entry_price, exit_price, side ('long'/'short').

    Usage:
        ax = axlist[0]  # from mpf.plot(..., returnfig=True)
        overlay_trades(ax, trades_df)
    """
    if trades_df is None or len(trades_df) == 0:
        return ax

    t = trades_df

    # Longs
    lmask = (t["side"] == "long")
    if lmask.any():
        ax.scatter(
            t.loc[lmask, "entry_bar"], t.loc[lmask, "entry_price"],
            marker="^", s=60, zorder=6, label="Entry Long"
        )
        ax.scatter(
            t.loc[lmask, "exit_bar"], t.loc[lmask, "exit_price"],
            marker="v", s=60, zorder=6, label="Exit Long"
        )

    # Shorts
    smask = (t["side"] == "short")
    if smask.any():
        ax.scatter(
            t.loc[smask, "entry_bar"], t.loc[smask, "entry_price"],
            marker="v", s=60, zorder=6, label="Entry Short"
        )
        ax.scatter(
            t.loc[smask, "exit_bar"], t.loc[smask, "exit_price"],
            marker="^", s=60, zorder=6, label="Exit Short"
        )

    # Optional rails between entry and exit
    if show_lines:
        for r in t.itertuples(index=False):
            ax.plot([r.entry_bar, r.exit_bar], [r.entry_price, r.exit_price], linewidth=1.0, alpha=0.6, zorder=5)

    # Keep legend tidy (deduplicate labels)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        uniq_h, uniq_l = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = True
                uniq_h.append(h); uniq_l.append(l)
        if uniq_l:
            ax.legend(uniq_h, uniq_l, loc="upper left")
    return ax


def overlay_fills(ax, fills_df):
    """
    (Optional) If you pass the `fills_df` from simulate(...), this will
    draw small tick marks at ENTRY / EXIT_* bars for quick debugging.
    """
    if fills_df is None or len(fills_df) == 0:
        return ax
    f = fills_df
    y = ax.get_ylim()[0]  # near bottom
    for r in f.itertuples(index=False):
        ax.plot([r.bar, r.bar], [y, y], marker="_", linewidth=0.8, alpha=0.7)
    return ax


def overlay_signals_rail(ax_price, signals_df, *, height_frac: float = 0.12):
    """
    Add a compact 'signals rail' below the price axis.
    - ax_price: the mplfinance price axis (axlist[0])
    - signals_df: DataFrame with ['bar_idx','side']
    - height_frac: fraction of the price-axes height to allocate to the rail
    Returns: the created rail Axes
    """
    import matplotlib as mpl
    import numpy as np

    if signals_df is None or len(signals_df) == 0:
        return None

    # Make a new axes just below the price axes (shares X with price)
    fig = ax_price.figure
    bbox = ax_price.get_position()
    pad = 0.01 * bbox.height
    rail_h = bbox.height * height_frac
    rail_bbox = mpl.transforms.Bbox.from_bounds(
        bbox.x0, bbox.y0 - rail_h - pad, bbox.width, rail_h
    )
    ax_rail = fig.add_axes(rail_bbox, sharex=ax_price)

    # Prepare data
    x = signals_df["bar_idx"].to_numpy()
    y = np.where(signals_df["side"].to_numpy() == "long", 1.0, -1.0)

    # Stems + markers
    ax_rail.vlines(x, 0.0, y, linewidth=1.0, alpha=0.7)
    ax_rail.scatter(x[y > 0], y[y > 0], marker="^", s=25, zorder=3, label="Long")
    ax_rail.scatter(x[y < 0], y[y < 0], marker="v", s=25, zorder=3, label="Short")

    # Cosmetics
    ax_rail.set_ylim(-1.6, 1.6)
    ax_rail.set_yticks([])                    # hide y numbers
    ax_rail.grid(axis="x", which="both", alpha=0.15)
    ax_rail.set_ylabel("Sig", rotation=0, labelpad=10)
    for tl in ax_rail.get_xticklabels():
        tl.set_visible(False)                  # price axis already shows x labels
    ax_rail.patch.set_alpha(0.0)              # transparent background
    # Avoid duplicate legends from price axis
    handles, labels = ax_rail.get_legend_handles_labels()
    if labels:
        ax_rail.legend(loc="center left", ncols=2, fontsize=8, frameon=False)
    return ax_rail
