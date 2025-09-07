# File: patterns/viz.py

import matplotlib.pyplot as plt

def overlay_patterns(ax, detections, code_map=None, y_pos="top"):
    """
    Draw compact pattern markers and vertical lines on an existing candlestick axis.
    Expects 'date' and 'pattern' columns in detections.
    """
    if detections is None or detections.empty:
        return

    if code_map is None:
        code_map = {
            "BullishEngulfing": "BE↑", "BearishEngulfing": "BE↓",
            "Hammer": "HM", "Doji": "DJ",
            "Breakout20": "BO20", "Breakout55": "BO55",
            "Breakdown20": "BD20", "Breakdown55": "BD55",
            "PullbackToMA20": "PB20",
            "BB_BuyDip": "BB↓", "BB_SellFade": "BB↑",
            "RSI_40_Up": "RSI40↑", "RSI_60_Down": "RSI60↓",
            "MACD_Cross_Zeroish_Up": "MACD0×↑",
            "MACD_Cross_Zeroish_Down": "MACD0×↓",
        }

    y_top = ax.get_ylim()[1]
    y_bottom = ax.get_ylim()[0]
    y = y_top if y_pos == "top" else y_bottom

    for _, row in detections.iterrows():
        ax.axvline(row["date"], linestyle="--", linewidth=0.6, alpha=0.3)
        label = code_map.get(row["pattern"], row["pattern"])
        ax.text(row["date"], y, label, rotation=90, va="top" if y_pos=="top" else "bottom",
                ha="center", fontsize=7, alpha=0.75)
    ax.figure.canvas.draw_idle()
