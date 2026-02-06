"""
Stop loss management for backtesting.
Simulates stop-to-breakeven logic based on RSI divergence confirmation candles.
"""

import pandas as pd

from src.indicators.rsi import calculate_rsi
from src.utils.logger import get_logger

logger = get_logger(__name__)


def check_stop_hit(
    candle: pd.Series,
    direction: str,
    stop_price: float,
) -> bool:
    """
    Check if a candle would have triggered the stop loss.

    For longs: stop is hit if low <= stop_price.
    For shorts: stop is hit if high >= stop_price.

    Args:
        candle: A single candle row with 'high', 'low' columns.
        direction: "LONG" or "SHORT".
        stop_price: Current stop loss price.

    Returns:
        True if stop was triggered.
    """
    if direction == "LONG":
        return candle["low"] <= stop_price
    else:
        return candle["high"] >= stop_price


def check_tp_hit(
    candle: pd.Series,
    direction: str,
    tp_price: float,
) -> bool:
    """
    Check if a candle would have triggered a take profit level.

    For longs: TP hit if high >= tp_price.
    For shorts: TP hit if low <= tp_price.
    """
    if direction == "LONG":
        return candle["high"] >= tp_price
    else:
        return candle["low"] <= tp_price


def should_move_stop_to_be(
    candles: pd.DataFrame,
    current_index: int,
    direction: str,
    entry_index: int,
) -> bool:
    """
    Check if the RSI divergence has been confirmed by a closing candle,
    meaning the stop should be moved to breakeven.

    Confirmation criteria:
    - For longs: a green candle closes (close > open)
    - For shorts: a red candle closes (close < open)
    - RSI still shows the divergence pattern (not reversed)
    - Must be after the entry candle

    Args:
        candles: Full OHLCV DataFrame.
        current_index: Index of the current candle to check.
        direction: "LONG" or "SHORT".
        entry_index: Index where the position was entered.

    Returns:
        True if stop should be moved to breakeven.
    """
    if current_index <= entry_index:
        return False

    candle = candles.iloc[current_index]

    if direction == "LONG":
        # Green candle close confirms bullish divergence
        if candle["close"] <= candle["open"]:
            return False

        # Check RSI is still rising from oversold
        rsi = calculate_rsi(candles["close"].iloc[:current_index + 1])
        if len(rsi) < 2:
            return False
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]

        # RSI should be rising and above its recent low
        return current_rsi > prev_rsi and current_rsi > 30

    else:  # SHORT
        # Red candle close confirms bearish divergence
        if candle["close"] >= candle["open"]:
            return False

        rsi = calculate_rsi(candles["close"].iloc[:current_index + 1])
        if len(rsi) < 2:
            return False
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]

        # RSI should be falling and below its recent high
        return current_rsi < prev_rsi and current_rsi < 70
