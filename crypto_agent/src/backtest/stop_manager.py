"""
Stop loss management for backtesting.

Enhanced with:
- Stop-to-breakeven with buffer (not exact BE — avoids premature stops)
- ATR-based trailing stop after reaching profit threshold
- RSI divergence confirmation candle check
"""

import pandas as pd

from src.indicators.rsi import calculate_rsi
from src.indicators.trend import calculate_atr, calculate_keltner_channel
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Breakeven buffer: move stop to entry ± 0.15% (not exact breakeven)
# This avoids getting stopped out on normal retracement noise
BE_BUFFER_PCT = 0.0015

# Trailing stop activates after this many R of profit
TRAILING_ACTIVATION_R = 1.5

# Keltner Channel trailing parameters
KELTNER_EMA_PERIOD = 20
KELTNER_ATR_PERIOD = 14
KELTNER_ATR_MULTIPLIER = 2.0


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


def get_be_price(entry_price: float, direction: str) -> float:
    """
    Get breakeven price with buffer.

    Instead of exact breakeven (which gets stopped out on normal noise),
    sets stop slightly in profit to cover costs and avoid whipsaws.

    Args:
        entry_price: Original entry price.
        direction: "LONG" or "SHORT".

    Returns:
        Breakeven price with buffer.
    """
    if direction == "LONG":
        return entry_price * (1 + BE_BUFFER_PCT)
    else:
        return entry_price * (1 - BE_BUFFER_PCT)


def calculate_trailing_stop(
    candles: pd.DataFrame,
    current_index: int,
    direction: str,
    entry_price: float,
    stop_price: float,
    current_stop: float,
) -> float:
    """
    Calculate Keltner Channel trailing stop.

    Uses the Keltner Channel lower band (for longs) or upper band (for shorts)
    as the trailing stop level. Research shows Keltner trailing is the #1 stop
    strategy across 87 tested methods.

    Only activates after TRAILING_ACTIVATION_R of profit to avoid premature trailing.
    Only moves stop in the profitable direction (never widens risk).

    Args:
        candles: OHLCV DataFrame.
        current_index: Current candle index.
        direction: "LONG" or "SHORT".
        entry_price: Original entry price.
        stop_price: Original stop loss price.
        current_stop: Current stop level.

    Returns:
        New stop price (may be same as current_stop if no improvement).
    """
    if current_index < max(KELTNER_EMA_PERIOD, KELTNER_ATR_PERIOD) + 1:
        return current_stop

    window = candles.iloc[:current_index + 1]
    keltner = calculate_keltner_channel(
        window,
        ema_period=KELTNER_EMA_PERIOD,
        atr_period=KELTNER_ATR_PERIOD,
        atr_multiplier=KELTNER_ATR_MULTIPLIER,
    )

    if len(keltner) < 1:
        return current_stop

    current_price = float(candles["close"].iloc[current_index])
    risk_distance = abs(entry_price - stop_price)

    if risk_distance == 0:
        return current_stop

    # Check if we've reached the trailing activation threshold
    if direction == "LONG":
        profit_r = (current_price - entry_price) / risk_distance
        if profit_r >= TRAILING_ACTIVATION_R:
            # Trail at Keltner lower band
            keltner_lower = float(keltner["lower"].iloc[-1])
            # Only move stop up, never down
            return max(current_stop, keltner_lower)
    else:
        profit_r = (entry_price - current_price) / risk_distance
        if profit_r >= TRAILING_ACTIVATION_R:
            # Trail at Keltner upper band
            keltner_upper = float(keltner["upper"].iloc[-1])
            # Only move stop down, never up
            return min(current_stop, keltner_upper)

    return current_stop
