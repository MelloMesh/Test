"""
Fibonacci retracement/extension calculation and swing high/low detection.

Swing detection uses a fractal-based approach: a swing high is a candle whose
high is the highest within N bars on each side. A swing low is a candle whose
low is the lowest within N bars on each side. This is equivalent to Williams'
fractals and is the most common method on TradingView.

Key design decisions:
- Fractals with configurable window (min_bars on each side). The default of 5
  means a swing high must be higher than the 5 bars before and after it.
- Swings are detected on OHLC data (using high/low, not close) for accuracy.
- Fibonacci levels are always calculated dynamically from detected swings.
- The golden pocket (0.618-0.886) serves as the entry zone filter.
- Higher-timeframe swings take priority when both 15m and 30m swings exist.
"""

import numpy as np
import pandas as pd

from src.config import (
    FIB_EXTENSION_LEVELS,
    FIB_RETRACEMENT_LEVELS,
    GOLDEN_POCKET_END,
    GOLDEN_POCKET_START,
    SWING_LOOKBACK,
    SWING_MIN_BARS,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def detect_swings(
    candles: pd.DataFrame,
    lookback: int = SWING_LOOKBACK,
    min_bars: int = SWING_MIN_BARS,
) -> list[dict]:
    """
    Find significant swing highs and lows using the fractal method.

    A swing high at index i requires:
        high[i] >= max(high[i-min_bars:i]) AND high[i] >= max(high[i+1:i+min_bars+1])

    A swing low at index i requires:
        low[i] <= min(low[i-min_bars:i]) AND low[i] <= min(low[i+1:i+min_bars+1])

    Only the most recent `lookback` candles are analyzed.

    Args:
        candles: DataFrame with columns: timestamp, open, high, low, close, volume.
        lookback: Number of recent candles to analyze.
        min_bars: Minimum candles on each side of the swing point.

    Returns:
        List of swing dicts sorted by index, each with:
        - type: "high" or "low"
        - price: The high/low price at the swing
        - index: Integer position in the original DataFrame
        - timestamp: Timestamp of the swing candle
    """
    if len(candles) < 2 * min_bars + 1:
        return []

    # Work on the most recent `lookback` candles
    df = candles.tail(lookback).reset_index(drop=True)
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    swings: list[dict] = []

    for i in range(min_bars, n - min_bars):
        # Check swing high: highest in the window
        left_highs = highs[i - min_bars:i]
        right_highs = highs[i + 1:i + min_bars + 1]

        if highs[i] >= max(left_highs) and highs[i] >= max(right_highs):
            # Ensure it's strictly higher than at least one side
            # (prevents flat regions from being detected as swings)
            if highs[i] > min(max(left_highs), max(right_highs)):
                original_idx = len(candles) - lookback + i if lookback <= len(candles) else i
                swings.append({
                    "type": "high",
                    "price": float(highs[i]),
                    "index": int(original_idx),
                    "timestamp": df["timestamp"].iloc[i],
                })

        # Check swing low: lowest in the window
        left_lows = lows[i - min_bars:i]
        right_lows = lows[i + 1:i + min_bars + 1]

        if lows[i] <= min(left_lows) and lows[i] <= min(right_lows):
            if lows[i] < max(min(left_lows), min(right_lows)):
                original_idx = len(candles) - lookback + i if lookback <= len(candles) else i
                swings.append({
                    "type": "low",
                    "price": float(lows[i]),
                    "index": int(original_idx),
                    "timestamp": df["timestamp"].iloc[i],
                })

    # Sort by index (chronological order)
    swings.sort(key=lambda s: s["index"])

    # Remove duplicate swings that are too close together
    swings = _deduplicate_swings(swings, min_distance=min_bars)

    logger.debug(
        f"Detected {len(swings)} swings "
        f"({sum(1 for s in swings if s['type']=='high')} highs, "
        f"{sum(1 for s in swings if s['type']=='low')} lows) "
        f"in {n} candles"
    )
    return swings


def _deduplicate_swings(swings: list[dict], min_distance: int) -> list[dict]:
    """
    Remove duplicate swings that are too close together.
    When two consecutive swings of the same type are within min_distance,
    keep the more extreme one (higher high or lower low).
    """
    if len(swings) < 2:
        return swings

    result: list[dict] = [swings[0]]

    for i in range(1, len(swings)):
        prev = result[-1]
        curr = swings[i]

        if curr["type"] == prev["type"] and abs(curr["index"] - prev["index"]) < min_distance:
            # Same type, too close — keep the more extreme one
            if curr["type"] == "high":
                if curr["price"] > prev["price"]:
                    result[-1] = curr
            else:
                if curr["price"] < prev["price"]:
                    result[-1] = curr
        else:
            result.append(curr)

    return result


def calculate_fib_levels(swing_high: float, swing_low: float) -> dict[float, float]:
    """
    Calculate Fibonacci retracement and extension levels.

    For an upswing (low → high), retracement measures pullback from the high:
        level_price = swing_high - (swing_high - swing_low) * fib_ratio

    Extensions project beyond the swing:
        extension_price = swing_high + (swing_high - swing_low) * (ext_ratio - 1.0)

    Args:
        swing_high: The high price of the swing.
        swing_low: The low price of the swing.

    Returns:
        Dict mapping fib ratio to price level:
        {0.0: swing_high, 0.236: ..., ..., 1.0: swing_low, 1.272: ..., ...}
    """
    diff = swing_high - swing_low

    if diff <= 0:
        logger.warning(f"Invalid swing: high={swing_high} <= low={swing_low}")
        return {}

    levels: dict[float, float] = {}

    # Retracement levels (from high toward low)
    for ratio in FIB_RETRACEMENT_LEVELS:
        levels[ratio] = swing_high - (diff * ratio)

    # Extension levels (below the swing low)
    for ratio in FIB_EXTENSION_LEVELS:
        levels[ratio] = swing_high - (diff * ratio)

    return levels


def is_in_golden_pocket(
    price: float,
    fib_levels: dict[float, float],
) -> bool:
    """
    Check if price is in the Fibonacci golden pocket (0.618 - 0.886).

    This is a HARD filter — if price is outside the golden pocket,
    no trade should be considered regardless of RSI signals.

    Args:
        price: Current price to check.
        fib_levels: Fib level dict from calculate_fib_levels().

    Returns:
        True if price is between the 0.618 and 0.886 retracement levels.
    """
    if GOLDEN_POCKET_START not in fib_levels or GOLDEN_POCKET_END not in fib_levels:
        return False

    gp_start = fib_levels[GOLDEN_POCKET_START]  # Higher price (0.618)
    gp_end = fib_levels[GOLDEN_POCKET_END]       # Lower price (0.886)

    # gp_start > gp_end for upswing fibs (retracement from high)
    upper = max(gp_start, gp_end)
    lower = min(gp_start, gp_end)

    return lower <= price <= upper


def golden_pocket_depth(
    price: float,
    fib_levels: dict[float, float],
) -> float | None:
    """
    Calculate how deep into the golden pocket the price is.

    Returns:
        Float from 0.0 (at 0.618 edge) to 1.0 (at 0.886 edge), or None if
        price is outside the golden pocket.
    """
    if not is_in_golden_pocket(price, fib_levels):
        return None

    gp_start = fib_levels[GOLDEN_POCKET_START]
    gp_end = fib_levels[GOLDEN_POCKET_END]

    upper = max(gp_start, gp_end)
    lower = min(gp_start, gp_end)
    pocket_range = upper - lower

    if pocket_range == 0:
        return 0.0

    # 0.0 at the 0.618 level, 1.0 at the 0.886 level
    if gp_start > gp_end:
        # Normal upswing fib: 0.618 is higher, 0.886 is lower
        return (gp_start - price) / pocket_range
    else:
        return (price - gp_start) / pocket_range


def get_latest_swing_pair(swings: list[dict]) -> tuple[dict, dict] | None:
    """
    Get the most recent swing high-low pair for Fibonacci calculation.

    Looks for the latest pair of alternating swing types (high-low or low-high).
    The most recent swing determines the direction:
    - If last swing is HIGH → uptrend (fib from low to high, look for long retracement)
    - If last swing is LOW → downtrend (fib from high to low, look for short retracement)

    Returns:
        Tuple of (swing_for_low, swing_for_high) or None if insufficient swings.
    """
    if len(swings) < 2:
        return None

    # Find the last two swings of different types
    last = swings[-1]
    for i in range(len(swings) - 2, -1, -1):
        if swings[i]["type"] != last["type"]:
            if last["type"] == "high":
                return (swings[i], last)  # (low, high) pair
            else:
                return (last, swings[i])  # (low, high) pair — reversed order
            break

    return None


def get_fib_take_profit_prices(
    fib_levels: dict[float, float],
    direction: str,
    tp_config: list[dict] | None = None,
) -> list[dict]:
    """
    Calculate take-profit prices from Fibonacci levels.

    For LONG trades (price moving up from golden pocket):
        TPs are at 0.5, 0.382, 0.236 fib levels and 1.272/1.618 extensions.

    For SHORT trades (price moving down):
        TPs are at the opposite side of the fib levels.

    Args:
        fib_levels: Fib level dict from calculate_fib_levels().
        direction: "LONG" or "SHORT".
        tp_config: Optional custom TP config list of {"level": float, "pct": float}.

    Returns:
        List of TP dicts with 'level', 'price', 'pct' keys.
    """
    from src.config import DEFAULT_TP_CONFIG

    if tp_config is None:
        tp_config = DEFAULT_TP_CONFIG

    take_profits = []
    for tp in tp_config:
        level = tp["level"]
        if level in fib_levels:
            take_profits.append({
                "level": level,
                "price": fib_levels[level],
                "pct": tp["pct"],
            })

    return take_profits
