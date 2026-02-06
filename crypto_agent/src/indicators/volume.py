"""
Volume-based indicators for confluence scoring.
OBV (On-Balance Volume), OBV divergence detection, and volume confirmation.
"""

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).

    OBV adds volume on up-close candles and subtracts on down-close candles.
    Rising OBV with price confirms trend; diverging OBV signals potential reversal.

    Args:
        close: Close price series.
        volume: Volume series.

    Returns:
        OBV series.
    """
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    obv = (volume * direction).cumsum()
    return obv


def detect_obv_divergence(
    candles: pd.DataFrame,
    rsi_divergence: dict,
    lookback: int = 5,
) -> bool:
    """
    Check if OBV confirms an RSI divergence (double divergence).

    Double divergence (RSI + OBV both diverging from price) has significantly
    higher win rate than RSI divergence alone.

    For bullish RSI divergence (price lower low, RSI higher low):
    - OBV should also show higher low (not making new lows with price)

    For bearish RSI divergence (price higher high, RSI lower high):
    - OBV should also show lower high (not confirming new highs)

    Args:
        candles: OHLCV DataFrame.
        rsi_divergence: Divergence dict with 'type', 'peak1', 'peak2' keys.
        lookback: Bars around each peak to find OBV extreme.

    Returns:
        True if OBV divergence confirms the RSI divergence.
    """
    if "volume" not in candles.columns:
        return False

    peak1 = rsi_divergence.get("peak1", {})
    peak2 = rsi_divergence.get("peak2", {})
    peak1_idx = peak1.get("index")
    peak2_idx = peak2.get("index")

    if peak1_idx is None or peak2_idx is None:
        return False

    if peak1_idx >= len(candles) or peak2_idx >= len(candles):
        return False

    obv = calculate_obv(candles["close"], candles["volume"])
    if len(obv) < max(peak1_idx, peak2_idx) + 1:
        return False

    div_type = rsi_divergence.get("type", "")

    # Get OBV values around each peak (use min/max in small window)
    p1_start = max(0, peak1_idx - lookback)
    p1_end = min(len(obv), peak1_idx + lookback + 1)
    p2_start = max(0, peak2_idx - lookback)
    p2_end = min(len(obv), peak2_idx + lookback + 1)

    if div_type in ("bullish", "hidden_bullish"):
        # For bullish: price made lower low but OBV should make higher low
        obv_at_peak1 = float(obv.iloc[p1_start:p1_end].min())
        obv_at_peak2 = float(obv.iloc[p2_start:p2_end].min())
        confirmed = obv_at_peak2 > obv_at_peak1
    elif div_type in ("bearish", "hidden_bearish"):
        # For bearish: price made higher high but OBV should make lower high
        obv_at_peak1 = float(obv.iloc[p1_start:p1_end].max())
        obv_at_peak2 = float(obv.iloc[p2_start:p2_end].max())
        confirmed = obv_at_peak2 < obv_at_peak1
    else:
        return False

    logger.debug(
        f"OBV divergence check ({div_type}): "
        f"OBV@peak1={obv_at_peak1:.0f}, OBV@peak2={obv_at_peak2:.0f} "
        f"→ {'CONFIRMED' if confirmed else 'NO'}"
    )
    return confirmed


def volume_confirms_divergence(
    candles: pd.DataFrame,
    divergence_index: int,
    lookback: int = 10,
) -> bool:
    """
    Check if volume supports a divergence signal.

    Volume confirmation means:
    - For bullish divergence: volume is increasing on the second low
      (more participation in the reversal attempt).
    - For bearish divergence: volume is increasing on the second high.

    A simple heuristic: average volume in the recent `lookback` candles
    around the divergence is higher than the average volume in the prior
    `lookback` candles.

    Args:
        candles: OHLCV DataFrame.
        divergence_index: Index of the divergence signal.
        lookback: Number of candles to compare.

    Returns:
        True if volume is increasing around the divergence.
    """
    if divergence_index < 2 * lookback:
        return False

    recent_start = max(0, divergence_index - lookback)
    prior_start = max(0, recent_start - lookback)

    recent_vol = candles["volume"].iloc[recent_start:divergence_index + 1].mean()
    prior_vol = candles["volume"].iloc[prior_start:recent_start].mean()

    if prior_vol == 0:
        return False

    # Volume should be at least 10% higher
    confirmed = recent_vol > prior_vol * 1.1

    logger.debug(
        f"Volume confirmation: recent_avg={recent_vol:.0f} vs prior_avg={prior_vol:.0f} "
        f"→ {'YES' if confirmed else 'NO'}"
    )
    return confirmed
