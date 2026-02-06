"""
Volume-based indicators for confluence scoring.
OBV (On-Balance Volume) and volume confirmation for divergence signals.
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
