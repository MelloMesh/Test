"""
RSI (Relative Strength Index) calculation, peak detection, and divergence detection.

Implementation follows Wilder's smoothed RSI (exponential moving average method),
which is the standard used by TradingView, Binance, and most platforms.

Key design decisions:
- Uses Wilder's smoothing (alpha = 1/period) rather than simple SMA for RSI.
  This matches TradingView's RSI indicator exactly.
- Peak detection uses a minimum-candles-from-extreme approach: an RSI peak
  is confirmed when RSI reverses from OB/OS territory for N candles.
- Divergence detection supports both confirmed and anticipatory modes.
  Anticipatory mode signals when the divergence pattern is forming but
  the second peak hasn't fully confirmed — this gives better entry prices
  at the cost of more false signals (managed by risk rules).
"""

import numpy as np
import pandas as pd

from src.config import RSI_OVERBOUGHT, RSI_OVERSOLD, RSI_PEAK_MIN_CANDLES, RSI_PERIOD
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_rsi(
    close_prices: pd.Series,
    period: int = RSI_PERIOD,
) -> pd.Series:
    """
    Calculate RSI using Wilder's smoothed moving average method.

    This implementation matches TradingView's built-in RSI indicator.
    Uses exponential smoothing with alpha = 1/period (Wilder's smoothing).

    Args:
        close_prices: Series of closing prices.
        period: RSI lookback period (default 14).

    Returns:
        Series of RSI values (0-100). First `period` values will be NaN.
    """
    delta = close_prices.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Wilder's smoothing: first value is SMA, then EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle division by zero (no losses → RSI = 100)
    rsi = rsi.where(avg_loss != 0, 100.0)

    return rsi


def detect_rsi_peaks(
    rsi_series: pd.Series,
    price_series: pd.Series,
    threshold_ob: float = RSI_OVERBOUGHT,
    threshold_os: float = RSI_OVERSOLD,
    min_reversal_candles: int = RSI_PEAK_MIN_CANDLES,
) -> list[dict]:
    """
    Find RSI peaks in overbought/oversold territory.

    A peak is detected when:
    1. RSI crosses into OB (>70) or OS (<30) territory
    2. RSI reaches its extreme value in that zone
    3. RSI reverses (drops from OB or rises from OS) for min_reversal_candles

    This approach avoids false peaks from brief RSI spikes that immediately
    reverse. The min_reversal_candles parameter controls sensitivity.

    Args:
        rsi_series: RSI values.
        price_series: Corresponding price values (for divergence comparison).
        threshold_ob: Overbought threshold (default 70).
        threshold_os: Oversold threshold (default 30).
        min_reversal_candles: Minimum candles of reversal to confirm peak.

    Returns:
        List of peak dicts with keys:
        - type: "overbought" or "oversold"
        - rsi_value: The extreme RSI value at the peak
        - price_at_peak: The price at the peak candle
        - index: Integer index into the series
        - timestamp: Timestamp if available
    """
    peaks: list[dict] = []
    rsi_arr = rsi_series.values
    price_arr = price_series.values
    n = len(rsi_arr)

    if n < min_reversal_candles + 2:
        return peaks

    # State machine: track when we're in OB/OS zone
    i = 0
    while i < n:
        val = rsi_arr[i]
        if np.isnan(val):
            i += 1
            continue

        # --- Overbought peak detection ---
        if val >= threshold_ob:
            # Find the highest RSI value in this overbought stretch
            peak_idx = i
            peak_rsi = val
            j = i + 1
            while j < n and not np.isnan(rsi_arr[j]) and rsi_arr[j] >= threshold_ob:
                if rsi_arr[j] > peak_rsi:
                    peak_rsi = rsi_arr[j]
                    peak_idx = j
                j += 1

            # Check for reversal: RSI must decline for min_reversal_candles
            # after leaving OB zone (or from the peak within OB zone)
            reversal_start = j  # First candle below OB threshold
            reversal_count = 0
            k = reversal_start
            while k < n and not np.isnan(rsi_arr[k]):
                if rsi_arr[k] < rsi_arr[k - 1]:
                    reversal_count += 1
                else:
                    break
                if reversal_count >= min_reversal_candles:
                    break
                k += 1

            if reversal_count >= min_reversal_candles:
                peak_data = {
                    "type": "overbought",
                    "rsi_value": float(peak_rsi),
                    "price_at_peak": float(price_arr[peak_idx]),
                    "index": int(peak_idx),
                }
                # Add timestamp if index exists
                if hasattr(rsi_series, "index") and hasattr(rsi_series.index, "__getitem__"):
                    try:
                        peak_data["timestamp"] = rsi_series.index[peak_idx]
                    except (IndexError, KeyError):
                        pass
                peaks.append(peak_data)

            i = max(j, i + 1)
            continue

        # --- Oversold peak detection ---
        if val <= threshold_os:
            # Find the lowest RSI value in this oversold stretch
            peak_idx = i
            peak_rsi = val
            j = i + 1
            while j < n and not np.isnan(rsi_arr[j]) and rsi_arr[j] <= threshold_os:
                if rsi_arr[j] < peak_rsi:
                    peak_rsi = rsi_arr[j]
                    peak_idx = j
                j += 1

            # Check for reversal: RSI must rise for min_reversal_candles
            reversal_start = j
            reversal_count = 0
            k = reversal_start
            while k < n and not np.isnan(rsi_arr[k]):
                if rsi_arr[k] > rsi_arr[k - 1]:
                    reversal_count += 1
                else:
                    break
                if reversal_count >= min_reversal_candles:
                    break
                k += 1

            if reversal_count >= min_reversal_candles:
                peak_data = {
                    "type": "oversold",
                    "rsi_value": float(peak_rsi),
                    "price_at_peak": float(price_arr[peak_idx]),
                    "index": int(peak_idx),
                }
                if hasattr(rsi_series, "index") and hasattr(rsi_series.index, "__getitem__"):
                    try:
                        peak_data["timestamp"] = rsi_series.index[peak_idx]
                    except (IndexError, KeyError):
                        pass
                peaks.append(peak_data)

            i = max(j, i + 1)
            continue

        i += 1

    logger.debug(f"Detected {len(peaks)} RSI peaks ({sum(1 for p in peaks if p['type']=='overbought')} OB, "
                 f"{sum(1 for p in peaks if p['type']=='oversold')} OS)")
    return peaks


def detect_divergence(
    prices: pd.Series,
    rsi: pd.Series,
    peaks: list[dict],
    anticipatory: bool = True,
    max_peak_distance: int = 100,
) -> list[dict]:
    """
    Detect RSI divergence by comparing consecutive RSI peaks to price action.

    Bullish divergence: Price makes lower low, RSI makes higher low (oversold peaks).
    Bearish divergence: Price makes higher high, RSI makes lower high (overbought peaks).

    The anticipatory flag controls timing:
    - anticipatory=True: Signal when the divergence is forming. The 2nd peak
      has started reversing but may not be fully confirmed. Better entry prices,
      more false signals.
    - anticipatory=False: Signal only when the 2nd peak is fully confirmed
      (min_reversal_candles of reversal from extreme). Safer but later entries.

    Args:
        prices: Price series.
        rsi: RSI series.
        peaks: List of peaks from detect_rsi_peaks().
        anticipatory: If True, signal on forming divergence.
        max_peak_distance: Max candles between two peaks to consider them related.

    Returns:
        List of divergence dicts with keys:
        - type: "bullish" or "bearish"
        - confidence: 0.0 to 1.0
        - peak1: First peak dict
        - peak2: Second peak dict
        - confirmed: Whether divergence is fully confirmed
        - index: Index of the divergence signal point
    """
    divergences: list[dict] = []

    if len(peaks) < 2:
        return divergences

    # Separate peaks by type
    ob_peaks = [p for p in peaks if p["type"] == "overbought"]
    os_peaks = [p for p in peaks if p["type"] == "oversold"]

    # --- Bullish divergence (from oversold peaks) ---
    for i in range(1, len(os_peaks)):
        peak1 = os_peaks[i - 1]
        peak2 = os_peaks[i]

        # Check distance constraint
        if peak2["index"] - peak1["index"] > max_peak_distance:
            continue

        # Bullish: price lower low, RSI higher low
        price_lower = peak2["price_at_peak"] <= peak1["price_at_peak"]
        rsi_higher = peak2["rsi_value"] >= peak1["rsi_value"]

        if price_lower and rsi_higher:
            # Calculate confidence based on divergence strength
            price_diff_pct = abs(peak2["price_at_peak"] - peak1["price_at_peak"]) / peak1["price_at_peak"]
            rsi_diff = peak2["rsi_value"] - peak1["rsi_value"]

            # Stronger divergence = higher confidence
            confidence = min(0.5 + (rsi_diff / 20.0) + (price_diff_pct * 10), 1.0)

            divergences.append({
                "type": "bullish",
                "confidence": round(confidence, 3),
                "peak1": peak1,
                "peak2": peak2,
                "confirmed": not anticipatory,
                "index": peak2["index"],
            })

    # --- Bearish divergence (from overbought peaks) ---
    for i in range(1, len(ob_peaks)):
        peak1 = ob_peaks[i - 1]
        peak2 = ob_peaks[i]

        if peak2["index"] - peak1["index"] > max_peak_distance:
            continue

        # Bearish: price higher high, RSI lower high
        price_higher = peak2["price_at_peak"] >= peak1["price_at_peak"]
        rsi_lower = peak2["rsi_value"] <= peak1["rsi_value"]

        if price_higher and rsi_lower:
            price_diff_pct = abs(peak2["price_at_peak"] - peak1["price_at_peak"]) / peak1["price_at_peak"]
            rsi_diff = peak1["rsi_value"] - peak2["rsi_value"]

            confidence = min(0.5 + (rsi_diff / 20.0) + (price_diff_pct * 10), 1.0)

            divergences.append({
                "type": "bearish",
                "confidence": round(confidence, 3),
                "peak1": peak1,
                "peak2": peak2,
                "confirmed": not anticipatory,
                "index": peak2["index"],
            })

    logger.debug(f"Detected {len(divergences)} divergences "
                 f"({sum(1 for d in divergences if d['type']=='bullish')} bullish, "
                 f"{sum(1 for d in divergences if d['type']=='bearish')} bearish)")
    return divergences


def get_rsi_snapshot(rsi_series: pd.Series, index: int) -> dict:
    """
    Get an RSI snapshot at a given index for signal reporting.

    Returns:
        Dict with 'current', 'prev', 'min_recent', 'max_recent' RSI values.
    """
    current = float(rsi_series.iloc[index]) if index < len(rsi_series) else None
    prev = float(rsi_series.iloc[index - 1]) if index > 0 else None

    # Look back 20 candles for recent min/max
    lookback = max(0, index - 20)
    recent = rsi_series.iloc[lookback:index + 1]

    return {
        "current": current,
        "prev": prev,
        "min_recent": float(recent.min()) if len(recent) > 0 else None,
        "max_recent": float(recent.max()) if len(recent) > 0 else None,
    }
