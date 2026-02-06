"""
Trend and market regime indicators.

Provides:
- EMA calculation for trend direction
- ADX for trend strength / regime detection
- ATR for volatility-adjusted stops
- Multi-timeframe trend alignment check
"""

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.

    Args:
        series: Price series.
        period: EMA lookback period.

    Returns:
        EMA series.
    """
    return series.ewm(span=period, adjust=False).mean()


def calculate_atr(
    candles: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """
    Calculate Average True Range (ATR) using Wilder's smoothing.

    ATR measures volatility — used for dynamic stop placement
    and regime detection.

    Args:
        candles: OHLCV DataFrame with high, low, close columns.
        period: ATR lookback period (default 14).

    Returns:
        ATR series.
    """
    high = candles["high"]
    low = candles["low"]
    close = candles["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    return atr


def calculate_adx(
    candles: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """
    Calculate ADX (Average Directional Index) for trend strength.

    ADX values:
    - < 20: Weak trend / ranging market (good for mean-reversion/divergence)
    - 20-40: Moderate trend (divergence can work with trend)
    - > 40: Strong trend (divergence signals get run over — AVOID)

    Args:
        candles: OHLCV DataFrame.
        period: ADX lookback period (default 14).

    Returns:
        DataFrame with columns: adx, plus_di, minus_di
    """
    high = candles["high"].values
    low = candles["low"].values
    close = candles["close"].values
    n = len(candles)

    if n < period + 1:
        return pd.DataFrame(
            {"adx": [np.nan] * n, "plus_di": [np.nan] * n, "minus_di": [np.nan] * n},
            index=candles.index,
        )

    # True Range
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]

        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0

    # Wilder's smoothing for TR, +DM, -DM
    alpha = 1.0 / period
    atr_smooth = np.zeros(n)
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)

    # First value: simple sum of first `period` values
    atr_smooth[period] = np.sum(tr[1:period + 1])
    plus_dm_smooth[period] = np.sum(plus_dm[1:period + 1])
    minus_dm_smooth[period] = np.sum(minus_dm[1:period + 1])

    for i in range(period + 1, n):
        atr_smooth[i] = atr_smooth[i - 1] - (atr_smooth[i - 1] / period) + tr[i]
        plus_dm_smooth[i] = plus_dm_smooth[i - 1] - (plus_dm_smooth[i - 1] / period) + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i - 1] - (minus_dm_smooth[i - 1] / period) + minus_dm[i]

    # +DI and -DI
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    dx = np.zeros(n)

    for i in range(period, n):
        if atr_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr_smooth[i]
        else:
            plus_di[i] = 0
            minus_di[i] = 0

        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0

    # ADX: Wilder's smoothed average of DX
    adx = np.full(n, np.nan)
    adx_start = 2 * period
    if adx_start < n:
        adx[adx_start] = np.mean(dx[period:adx_start + 1])
        for i in range(adx_start + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return pd.DataFrame(
        {"adx": adx, "plus_di": plus_di, "minus_di": minus_di},
        index=candles.index,
    )


def get_trend_direction(candles: pd.DataFrame, ema_period: int = 50) -> str:
    """
    Determine trend direction from EMA slope and price position.

    Returns:
        "BULLISH", "BEARISH", or "NEUTRAL"
    """
    if len(candles) < ema_period + 5:
        return "NEUTRAL"

    close = candles["close"]
    ema = calculate_ema(close, ema_period)

    current_price = float(close.iloc[-1])
    current_ema = float(ema.iloc[-1])
    prev_ema = float(ema.iloc[-6])  # 5 bars ago

    ema_slope = (current_ema - prev_ema) / prev_ema

    if current_price > current_ema and ema_slope > 0.0005:
        return "BULLISH"
    elif current_price < current_ema and ema_slope < -0.0005:
        return "BEARISH"
    return "NEUTRAL"


def get_market_regime(candles: pd.DataFrame, adx_period: int = 14) -> dict:
    """
    Classify the current market regime.

    Returns dict with:
    - regime: "TRENDING", "RANGING", or "VOLATILE"
    - adx: Current ADX value
    - trend_direction: "BULLISH", "BEARISH", or "NEUTRAL"
    - atr_pct: ATR as percentage of price (volatility measure)
    - safe_for_divergence: Whether divergence trading is advisable
    """
    adx_data = calculate_adx(candles, adx_period)
    atr = calculate_atr(candles, adx_period)

    current_adx = float(adx_data["adx"].iloc[-1]) if not np.isnan(adx_data["adx"].iloc[-1]) else 0
    current_atr = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 0
    current_price = float(candles["close"].iloc[-1])
    atr_pct = current_atr / current_price if current_price > 0 else 0

    trend = get_trend_direction(candles)

    # Classify regime
    if current_adx > 40:
        regime = "TRENDING"
        safe = False  # Strong trend — divergences get run over
    elif current_adx < 20:
        regime = "RANGING"
        safe = True  # Range-bound — ideal for divergence/mean-reversion
    else:
        regime = "MODERATE"
        # Moderate trend: divergence OK if aligned with trend
        safe = True

    return {
        "regime": regime,
        "adx": round(current_adx, 2),
        "trend_direction": trend,
        "atr_pct": round(atr_pct, 6),
        "atr_value": round(current_atr, 2),
        "safe_for_divergence": safe,
    }


def check_mtf_alignment(
    signal_direction: str,
    htf_candles: pd.DataFrame | None,
    ema_period: int = 50,
) -> tuple[bool, str]:
    """
    Check if a signal aligns with the higher-timeframe trend.

    A LONG signal should be aligned with a BULLISH or NEUTRAL HTF trend.
    A SHORT signal should be aligned with a BEARISH or NEUTRAL HTF trend.
    Trading AGAINST the HTF trend is the #1 cause of false divergence signals.

    Args:
        signal_direction: "LONG" or "SHORT".
        htf_candles: Higher timeframe OHLCV data. If None, returns aligned (no filter).
        ema_period: EMA period for trend detection.

    Returns:
        Tuple of (is_aligned, reason).
    """
    if htf_candles is None or len(htf_candles) < ema_period + 5:
        return True, "no_htf_data"

    htf_trend = get_trend_direction(htf_candles, ema_period)

    if signal_direction == "LONG":
        if htf_trend == "BEARISH":
            return False, f"long_against_bearish_htf"
        return True, f"long_aligned_{htf_trend.lower()}"
    else:
        if htf_trend == "BULLISH":
            return False, f"short_against_bullish_htf"
        return True, f"short_aligned_{htf_trend.lower()}"
