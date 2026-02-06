"""
Core strategy: RSI Divergence at Fibonacci Golden Pocket.

Combines WHERE (Fibonacci structure) with WHEN (RSI divergence timing)
to find high-probability reversal entries on 15m and 30m timeframes.

Entry conditions (ALL must be true):
1. Price is in the golden pocket (0.618 - 0.886 fib retracement)
2. RSI recently hit overbought/oversold and has peaked
3. RSI divergence is forming (anticipatory) or confirmed
4. Risk manager approves the trade

Confluence scoring determines 1% vs 2% risk per trade.
"""

from datetime import datetime, timezone

import pandas as pd

from src.config import (
    DEFAULT_TP_CONFIG,
    GOLDEN_POCKET_END,
    MAX_STOP_DISTANCE_PCT,
    STRATEGY_TIMEFRAMES,
    SWING_LOOKBACK,
    SWING_MIN_BARS,
)
from src.indicators.fibonacci import (
    calculate_fib_levels,
    detect_swings,
    get_fib_take_profit_prices,
    get_latest_swing_pair,
    golden_pocket_depth,
    is_in_golden_pocket,
)
from src.indicators.rsi import (
    calculate_rsi,
    detect_divergence,
    detect_rsi_peaks,
    get_rsi_snapshot,
)
from src.indicators.volume import volume_confirms_divergence
from src.strategy.signals import Signal
from src.utils.logger import get_logger

logger = get_logger(__name__)


def analyze_timeframe(
    candles: pd.DataFrame,
    symbol: str,
    timeframe: str,
    anticipatory: bool = True,
) -> list[Signal]:
    """
    Analyze a single timeframe for golden pocket + RSI divergence signals.

    Args:
        candles: OHLCV DataFrame with at least 100 candles.
        symbol: Trading pair (e.g., "BTC/USDT:USDT").
        timeframe: Candle timeframe (e.g., "15m").
        anticipatory: If True, signal on forming divergence.

    Returns:
        List of Signal objects found in the data.
    """
    if len(candles) < 50:
        logger.warning(f"Insufficient candles for {symbol} {timeframe}: {len(candles)}")
        return []

    signals: list[Signal] = []

    # Step 1: Detect swings and calculate Fibonacci levels
    swings = detect_swings(candles, lookback=SWING_LOOKBACK, min_bars=SWING_MIN_BARS)
    swing_pair = get_latest_swing_pair(swings)

    if swing_pair is None:
        logger.debug(f"No valid swing pair found for {symbol} {timeframe}")
        return []

    swing_low_dict, swing_high_dict = swing_pair
    swing_low = swing_low_dict["price"]
    swing_high = swing_high_dict["price"]

    fib_levels = calculate_fib_levels(swing_high, swing_low)
    if not fib_levels:
        return []

    logger.debug(
        f"{symbol} {timeframe} fibs: "
        f"swing {swing_low:.2f}→{swing_high:.2f}, "
        f"GP {fib_levels.get(0.618, 0):.2f}–{fib_levels.get(0.886, 0):.2f}"
    )

    # Step 2: Calculate RSI
    rsi = calculate_rsi(candles["close"])

    # Step 3: Detect RSI peaks and divergences
    peaks = detect_rsi_peaks(rsi, candles["close"])
    divergences = detect_divergence(
        candles["close"], rsi, peaks, anticipatory=anticipatory
    )

    if not divergences:
        logger.debug(f"No divergences found for {symbol} {timeframe}")
        return []

    # Step 4: For each divergence, check if price is in golden pocket
    for div in divergences:
        div_index = div["index"]
        if div_index >= len(candles):
            continue

        price_at_signal = float(candles["close"].iloc[div_index])

        if not is_in_golden_pocket(price_at_signal, fib_levels):
            logger.debug(
                f"Divergence at index {div_index} outside golden pocket "
                f"(price={price_at_signal:.2f})"
            )
            continue

        # Step 5: Determine direction and stop loss
        direction = "LONG" if div["type"] == "bullish" else "SHORT"

        # Stop at 0.886 fib level
        stop_price = fib_levels.get(GOLDEN_POCKET_END, 0)

        if direction == "LONG":
            # For longs, stop is below 0.886 (which is below the golden pocket)
            # Add a small buffer below the 0.886 level
            stop_price = stop_price * 0.999
        else:
            # For shorts, stop is above the 0.886 level of a downswing
            # The fib is drawn from high to low, so 0.886 is close to the low
            # For a short, we need the stop ABOVE the entry zone
            stop_price = fib_levels.get(0.0, swing_high) * 1.001

        # Cap stop at 5% from entry
        stop_distance = abs(price_at_signal - stop_price) / price_at_signal
        if stop_distance > MAX_STOP_DISTANCE_PCT:
            if direction == "LONG":
                stop_price = price_at_signal * (1 - MAX_STOP_DISTANCE_PCT)
            else:
                stop_price = price_at_signal * (1 + MAX_STOP_DISTANCE_PCT)

        # Step 6: Score confluence
        confidence, risk_pct, factors = score_confluence(
            candles=candles,
            div=div,
            fib_levels=fib_levels,
            price=price_at_signal,
            timeframe=timeframe,
        )

        # Step 7: Calculate take profit levels
        take_profits = get_fib_take_profit_prices(fib_levels, direction)

        # Step 8: Build RSI snapshot
        rsi_snapshot = get_rsi_snapshot(rsi, div_index)
        rsi_snapshot["peak1"] = div["peak1"]["rsi_value"]
        rsi_snapshot["peak2"] = div["peak2"]["rsi_value"]

        # Step 9: Determine timestamp
        ts = candles["timestamp"].iloc[div_index]
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        # Step 10: Build reason string
        gp_depth = golden_pocket_depth(price_at_signal, fib_levels)
        depth_str = f"{gp_depth:.0%} deep" if gp_depth is not None else ""
        reason = (
            f"{div['type'].title()} RSI div at {price_at_signal:.2f} "
            f"({depth_str} in GP), {timeframe}"
        )

        signal = Signal(
            direction=direction,
            confidence=confidence,
            risk_pct=risk_pct,
            entry_price=price_at_signal,
            stop_loss=round(stop_price, 2),
            take_profits=take_profits,
            timeframe=timeframe,
            symbol=symbol,
            fib_levels=fib_levels,
            swing_high=swing_high,
            swing_low=swing_low,
            rsi_values=rsi_snapshot,
            divergence_type=div["type"],
            confluence_factors=factors,
            timestamp=ts,
            reason=reason,
        )

        logger.info(f"SIGNAL: {signal}")
        signals.append(signal)

    return signals


def score_confluence(
    candles: pd.DataFrame,
    div: dict,
    fib_levels: dict,
    price: float,
    timeframe: str,
) -> tuple[float, float, list[str]]:
    """
    Score confluence factors to determine confidence and risk percentage.

    Base signal: confidence=0.7, risk=0.01 (1%)
    Each confluence factor adds +0.075 confidence and +0.0025 risk.
    Max: confidence=1.0, risk=0.02 (2%)

    Args:
        candles: OHLCV DataFrame.
        div: Divergence dict.
        fib_levels: Fibonacci levels dict.
        price: Current price.
        timeframe: Timeframe string.

    Returns:
        Tuple of (confidence, risk_pct, list of factor descriptions).
    """
    confidence = 0.7
    risk_pct = 0.01
    factors: list[str] = []

    # Factor 1: Volume confirmation
    if volume_confirms_divergence(candles, div["index"]):
        confidence += 0.075
        risk_pct += 0.0025
        factors.append("volume_confirmation")

    # Factor 2: Deep golden pocket (0.786 - 0.886)
    gp_depth = golden_pocket_depth(price, fib_levels)
    if gp_depth is not None and gp_depth >= 0.6:  # Deep in pocket
        confidence += 0.075
        risk_pct += 0.0025
        factors.append(f"deep_golden_pocket({gp_depth:.0%})")

    # Factor 3: Strong divergence (high confidence from divergence detection)
    if div.get("confidence", 0) >= 0.8:
        confidence += 0.075
        risk_pct += 0.0025
        factors.append("strong_divergence")

    # Factor 4: Divergence is confirmed (not just anticipatory)
    if div.get("confirmed", False):
        confidence += 0.075
        risk_pct += 0.0025
        factors.append("divergence_confirmed")

    # Cap at maximums
    confidence = min(confidence, 1.0)
    risk_pct = min(risk_pct, 0.02)

    return round(confidence, 3), round(risk_pct, 4), factors


def scan_for_signals(
    symbol: str = "BTC/USDT:USDT",
    days: int = 7,
    timeframes: list[str] | None = None,
    anticipatory: bool = True,
) -> list[Signal]:
    """
    Scan historical data for golden pocket + RSI divergence signals.

    Loads candle data from storage and analyzes each timeframe.

    Args:
        symbol: Trading pair.
        days: Days of history to analyze.
        timeframes: Timeframes to scan (default: 15m, 30m).
        anticipatory: If True, detect forming divergences.

    Returns:
        List of all signals found, sorted by timestamp.
    """
    from src.data.storage import load_candles

    if timeframes is None:
        timeframes = STRATEGY_TIMEFRAMES

    all_signals: list[Signal] = []

    for tf in timeframes:
        candles = load_candles(symbol=symbol, timeframe=tf, days=days)
        if candles.empty:
            logger.warning(f"No candle data for {symbol} {tf} — skipping")
            continue

        signals = analyze_timeframe(candles, symbol, tf, anticipatory)
        all_signals.extend(signals)

    # Sort by timestamp
    all_signals.sort(key=lambda s: s.timestamp)

    logger.info(
        f"Scan complete for {symbol}: {len(all_signals)} signals across "
        f"{', '.join(timeframes)} ({days} days)"
    )
    return all_signals
