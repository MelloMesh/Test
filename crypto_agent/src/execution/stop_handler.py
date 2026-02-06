"""
Stop loss handler — monitors for confirmation candle → move stop to breakeven.
Runs alongside the main trading loop.
"""

import pandas as pd

from src.execution.order_manager import OrderManager
from src.execution.position_tracker import PositionTracker
from src.indicators.rsi import calculate_rsi
from src.utils.logger import get_logger

logger = get_logger(__name__)


def check_and_move_stops(
    tracker: PositionTracker,
    order_manager: OrderManager,
    candles_cache: dict[str, pd.DataFrame],
) -> int:
    """
    Check all tracked positions for stop-to-breakeven confirmation.

    Called on every new candle close. For each open position:
    1. Check if RSI divergence has been confirmed by a closing candle
    2. If confirmed: move stop to breakeven immediately
    3. Log the action

    Args:
        tracker: Position tracker with all open positions.
        order_manager: For modifying stop orders.
        candles_cache: Dict mapping symbol to recent candles DataFrame.

    Returns:
        Number of stops moved to breakeven.
    """
    moved = 0

    for key, data in list(tracker.tracked.items()):
        if data["stop_moved_to_be"]:
            continue  # Already at breakeven

        pos = data["position"]
        candles = candles_cache.get(pos.symbol)

        if candles is None or len(candles) < 20:
            continue

        # Check confirmation candle
        latest = candles.iloc[-1]
        direction = pos.direction

        confirmed = False
        if direction == "LONG":
            # Green candle close + RSI rising from oversold
            if latest["close"] > latest["open"]:
                rsi = calculate_rsi(candles["close"])
                if len(rsi) >= 2:
                    current_rsi = float(rsi.iloc[-1])
                    prev_rsi = float(rsi.iloc[-2])
                    if current_rsi > prev_rsi and current_rsi > 30:
                        confirmed = True
        else:  # SHORT
            # Red candle close + RSI falling from overbought
            if latest["close"] < latest["open"]:
                rsi = calculate_rsi(candles["close"])
                if len(rsi) >= 2:
                    current_rsi = float(rsi.iloc[-1])
                    prev_rsi = float(rsi.iloc[-2])
                    if current_rsi < prev_rsi and current_rsi < 70:
                        confirmed = True

        if confirmed:
            try:
                new_stop = order_manager.move_stop_to_breakeven(
                    symbol=pos.symbol,
                    direction=direction,
                    old_order_id=data["stop_order_id"],
                    size_contracts=pos.size_contracts,
                    entry_price=pos.entry_price,
                )
                data["stop_moved_to_be"] = True
                data["stop_order_id"] = new_stop.get("id", "")
                pos.stop_moved_to_be = True
                moved += 1

                logger.info(
                    f"STOP→BE: {pos.symbol} {direction} | "
                    f"Entry={pos.entry_price:.2f} | "
                    f"Confirmation candle at {latest['timestamp']}"
                )
            except Exception as e:
                logger.error(f"Failed to move stop to BE for {pos.symbol}: {e}")

    return moved
