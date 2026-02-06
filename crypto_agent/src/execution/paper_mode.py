"""
Paper trading (mock) execution layer.
All order functions return mock responses and log "[PAPER] Would have placed...".
Used when LIVE_TRADING_ENABLED=false (default).
"""

import time
from datetime import datetime, timezone

from src.utils.logger import get_logger

logger = get_logger(__name__)

_order_counter = 0


def _next_order_id() -> str:
    global _order_counter
    _order_counter += 1
    return f"PAPER-{_order_counter:06d}"


def place_market_order(
    symbol: str,
    side: str,
    size_contracts: float,
    current_price: float,
) -> dict:
    """
    Simulate placing a market order.

    Returns:
        Mock order response matching ccxt format.
    """
    order_id = _next_order_id()
    logger.info(
        f"[PAPER] Would have placed MARKET {side} {size_contracts:.6f} {symbol} "
        f"@ ~{current_price:.2f}"
    )
    return {
        "id": order_id,
        "symbol": symbol,
        "type": "market",
        "side": side,
        "amount": size_contracts,
        "price": current_price,
        "filled": size_contracts,
        "remaining": 0,
        "status": "closed",
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
        "datetime": datetime.now(timezone.utc).isoformat(),
        "fee": {"cost": current_price * size_contracts * 0.0004, "currency": "USDT"},
    }


def place_stop_loss_order(
    symbol: str,
    side: str,
    size_contracts: float,
    stop_price: float,
) -> dict:
    """Simulate placing a stop loss order."""
    order_id = _next_order_id()
    logger.info(
        f"[PAPER] Would have placed STOP {side} {size_contracts:.6f} {symbol} "
        f"@ {stop_price:.2f}"
    )
    return {
        "id": order_id,
        "symbol": symbol,
        "type": "stop_market",
        "side": side,
        "amount": size_contracts,
        "stopPrice": stop_price,
        "status": "open",
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
    }


def place_take_profit_order(
    symbol: str,
    side: str,
    size_contracts: float,
    tp_price: float,
) -> dict:
    """Simulate placing a take profit order."""
    order_id = _next_order_id()
    logger.info(
        f"[PAPER] Would have placed TP {side} {size_contracts:.6f} {symbol} "
        f"@ {tp_price:.2f}"
    )
    return {
        "id": order_id,
        "symbol": symbol,
        "type": "take_profit_market",
        "side": side,
        "amount": size_contracts,
        "stopPrice": tp_price,
        "status": "open",
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
    }


def cancel_order(symbol: str, order_id: str) -> dict:
    """Simulate cancelling an order."""
    logger.info(f"[PAPER] Would have cancelled order {order_id} for {symbol}")
    return {
        "id": order_id,
        "symbol": symbol,
        "status": "canceled",
    }


def modify_stop_loss(
    symbol: str,
    order_id: str,
    new_stop_price: float,
    size_contracts: float,
) -> dict:
    """
    Simulate modifying a stop loss order.
    In practice: cancel old stop, place new one.
    """
    cancel_order(symbol, order_id)
    side = "sell"  # For long positions, stop is a sell
    return place_stop_loss_order(symbol, side, size_contracts, new_stop_price)
