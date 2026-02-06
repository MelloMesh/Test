"""
Trading cost simulation for backtesting.

Uses a flat round-trip percentage for simplicity.
P&L is expressed in R-multiples (risk units).
"""

from src.config import ROUND_TRIP_COST_PCT
from src.utils.logger import get_logger

logger = get_logger(__name__)


def estimate_round_trip_cost(size_usd: float) -> float:
    """
    Flat round-trip cost estimate (entry + exit fees + slippage).

    Uses ROUND_TRIP_COST_PCT from config (default 0.14%).

    Args:
        size_usd: Position size in USDT.

    Returns:
        Estimated total cost in USDT.
    """
    return size_usd * ROUND_TRIP_COST_PCT


def calculate_r_multiple(
    direction: str,
    entry_price: float,
    exit_price: float,
    stop_price: float,
) -> float:
    """
    Calculate P&L as an R-multiple (before costs).

    1R = the distance from entry to stop loss.
    A stop-loss hit returns -1.0R, take profits return positive R values.

    Args:
        direction: "LONG" or "SHORT".
        entry_price: Entry price.
        exit_price: Price at which position was closed.
        stop_price: Original stop loss price.

    Returns:
        R-multiple (e.g., -1.0 for stop hit, +2.0 for 2R winner).
    """
    risk_distance = abs(entry_price - stop_price)
    if risk_distance == 0:
        return 0.0

    if direction == "LONG":
        move = exit_price - entry_price
    else:
        move = entry_price - exit_price

    return move / risk_distance


def cost_as_r(size_usd: float, risk_amount: float) -> float:
    """
    Express the flat round-trip cost as a fraction of R.

    Args:
        size_usd: Position size in USDT.
        risk_amount: Dollar amount of 1R (risk_pct * equity).

    Returns:
        Cost in R-units (e.g., 0.07 means cost = 0.07R).
    """
    if risk_amount <= 0:
        return 0.0
    return estimate_round_trip_cost(size_usd) / risk_amount


def calculate_funding_cost(
    size_usd: float,
    funding_rate: float,
    direction: str,
    periods: int = 1,
) -> float:
    """
    Calculate funding rate cost for holding a position.

    Funding is paid/received every 8 hours on Binance Futures.

    Args:
        size_usd: Position notional value.
        funding_rate: The funding rate (e.g., 0.0001 = 0.01%).
        direction: "LONG" or "SHORT".
        periods: Number of 8-hour periods held.

    Returns:
        Funding cost (positive = you pay, negative = you receive).
    """
    if direction == "LONG":
        cost = size_usd * funding_rate * periods
    else:
        cost = -size_usd * funding_rate * periods

    return cost
