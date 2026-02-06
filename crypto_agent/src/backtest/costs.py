"""
Trading cost simulation for backtesting.
Includes maker/taker fees, slippage, and funding rate costs.
"""

from src.config import MAKER_FEE, SLIPPAGE_EST, TAKER_FEE
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_entry_cost(size_usd: float, is_taker: bool = True) -> float:
    """
    Calculate the cost of opening a position.

    Args:
        size_usd: Position size in USDT.
        is_taker: True for market orders (taker fee), False for limit (maker fee).

    Returns:
        Total entry cost in USDT (fee + slippage).
    """
    fee_rate = TAKER_FEE if is_taker else MAKER_FEE
    fee = size_usd * fee_rate
    slippage = size_usd * SLIPPAGE_EST
    return fee + slippage


def calculate_exit_cost(size_usd: float, is_taker: bool = True) -> float:
    """
    Calculate the cost of closing a position.
    Same as entry cost — fees apply both ways.
    """
    fee_rate = TAKER_FEE if is_taker else MAKER_FEE
    fee = size_usd * fee_rate
    slippage = size_usd * SLIPPAGE_EST
    return fee + slippage


def calculate_round_trip_cost(size_usd: float) -> float:
    """
    Total cost for opening and closing a position (both sides taker).
    """
    return calculate_entry_cost(size_usd) + calculate_exit_cost(size_usd)


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
    # Long pays funding when rate is positive (shorts receive)
    # Short pays funding when rate is negative (longs receive)
    if direction == "LONG":
        cost = size_usd * funding_rate * periods
    else:
        cost = -size_usd * funding_rate * periods

    return cost
