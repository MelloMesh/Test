"""
Position sizing based on risk percentage and stop distance.

The core formula:
    position_size_usd = (equity * risk_pct) / stop_distance_pct

This ensures we never risk more than risk_pct of equity on a single trade,
regardless of leverage or position size.
"""

from src.config import DEFAULT_LEVERAGE, MAX_LEVERAGE, MAX_STOP_DISTANCE_PCT
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_position_size(
    equity: float,
    risk_pct: float,
    entry_price: float,
    stop_loss_price: float,
    leverage: int = DEFAULT_LEVERAGE,
) -> dict:
    """
    Calculate position size from risk parameters and Fibonacci-based stop.

    The stop loss price comes from the 0.886 Fibonacci level (or 5% cap).
    Position size is determined so that if the stop is hit, we lose exactly
    risk_pct of equity.

    Args:
        equity: Total account equity in USDT.
        risk_pct: Risk as decimal (0.01 = 1%, 0.02 = 2%).
        entry_price: Expected entry price.
        stop_loss_price: Stop loss price (from 0.886 fib or 5% cap).
        leverage: Leverage multiplier (default from config).

    Returns:
        Dict with position sizing details:
        - size_usd: Position size in USDT
        - size_contracts: Position size in base currency
        - leverage: Applied leverage
        - margin_required: USDT margin needed
        - liquidation_price: Estimated liquidation price
        - risk_amount: Dollar amount at risk
        - risk_pct_actual: Actual risk percentage
        - stop_distance_pct: Distance from entry to stop as percentage
    """
    if equity <= 0:
        raise ValueError(f"Equity must be positive, got {equity}")
    if entry_price <= 0:
        raise ValueError(f"Entry price must be positive, got {entry_price}")
    if not (0 < risk_pct <= 0.02):
        raise ValueError(f"Risk pct must be between 0 and 0.02, got {risk_pct}")
    if leverage < 1 or leverage > MAX_LEVERAGE:
        raise ValueError(f"Leverage must be 1-{MAX_LEVERAGE}, got {leverage}")

    # Calculate stop distance
    stop_distance_pct = abs(entry_price - stop_loss_price) / entry_price

    # Enforce 5% stop cap
    if stop_distance_pct > MAX_STOP_DISTANCE_PCT:
        logger.warning(
            f"Stop distance {stop_distance_pct:.2%} exceeds {MAX_STOP_DISTANCE_PCT:.0%} cap. "
            f"Capping stop at {MAX_STOP_DISTANCE_PCT:.0%} from entry."
        )
        if entry_price > stop_loss_price:  # LONG
            stop_loss_price = entry_price * (1 - MAX_STOP_DISTANCE_PCT)
        else:  # SHORT
            stop_loss_price = entry_price * (1 + MAX_STOP_DISTANCE_PCT)
        stop_distance_pct = MAX_STOP_DISTANCE_PCT

    # Core calculation
    risk_amount = equity * risk_pct
    size_usd = risk_amount / stop_distance_pct
    margin_required = size_usd / leverage
    size_contracts = size_usd / entry_price

    # Liquidation price estimate (isolated margin)
    # For isolated margin, liquidation occurs when unrealized loss ≈ margin
    # Long: liq_price ≈ entry * (1 - 1/leverage)
    # Short: liq_price ≈ entry * (1 + 1/leverage)
    if entry_price > stop_loss_price:  # LONG
        liquidation_price = entry_price * (1 - 1 / leverage)
    else:  # SHORT
        liquidation_price = entry_price * (1 + 1 / leverage)

    result = {
        "size_usd": round(size_usd, 2),
        "size_contracts": round(size_contracts, 6),
        "leverage": leverage,
        "margin_required": round(margin_required, 2),
        "liquidation_price": round(liquidation_price, 2),
        "risk_amount": round(risk_amount, 2),
        "risk_pct_actual": risk_pct,
        "stop_distance_pct": round(stop_distance_pct, 6),
        "stop_loss_price": round(stop_loss_price, 2),
    }

    logger.info(
        f"Position sized: ${size_usd:.2f} ({size_contracts:.6f} contracts) | "
        f"risk=${risk_amount:.2f} ({risk_pct:.1%}) | "
        f"stop={stop_distance_pct:.2%} from entry | "
        f"margin=${margin_required:.2f} at {leverage}x"
    )

    return result
